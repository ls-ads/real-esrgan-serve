// Package upscale implements the `upscale` subcommand: one-shot
// inference via subprocess to the Python runtime helper.
//
// Flow:
//  1. Validate flags + resolve input/output paths
//  2. Resolve runtime (helper script + python interpreter)
//  3. Resolve model path (--model is a name; we look it up in the
//     manifest cache, fetch on miss)
//  4. Spawn `python3 runtime/upscaler.py --image ... --out ...`
//  5. Pipe stdout (JSON events when --json-events) and stderr through
//  6. Exit with the helper's exit code
//
// The subprocess boundary is deliberate. Going through CGO to a C++
// TensorRT engine (the previous implementation) coupled the Go
// release to a specific CUDA/TensorRT ABI and forced the user to
// install nvcr.io/nvidia/tensorrt — license-tangled. A subprocess
// to Python lets us cross-compile the Go binary with `GOOS=darwin
// GOARCH=arm64 go build` and ship one tarball per platform; the
// runtime requirement (Python + onnxruntime) is what the install
// script + Dockerfile provide.
package upscale

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	rrt "github.com/ls-ads/real-esrgan-serve/internal/runtime"
	"github.com/spf13/cobra"
)

type opts struct {
	input         string
	output        string
	model         string
	gpuID         int
	scale         int
	jsonEvents    bool
	continueOnErr bool
	pythonBin     string
	runtimeScript string
	modelPath     string // override the manifest lookup; absolute path to .onnx
}

// Command returns the Cobra command tree for `upscale`.
func Command() *cobra.Command {
	o := &opts{}
	cmd := &cobra.Command{
		Use:   "upscale",
		Short: "Upscale a single image or directory via local subprocess",
		Long: `Run Real-ESRGAN inference on a single image or every image in a
directory. The CLI subprocesses to runtime/upscaler.py, which uses
onnxruntime (CUDA EP if available, CPU EP otherwise). For hot-path
workloads, prefer 'real-esrgan-serve serve' to keep the engine warm
across many requests.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return run(o)
		},
	}

	f := cmd.Flags()
	f.StringVarP(&o.input, "input", "i", "", "Input image file or directory (required)")
	f.StringVarP(&o.output, "output", "o", "", "Output path (auto-derived if omitted: <name>_4x.<ext>)")
	f.StringVar(&o.model, "model", "realesrgan-x4plus", "Model name (looked up in cache, fetched if missing)")
	f.StringVar(&o.modelPath, "model-path", "", "Absolute path to .onnx (skips manifest lookup)")
	f.IntVarP(&o.gpuID, "gpu-id", "g", 0, "GPU device index (0 = first NVIDIA GPU; -1 = CPU)")
	f.IntVar(&o.scale, "scale", 4, "Upscale factor (model-native is 4)")
	f.BoolVar(&o.jsonEvents, "json-events", false, "Emit JSON progress events to stdout (for tooling)")
	f.BoolVarP(&o.continueOnErr, "continue-on-error", "c", false, "When input is a directory, keep going on per-file failures")
	f.StringVar(&o.pythonBin, "python", "", "Python interpreter (default: --python > $PYTHON > python3)")
	f.StringVar(&o.runtimeScript, "runtime", "", "Override path to runtime/upscaler.py (default: alongside the binary)")

	_ = cmd.MarkFlagRequired("input")

	return cmd
}

func run(o *opts) error {
	loc := &rrt.Locator{
		PythonOverride: o.pythonBin,
		ScriptOverride: o.runtimeScript,
	}
	resolved, err := loc.Locate()
	if err != nil {
		return err
	}

	model, err := resolveModel(o)
	if err != nil {
		return err
	}

	info, err := os.Stat(o.input)
	if err != nil {
		return fmt.Errorf("input: %w", err)
	}

	// Trap signals so a Ctrl-C kills the helper subprocess too.
	ctx, cancel := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	if !info.IsDir() {
		out := o.output
		if out == "" {
			out = derivedOutput(o.input)
		}
		return invokeOne(ctx, resolved, model, o.input, out, o)
	}

	// Directory mode
	outDir := o.output
	if outDir == "" {
		outDir = strings.TrimRight(o.input, "/\\") + "_4x"
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", outDir, err)
	}

	entries, err := os.ReadDir(o.input)
	if err != nil {
		return fmt.Errorf("readdir %s: %w", o.input, err)
	}
	var anyErr error
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".webp":
		default:
			continue
		}
		in := filepath.Join(o.input, e.Name())
		out := filepath.Join(outDir, e.Name())
		if err := invokeOne(ctx, resolved, model, in, out, o); err != nil {
			fmt.Fprintf(os.Stderr, "  ✗ %s: %v\n", e.Name(), err)
			anyErr = err
			if !o.continueOnErr {
				return err
			}
		}
	}
	if anyErr != nil && o.continueOnErr {
		// Surface that some files failed without halting the run.
		return fmt.Errorf("one or more files failed; see stderr above")
	}
	return nil
}

// derivedOutput returns "<name>_4x.<ext>" alongside the input.
func derivedOutput(input string) string {
	dir := filepath.Dir(input)
	base := filepath.Base(input)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	if ext == "" {
		ext = ".png"
	}
	return filepath.Join(dir, stem+"_4x"+ext)
}

func resolveModel(o *opts) (string, error) {
	if o.modelPath != "" {
		if _, err := os.Stat(o.modelPath); err != nil {
			return "", fmt.Errorf("--model-path %s: %w", o.modelPath, err)
		}
		return o.modelPath, nil
	}
	// Look in the standard cache paths. We do NOT auto-fetch here —
	// fetch is an explicit `fetch-model` action so users see when
	// network I/O happens. Helpful error if not found.
	candidates := modelCacheCandidates(o.model)
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf(
		"model %q not found in cache. Run:\n"+
			"  real-esrgan-serve fetch-model --name %s --variant fp16\n"+
			"  (looked in: %s)",
		o.model, o.model, strings.Join(candidates, ", "),
	)
}

// modelCacheCandidates lists where to look for a cached `.onnx` for
// the given model name. Mirrors modelfetch's resolveDest order.
func modelCacheCandidates(name string) []string {
	out := []string{}
	for _, variant := range []string{"_fp16", "_fp32"} {
		filename := name + variant + ".onnx"
		if x := os.Getenv("XDG_CACHE_HOME"); x != "" {
			out = append(out, filepath.Join(x, "real-esrgan-serve", "models", filename))
		}
		if home, err := os.UserHomeDir(); err == nil {
			out = append(out, filepath.Join(home, ".cache", "real-esrgan-serve", "models", filename))
		}
		out = append(out, filepath.Join("/var/cache/real-esrgan-serve/models", filename))
	}
	return out
}

func invokeOne(ctx context.Context, r *rrt.Resolved, model, in, out string, o *opts) error {
	args := []string{
		r.Script,
		"--input", in,
		"--output", out,
		"--model", model,
		"--gpu-id", fmt.Sprintf("%d", o.gpuID),
	}
	if o.jsonEvents {
		args = append(args, "--json-events")
	}

	if !o.jsonEvents {
		fmt.Fprintf(os.Stderr, "→ %s\n", in)
	}

	cmd := exec.CommandContext(ctx, r.Python, args...)
	// Stderr goes straight to ours so users see Python tracebacks
	// in real time.
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}

	// Stream the helper's stdout. In JSON-events mode this is JSONL
	// the caller (iosuite CLI / web UI) parses; in human mode it's
	// already-friendly progress text we just pass through.
	stream := bufio.NewScanner(stdout)
	stream.Buffer(make([]byte, 0, 64*1024), 1<<20) // helper events stay small
	for stream.Scan() {
		fmt.Println(stream.Text())
	}
	if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
		// non-fatal: we still wait for the process and let its exit
		// status be the truth
		fmt.Fprintf(os.Stderr, "warn: stdout scan: %v\n", err)
	}

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("upscaler failed: %w", err)
	}
	if !o.jsonEvents {
		fmt.Fprintf(os.Stderr, "  ✓ %s\n", out)
	}
	return nil
}
