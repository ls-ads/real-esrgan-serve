// Package server implements the `serve` subcommand: a long-lived HTTP
// daemon that holds a warm onnxruntime session for hot-path / batch
// workloads.
//
// `serve` is opt-in. The default user experience is one-shot subprocess
// via `upscale`, which is the right shape for ad-hoc CLI use and for
// the embedded iosuite web UI which prefers to spawn-on-demand
// (otherwise N tools = N daemons, see iosuite's ARCHITECTURE.md).
//
// Concurrency model:
//
//	┌── Go HTTP server ──┐
//	│   POST /upscale ──┼──┐
//	│   POST /upscale ──┼──┼──> request mux (single helper, FIFO over stdin)
//	│   POST /upscale ──┼──┘
//	└────────────────────┘            │
//	                                   ▼
//	                  ┌── Python helper subprocess (stays warm) ──┐
//	                  │   stdin: jsonl frames (one job per line)  │
//	                  │   stdout: jsonl results                   │
//	                  └────────────────────────────────────────────┘
//
// One Python helper process keeps the ORT session alive. The Go server
// muxes N concurrent HTTP handlers onto the helper's stdin/stdout via
// a per-job-ID result channel populated by a single stdout reader
// goroutine. Backpressure is natural: when the helper is slow,
// requests pile up in their own goroutines waiting for their channel.
package server

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	rrt "github.com/ls-ads/real-esrgan-serve/internal/runtime"
	"github.com/spf13/cobra"
)

type opts struct {
	port          int
	bind          string
	model         string
	modelPath     string
	concurrency   int
	gpuID         int
	pythonBin     string
	runtimeScript string
}

// Command returns the Cobra command tree for `serve`.
func Command() *cobra.Command {
	o := &opts{}
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Run as a long-lived HTTP server with a warm inference session",
		Long: `Start an HTTP server that holds an onnxruntime session warm across
requests. Use this when batching many images: the first request pays
the engine warmup cost (~1–30s depending on GPU + EP), subsequent
requests run at hot-path latency.

For one-shot use, prefer 'real-esrgan-serve upscale' — same code
path, no daemon to manage.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return run(o)
		},
	}

	f := cmd.Flags()
	f.IntVarP(&o.port, "port", "p", 8311, "TCP port to bind")
	f.StringVar(&o.bind, "bind", "127.0.0.1", "Bind address (use 0.0.0.0 to expose on LAN — opt-in)")
	f.StringVar(&o.model, "model", "realesrgan-x4plus", "Model to keep warm in the session")
	f.StringVar(&o.modelPath, "model-path", "", "Absolute path to .onnx (skips manifest lookup)")
	f.IntVar(&o.concurrency, "concurrency", 1, "Max in-flight requests; default 1 per physical GPU")
	f.IntVarP(&o.gpuID, "gpu-id", "g", 0, "GPU device index (-1 = CPU)")
	f.StringVar(&o.pythonBin, "python", "", "Python interpreter (default: --python > $PYTHON > python3)")
	f.StringVar(&o.runtimeScript, "runtime", "", "Override path to runtime/upscaler.py")

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

	probeCtx, probeCancel := context.WithTimeout(context.Background(), 10*time.Second)
	if err := resolved.Probe(probeCtx); err != nil {
		probeCancel()
		return err
	}
	probeCancel()

	// Start the warm helper before we open the listener — first
	// request never pays the warmup cost.
	helper, err := startHelper(resolved, model, o.gpuID)
	if err != nil {
		return err
	}
	defer helper.Close()

	srv := &Server{helper: helper, gates: make(chan struct{}, o.concurrency)}
	mux := http.NewServeMux()
	// /super-resolution is the canonical multipart route; /upscale is
	// kept as a name-only alias for any existing callers that learned
	// the legacy path. Same handler either way.
	mux.HandleFunc("/super-resolution", srv.handleUpscale)
	mux.HandleFunc("/upscale", srv.handleUpscale)
	mux.HandleFunc("/health", srv.handleHealth)
	// /runsync is the JSON envelope shape iosuite-serve and RunPod
	// workers use. The multipart routes above stay for ad-hoc curl /
	// `real-esrgan-serve super-resolution` local mode.
	// See deploy/SCHEMA.md for the wire contract.
	mux.HandleFunc("/runsync", srv.handleRunSync)

	addr := fmt.Sprintf("%s:%d", o.bind, o.port)
	httpSrv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	// Trap signals so Ctrl-C drains gracefully + reaps the helper.
	ctx, cancel := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	go func() {
		<-ctx.Done()
		fmt.Fprintln(os.Stderr, "shutting down…")
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutCancel()
		_ = httpSrv.Shutdown(shutCtx)
	}()

	fmt.Fprintf(os.Stderr, "real-esrgan-serve serving on http://%s (model=%s gpu=%d concurrency=%d)\n",
		addr, filepath.Base(model), o.gpuID, o.concurrency)
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("http: %w", err)
	}
	return nil
}

func resolveModel(o *opts) (string, error) {
	if o.modelPath != "" {
		if _, err := os.Stat(o.modelPath); err != nil {
			return "", fmt.Errorf("--model-path %s: %w", o.modelPath, err)
		}
		return o.modelPath, nil
	}
	// Mirror upscale's lookup paths.
	for _, variant := range []string{"_fp16", "_fp32"} {
		filename := o.model + variant + ".onnx"
		paths := []string{}
		if x := os.Getenv("XDG_CACHE_HOME"); x != "" {
			paths = append(paths, filepath.Join(x, "real-esrgan-serve", "models", filename))
		}
		if home, err := os.UserHomeDir(); err == nil {
			paths = append(paths, filepath.Join(home, ".cache", "real-esrgan-serve", "models", filename))
		}
		paths = append(paths, filepath.Join("/var/cache/real-esrgan-serve/models", filename))
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				return p, nil
			}
		}
	}
	return "", fmt.Errorf(
		"model %q not cached. Run: real-esrgan-serve fetch-model --name %s",
		o.model, o.model,
	)
}

// ─────────────────────────────────────────────────────────────────────
// Persistent helper subprocess
// ─────────────────────────────────────────────────────────────────────

// helperProc wraps the long-running `upscaler.py --serve` subprocess.
// One stdin lock + one stdout reader goroutine routes results back
// to per-job-ID channels.
type helperProc struct {
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  io.ReadCloser
	stdLock sync.Mutex

	pendingMu sync.Mutex
	pending   map[string]chan helperEvent

	closed atomic.Bool
}

type helperEvent struct {
	Event  string `json:"event"`
	ID     string `json:"id,omitempty"`
	Output string `json:"output,omitempty"`
	Msg    string `json:"msg,omitempty"`
}

func startHelper(r *rrt.Resolved, model string, gpuID int) (*helperProc, error) {
	cmd := exec.Command(r.Python,
		r.Script,
		"--serve",
		"--model", model,
		"--gpu-id", strconv.Itoa(gpuID),
	)
	cmd.Stderr = os.Stderr // helper logs to our stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start helper: %w", err)
	}

	hp := &helperProc{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		pending: make(map[string]chan helperEvent),
	}

	// Reader: dispatches every JSONL frame to the matching pending channel
	go hp.readLoop()

	// Wait for the helper's "ready" event before declaring success.
	// This is what makes `serve` startup feel synchronous from the
	// outside — caller can rely on first request having warm session.
	readyCh := make(chan helperEvent, 1)
	hp.subscribe("__ready__", readyCh)
	select {
	case ev := <-readyCh:
		if ev.Event != "ready" {
			return nil, fmt.Errorf("helper sent %s before ready: %s", ev.Event, ev.Msg)
		}
	case <-time.After(120 * time.Second):
		_ = cmd.Process.Kill()
		return nil, errors.New("helper did not signal ready within 120s")
	}
	hp.unsubscribe("__ready__")

	return hp, nil
}

func (h *helperProc) readLoop() {
	scanner := bufio.NewScanner(h.stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for scanner.Scan() {
		var ev helperEvent
		if err := json.Unmarshal(scanner.Bytes(), &ev); err != nil {
			fmt.Fprintf(os.Stderr, "[helper] non-json line: %s\n", scanner.Text())
			continue
		}
		// Bootstrap: route the one-time "ready" event to a synthetic ID
		if ev.Event == "ready" {
			h.dispatch("__ready__", ev)
			continue
		}
		if ev.ID != "" {
			h.dispatch(ev.ID, ev)
		}
	}
	// EOF or scan error — helper died. Close all pending channels so
	// in-flight requests fail fast rather than hang forever.
	h.closed.Store(true)
	h.pendingMu.Lock()
	for _, ch := range h.pending {
		close(ch)
	}
	h.pending = nil
	h.pendingMu.Unlock()
}

func (h *helperProc) subscribe(id string, ch chan helperEvent) {
	h.pendingMu.Lock()
	defer h.pendingMu.Unlock()
	h.pending[id] = ch
}

func (h *helperProc) unsubscribe(id string) {
	h.pendingMu.Lock()
	defer h.pendingMu.Unlock()
	delete(h.pending, id)
}

func (h *helperProc) dispatch(id string, ev helperEvent) {
	h.pendingMu.Lock()
	ch, ok := h.pending[id]
	h.pendingMu.Unlock()
	if !ok {
		return
	}
	select {
	case ch <- ev:
	default:
		// channel full — caller already got their answer
	}
}

func (h *helperProc) Close() error {
	if h.closed.Swap(true) {
		return nil
	}
	_ = h.stdin.Close()
	if h.cmd.Process != nil {
		// Give it a chance to exit cleanly, then kill.
		done := make(chan error, 1)
		go func() { done <- h.cmd.Wait() }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			_ = h.cmd.Process.Kill()
			<-done
		}
	}
	return nil
}

// upscale sends one job to the helper and waits for the result.
func (h *helperProc) upscale(ctx context.Context, jobID, in, out string) (helperEvent, error) {
	if h.closed.Load() {
		return helperEvent{}, errors.New("helper is dead — restart the server")
	}

	ch := make(chan helperEvent, 4)
	h.subscribe(jobID, ch)
	defer h.unsubscribe(jobID)

	frame, _ := json.Marshal(map[string]string{
		"id": jobID, "input": in, "output": out,
	})
	frame = append(frame, '\n')

	h.stdLock.Lock()
	_, err := h.stdin.Write(frame)
	h.stdLock.Unlock()
	if err != nil {
		return helperEvent{}, fmt.Errorf("helper stdin: %w", err)
	}

	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				return helperEvent{}, errors.New("helper died mid-job")
			}
			switch ev.Event {
			case "done":
				return ev, nil
			case "error":
				return ev, fmt.Errorf("helper error: %s", ev.Msg)
			default:
				// progress / preprocessing / inferring — keep listening
			}
		case <-ctx.Done():
			return helperEvent{}, ctx.Err()
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// HTTP server
// ─────────────────────────────────────────────────────────────────────

// Server holds the helper + a semaphore limiting in-flight jobs.
type Server struct {
	helper *helperProc
	gates  chan struct{}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if s.helper.closed.Load() {
		http.Error(w, "helper dead", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func (s *Server) handleUpscale(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil { // 32 MB headers/forms
		http.Error(w, fmt.Sprintf("multipart: %v", err), http.StatusBadRequest)
		return
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, fmt.Sprintf("missing 'image' field: %v", err), http.StatusBadRequest)
		return
	}
	defer file.Close()

	in, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, fmt.Sprintf("read input: %v", err), http.StatusInternalServerError)
		return
	}

	outExt := r.URL.Query().Get("ext")
	if outExt == "" {
		outExt = ".jpg"
	}
	if outExt[0] != '.' {
		outExt = "." + outExt
	}

	select {
	case s.gates <- struct{}{}:
	case <-r.Context().Done():
		return
	}
	out, _, err := s.runOnePathBased(r.Context(), in, outExt)
	<-s.gates
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	switch outExt {
	case ".png":
		w.Header().Set("Content-Type", "image/png")
	case ".jpg", ".jpeg":
		w.Header().Set("Content-Type", "image/jpeg")
	case ".webp":
		w.Header().Set("Content-Type", "image/webp")
	default:
		w.Header().Set("Content-Type", "application/octet-stream")
	}
	if _, err := w.Write(out); err != nil {
		// Already wrote headers; can't change status now. Log + move on.
		fmt.Fprintf(os.Stderr, "warn: stream output: %v\n", err)
	}
}

var jobSeq uint64

// handleRunSync — JSON-envelope alias of /upscale matching the
// iosuite-serve / RunPod-worker wire contract. iosuite's
// LocalProvider posts here unchanged from what it would post to a
// RunPod endpoint, which means local mode and serverless mode are
// indistinguishable from the iosuite caller's perspective.
//
// Wire shape (request):
//
//	{"input": {
//	    "images": [{"image_base64": "..."}, ...],
//	    "output_format": "jpg" | "png" | "webp",
//	    "tile": false                  // not yet supported here; rejected if true
//	}}
//
// Response:
//
//	{"status": "COMPLETED",
//	 "output": {"outputs": [{"image_base64": "...", "exec_ms": ...,
//	                          "output_format": "..."}]}}
//
// Errors return non-2xx so iosuite can branch on status code +
// JSON error body. tile:true returns 400 explaining the limitation.
func (s *Server) handleRunSync(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	const maxBody = 25 * 1024 * 1024
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)

	type imageInput struct {
		ImageBase64 string `json:"image_base64,omitempty"`
	}
	type runSyncInput struct {
		Images        []imageInput `json:"images"`
		OutputFormat  string       `json:"output_format,omitempty"`
		Tile          bool         `json:"tile,omitempty"`
		DiscardOutput bool         `json:"discard_output,omitempty"`
	}
	type runSyncReq struct {
		Input runSyncInput `json:"input"`
	}
	type imageOutput struct {
		ImageBase64  string `json:"image_base64,omitempty"`
		ExecMS       int    `json:"exec_ms"`
		OutputFormat string `json:"output_format,omitempty"`
	}
	type runSyncResp struct {
		Status string `json:"status"`
		Output struct {
			Outputs []imageOutput `json:"outputs"`
		} `json:"output"`
	}

	var req runSyncReq
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decode JSON: %v", err), http.StatusBadRequest)
		return
	}
	if len(req.Input.Images) == 0 {
		http.Error(w, "input.images is required (non-empty array)", http.StatusBadRequest)
		return
	}
	if req.Input.Tile {
		// Tile mode lives in the RunPod worker (providers/runpod/handler.py
		// + runtime/tiling.py). The local Go server currently shells a
		// per-image helper that doesn't honour the flag. Failing loud
		// here keeps the caller from silently getting un-tiled output.
		http.Error(w, "tile=true is not supported by `real-esrgan-serve serve` — use the RunPod worker for inputs >1280²", http.StatusBadRequest)
		return
	}

	outFormat := req.Input.OutputFormat
	if outFormat == "" {
		outFormat = "jpg"
	}
	outExt := "." + outFormat

	resp := runSyncResp{Status: "COMPLETED"}
	resp.Output.Outputs = make([]imageOutput, 0, len(req.Input.Images))

	for i, img := range req.Input.Images {
		if img.ImageBase64 == "" {
			http.Error(w, fmt.Sprintf("input.images[%d].image_base64 is required", i), http.StatusBadRequest)
			return
		}
		raw, err := base64.StdEncoding.DecodeString(img.ImageBase64)
		if err != nil {
			http.Error(w, fmt.Sprintf("decode input.images[%d].image_base64: %v", i, err), http.StatusBadRequest)
			return
		}

		// Backpressure gate per image — same semantics as
		// handleUpscale's single-image path.
		select {
		case s.gates <- struct{}{}:
		case <-r.Context().Done():
			return
		}

		out, execMS, err := s.runOnePathBased(r.Context(), raw, outExt)
		<-s.gates
		if err != nil {
			http.Error(w, fmt.Sprintf("upscale image %d: %v", i, err), http.StatusInternalServerError)
			return
		}

		var b64Out string
		if !req.Input.DiscardOutput {
			b64Out = base64.StdEncoding.EncodeToString(out)
		}
		resp.Output.Outputs = append(resp.Output.Outputs, imageOutput{
			ImageBase64:  b64Out,
			ExecMS:       execMS,
			OutputFormat: outFormat,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		fmt.Fprintf(os.Stderr, "warn: encode runsync response: %v\n", err)
	}
}

// runOnePathBased stages the input bytes to a tmp dir, calls the
// helper, reads the output, and returns it. Shared by /upscale's
// multipart path and /runsync's JSON path so both produce
// byte-identical results.
func (s *Server) runOnePathBased(ctx context.Context, in []byte, outExt string) ([]byte, int, error) {
	tmpDir, err := os.MkdirTemp("", "res-job-")
	if err != nil {
		return nil, 0, fmt.Errorf("tmpdir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	inPath := filepath.Join(tmpDir, "input.bin")
	outPath := filepath.Join(tmpDir, "output"+outExt)

	if err := os.WriteFile(inPath, in, 0o644); err != nil {
		return nil, 0, fmt.Errorf("write input: %w", err)
	}

	jobID := fmt.Sprintf("%d-%d", time.Now().UnixNano(), atomic.AddUint64(&jobSeq, 1))
	jobCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	t0 := time.Now()
	if _, err := s.helper.upscale(jobCtx, jobID, inPath, outPath); err != nil {
		return nil, 0, err
	}
	out, err := os.ReadFile(outPath)
	if err != nil {
		return nil, 0, fmt.Errorf("read output: %w", err)
	}
	return out, int(time.Since(t0).Milliseconds()), nil
}
