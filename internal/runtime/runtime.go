// Package runtime locates and invokes the Python helper
// (runtime/upscaler.py) that does the actual ONNX inference.
//
// Lookup order for the helper script:
//  1. --runtime flag (explicit)
//  2. $REAL_ESRGAN_RUNTIME env var
//  3. <binary-dir>/runtime/upscaler.py (relative to the Go binary —
//     this is how Linux package + tarball installs lay things out)
//  4. /usr/share/real-esrgan-serve/runtime/upscaler.py (system install
//     path used by the Dockerfile)
//  5. <repo-root>/runtime/upscaler.py (developer convenience when
//     `go run ./cmd/real-esrgan-serve ...` from the source tree)
//
// Lookup order for the Python interpreter:
//  1. --python flag
//  2. $PYTHON env var
//  3. python3 on $PATH
//
// We deliberately do NOT auto-install onnxruntime if missing. The
// install step is the user's responsibility (or the Dockerfile's, or
// the install script's). `Probe()` checks importability and surfaces
// a helpful error so the caller can render it.
package runtime

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// Locator resolves the Python interpreter + helper script paths,
// using the lookup orders documented above.
type Locator struct {
	PythonOverride  string // value of --python (may be empty)
	ScriptOverride  string // value of --runtime (may be empty)
}

// Resolved is the output of Locate(): both paths exist + work.
type Resolved struct {
	Python string
	Script string
}

// Locate finds the Python interpreter and helper script. Returns an
// error with a human-readable explanation of which step failed and
// what the user can do about it (callers print it as-is).
func (l *Locator) Locate() (*Resolved, error) {
	py, err := l.findPython()
	if err != nil {
		return nil, err
	}
	script, err := l.findScript()
	if err != nil {
		return nil, err
	}
	return &Resolved{Python: py, Script: script}, nil
}

func (l *Locator) findPython() (string, error) {
	candidates := []string{}
	if l.PythonOverride != "" {
		candidates = append(candidates, l.PythonOverride)
	}
	if env := os.Getenv("PYTHON"); env != "" {
		candidates = append(candidates, env)
	}
	candidates = append(candidates, "python3", "python")

	for _, c := range candidates {
		// `exec.LookPath` honours absolute paths and $PATH. If `c` is
		// already absolute, this checks the file is exec'able.
		if path, err := exec.LookPath(c); err == nil {
			return path, nil
		}
	}
	return "", errors.New(
		"no python interpreter found. Tried --python, $PYTHON, python3, python on $PATH. " +
			"Install python3 + pip, then `pip install onnxruntime-gpu numpy pillow`.",
	)
}

func (l *Locator) findScript() (string, error) {
	candidates := []string{}
	if l.ScriptOverride != "" {
		candidates = append(candidates, l.ScriptOverride)
	}
	if env := os.Getenv("REAL_ESRGAN_RUNTIME"); env != "" {
		candidates = append(candidates, env)
	}

	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, "runtime", "upscaler.py"),
			filepath.Join(exeDir, "..", "share", "real-esrgan-serve", "runtime", "upscaler.py"),
		)
	}
	candidates = append(candidates,
		"/usr/share/real-esrgan-serve/runtime/upscaler.py",
		"/usr/local/share/real-esrgan-serve/runtime/upscaler.py",
		// Developer convenience: when running from the repo root.
		"runtime/upscaler.py",
	)

	for _, c := range candidates {
		if c == "" {
			continue
		}
		abs, err := filepath.Abs(c)
		if err != nil {
			continue
		}
		if info, err := os.Stat(abs); err == nil && !info.IsDir() {
			return abs, nil
		}
	}
	return "", fmt.Errorf(
		"upscaler.py not found. Looked in: %v. "+
			"Set $REAL_ESRGAN_RUNTIME or use --runtime to point at it.",
		candidates,
	)
}

// Probe runs `python -c "import onnxruntime"` to verify the helper
// will be able to load the model. Returns a clear error if not.
//
// Cheap (sub-second) and worth doing once at startup so we surface
// missing-dep errors as a separate failure mode from inference
// failures. doctor command will use this too.
func (r *Resolved) Probe(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, r.Python, "-c",
		`import onnxruntime, numpy, PIL; print("ok")`,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf(
			"python deps probe failed:\n  python: %s\n  err: %v\n  out: %s\n"+
				"Install with: %s -m pip install onnxruntime-gpu numpy pillow",
			r.Python, err, string(out), r.Python,
		)
	}
	return nil
}
