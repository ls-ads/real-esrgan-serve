// real-esrgan-serve — GPU-side Real-ESRGAN serving CLI for the iosuite
// ecosystem. See ARCHITECTURE.md for the full design rationale.
//
// Three subcommands:
//
//	upscale     — one-shot inference; subprocesses to runtime/upscaler.py
//	serve       — long-lived HTTP daemon for hot-path / batch workloads
//	fetch-model — pull verified model artefacts from GitHub Releases
//
// The Go binary is a thin orchestrator. The actual ONNX inference
// is delegated to the Python runtime helper (subprocess boundary
// keeps the Go side pure-Go cross-compilable, isolates GPU crashes,
// and avoids the CGO bridge the previous version of this repo
// carried). The binary's only runtime requirement is "a working
// Python install with onnxruntime"; we check for it and fail loudly
// if missing.
package main

import (
	"fmt"
	"os"

	"github.com/ls-ads/real-esrgan-serve/internal/upscale"
	"github.com/ls-ads/real-esrgan-serve/internal/modelfetch"
	"github.com/ls-ads/real-esrgan-serve/internal/server"
	"github.com/spf13/cobra"
)

// version is overridden at build time via -ldflags "-X main.version=..."
// (see Makefile). Defaults to "dev" so local builds are obvious.
var version = "dev"

func main() {
	root := &cobra.Command{
		Use:   "real-esrgan-serve",
		Short: "Real-ESRGAN serving CLI — local subprocess + RunPod / vast.ai providers",
		Long: `real-esrgan-serve runs Real-ESRGAN inference locally (subprocess to
a Python runtime helper) or via a configured GPU provider. It is the
GPU-side of the iosuite ecosystem; the user-facing iosuite CLI wraps
this tool. See ARCHITECTURE.md for the full design.`,
		Version:       version,
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	root.AddCommand(upscale.Command())
	root.AddCommand(server.Command())
	root.AddCommand(modelfetch.Command())

	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
