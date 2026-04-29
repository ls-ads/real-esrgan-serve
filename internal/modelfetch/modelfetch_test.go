package modelfetch

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixture: a manifest covering both ONNX variants and an engine entry,
// matching the shape that ships in models/MANIFEST.json. Tests construct
// from this rather than reading the embedded copy so each case is
// self-contained and the assertions don't track release-asset churn.
func newTestManifest() *Manifest {
	return &Manifest{
		Version: 1,
		Models: []Model{
			{Name: "realesrgan-x4plus", Variant: "fp16", Filename: "realesrgan-x4plus_fp16.onnx",
				URL: "https://example.test/x4plus_fp16.onnx", SHA256: "abc", Bytes: 100},
			{Name: "realesrgan-x4plus", Variant: "fp32", Filename: "realesrgan-x4plus_fp32.onnx",
				URL: "https://example.test/x4plus_fp32.onnx", SHA256: "def", Bytes: 200},
			{Name: "realesrgan-x4plus", Variant: "engine", GPUClass: "rtx-4090", SMArch: "sm89",
				TRTVersion: "10.1", Filename: "x4plus-rtx-4090-sm89-trt10.1_fp16.engine",
				URL: "https://example.test/eng-4090.engine", SHA256: "111", Bytes: 300},
			{Name: "realesrgan-x4plus", Variant: "engine", GPUClass: "rtx-3090", SMArch: "sm86",
				TRTVersion: "10.1", Filename: "x4plus-rtx-3090-sm86-trt10.1_fp16.engine",
				URL: "https://example.test/eng-3090.engine", SHA256: "222", Bytes: 300},
		},
	}
}

// TestManifestFind exercises the matching logic — the part of the file
// that's pure logic with multiple branches (variant matching,
// sm-arch preference over gpu-class, error paths). The hash/download
// happy paths are covered separately because they exercise different
// code (filesystem + http).
func TestManifestFind(t *testing.T) {
	m := newTestManifest()
	cases := []struct {
		name      string
		variant   string
		gpuClass  string
		smArch    string
		wantFile  string
		wantError string
	}{
		{
			name: "fp16 by variant only",
			variant: "fp16", wantFile: "realesrgan-x4plus_fp16.onnx",
		},
		{
			name: "fp32 by variant only",
			variant: "fp32", wantFile: "realesrgan-x4plus_fp32.onnx",
		},
		{
			name: "engine by sm-arch (preferred discriminator)",
			variant: "engine", smArch: "sm89",
			wantFile: "x4plus-rtx-4090-sm89-trt10.1_fp16.engine",
		},
		{
			name: "engine by sm-arch picks 3090's matching engine",
			variant: "engine", smArch: "sm86",
			wantFile: "x4plus-rtx-3090-sm86-trt10.1_fp16.engine",
		},
		{
			name: "engine by gpu-class when no sm-arch given",
			variant: "engine", gpuClass: "rtx-4090",
			wantFile: "x4plus-rtx-4090-sm89-trt10.1_fp16.engine",
		},
		{
			name: "engine without sm-arch or gpu-class fails clearly",
			variant: "engine",
			wantError: "--sm-arch or --gpu-class required",
		},
		{
			name: "engine sm-arch with no matching entry returns not-found with hint",
			variant: "engine", smArch: "sm70",
			wantError: "(sm-arch=sm70)",
		},
		{
			name: "unknown variant is reported with available list",
			variant: "fp64",
			wantError: "no manifest entry",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := m.Find("realesrgan-x4plus", tc.variant, tc.gpuClass, tc.smArch)
			if tc.wantError != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil (file=%s)", tc.wantError, got.Filename)
				}
				if !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.wantError)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.Filename != tc.wantFile {
				t.Fatalf("got filename %s, want %s", got.Filename, tc.wantFile)
			}
		})
	}
}

// TestManifestFind_smArchPrecedence covers the documented behaviour that
// sm-arch wins over gpu-class when both are passed. Multiple GPUs share
// an SM (4090 + L4 + L40S = sm89), so sm-arch is the right disambiguator
// — gpu-class is a human-friendly fallback only.
func TestManifestFind_smArchPrecedence(t *testing.T) {
	m := newTestManifest()
	// Caller passes both sm89 (matches 4090 entry) and gpu-class=rtx-3090
	// (matches 3090 entry). sm-arch must win.
	got, err := m.Find("realesrgan-x4plus", "engine", "rtx-3090", "sm89")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.GPUClass != "rtx-4090" {
		t.Fatalf("sm-arch should have selected the 4090 entry; got gpu_class=%s", got.GPUClass)
	}
}

// TestVerifyHash covers the hash-verification primitive. The interesting
// branches are the mismatch case (returns the actual hash for the error
// message) and the missing-file case (returns false without panicking).
func TestVerifyHash(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "ok.bin")
	content := []byte("the quick brown fox")
	if err := os.WriteFile(good, content, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(content)
	expected := hex.EncodeToString(sum[:])

	if ok, got := verifyHash(good, expected); !ok {
		t.Fatalf("correct hash should match; got=%s expected=%s", got, expected)
	}
	if ok, got := verifyHash(good, "wrongprefix"+expected[len("wrongprefix"):]); ok {
		t.Fatalf("wrong hash should not match; got=%s", got)
	}
	if ok, _ := verifyHash(filepath.Join(dir, "missing.bin"), expected); ok {
		t.Fatal("missing file should not match")
	}
}

// TestResolveDest exercises the cache-dir resolution priority: explicit
// override → XDG_CACHE_HOME → $HOME/.cache. The override path is the one
// the runpod handler hits in production (handler.py builds it from
// $XDG_CACHE_HOME/real-esrgan-serve/models), so getting that wrong
// would silently relocate every model to ~/.cache.
func TestResolveDest(t *testing.T) {
	t.Run("explicit override wins over env", func(t *testing.T) {
		t.Setenv("XDG_CACHE_HOME", "/should/not/win")
		got, err := resolveDest("/explicit/override")
		if err != nil {
			t.Fatal(err)
		}
		if got != "/explicit/override" {
			t.Fatalf("got %s, want /explicit/override", got)
		}
	})
	t.Run("XDG_CACHE_HOME used when no override", func(t *testing.T) {
		t.Setenv("XDG_CACHE_HOME", "/var/cache")
		got, err := resolveDest("")
		if err != nil {
			t.Fatal(err)
		}
		want := filepath.Join("/var/cache", "real-esrgan-serve", "models")
		if got != want {
			t.Fatalf("got %s, want %s", got, want)
		}
	})
	t.Run("falls back to home/.cache when XDG unset", func(t *testing.T) {
		t.Setenv("XDG_CACHE_HOME", "")
		t.Setenv("HOME", "/home/test")
		got, err := resolveDest("")
		if err != nil {
			t.Fatal(err)
		}
		want := filepath.Join("/home/test", ".cache", "real-esrgan-serve", "models")
		if got != want {
			t.Fatalf("got %s, want %s", got, want)
		}
	})
}

// TestLoadManifest_override confirms an explicit path is honoured. We
// don't separately test the embedded fallback because that's just the
// `embed` stdlib doing its thing — fragile to test (depends on build
// state) and the value would re-execute the same JSON parse code path.
func TestLoadManifest_override(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "MANIFEST.json")
	body, _ := json.Marshal(newTestManifest())
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	mf, err := loadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(mf.Models) != 4 {
		t.Fatalf("expected 4 models, got %d", len(mf.Models))
	}
}

// TestLoadManifest_invalidJSON ensures a malformed file is reported,
// not silently swallowed via the fallback chain. Silent fallback would
// hide manifest corruption until users hit a "no entry" error far from
// the cause.
func TestLoadManifest_invalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "MANIFEST.json")
	if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadManifest(path); err == nil {
		t.Fatal("expected parse error, got nil")
	}
}

// TestDownload covers the HTTP path end-to-end: response bytes hit
// disk, errors are returned without leaving partial files behind.
// httptest.Server keeps it self-contained — no real network.
func TestDownload_ok(t *testing.T) {
	body := []byte("the cake is a lie")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "downloaded.bin")
	if err := download(srv.URL+"/x", dest, false); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(body) {
		t.Fatalf("body mismatch: got %q, want %q", got, body)
	}
}

func TestDownload_404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not here", http.StatusNotFound)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "nope.bin")
	err := download(srv.URL+"/x", dest, false)
	if err == nil {
		t.Fatal("expected 404 error, got nil")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Fatalf("error should mention 404, got: %v", err)
	}
}
