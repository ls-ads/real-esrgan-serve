// Package modelfetch implements `fetch-model`: pull a verified ONNX
// model from GitHub Releases, check its SHA-256 against the repo's
// `models/MANIFEST.json`, and place it under the user's cache dir.
//
// Why bother with a manifest:
//
//	The previous version of this repo carried weights and pre-baked
//	per-GPU artefacts committed in-tree (hundreds of MB). That
//	bloated `git clone`, blocked Apache-2.0 redistribution (some
//	weights have separate licences), and gave us no way to ship
//	updated weights without a new release. The new shape:
//
//	  1. Weights live in GH Releases of this repo (`v0.X.Y`).
//	  2. `models/MANIFEST.json` (committed, small) maps model names →
//	     download URL + SHA-256 + license note.
//	  3. `fetch-model` reads the manifest, downloads on demand, verifies,
//	     places under XDG_CACHE_HOME/real-esrgan-serve/models.
//	  4. iosuite CLI shells out here rather than reimplementing fetch.
//
// Hash mismatch = delete partial + exit non-zero. We do not "trust on
// first use".
package modelfetch

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// Embed the manifest into the binary so `fetch-model` works without
// the repo's source tree available (which is the common case — users
// have only the Go binary). The on-disk file under models/MANIFEST.json
// is still the source of truth at build time, embedded snapshot is
// what ships.
//
//go:embed manifest.json
var embeddedManifest embed.FS

type Manifest struct {
	Version int     `json:"version"`
	Models  []Model `json:"models"`
}

type Model struct {
	Name       string `json:"name"`
	Variant    string `json:"variant"`
	GPUClass   string `json:"gpu_class,omitempty"`
	SMArch     string `json:"sm_arch,omitempty"`
	TRTVersion string `json:"trt_version,omitempty"`
	Filename   string `json:"filename"`
	URL        string `json:"url"`
	SHA256     string `json:"sha256"`
	Bytes      int64  `json:"bytes"`
	License    string `json:"license"`
	LicenseURL string `json:"license_url"`
	Notes      string `json:"notes,omitempty"`
}

type opts struct {
	name     string
	variant  string
	gpuClass string
	smArch   string
	dest     string
	manifest string
	noVerify bool
	jsonEvts bool
}

// Command returns the Cobra command tree for `fetch-model`.
func Command() *cobra.Command {
	o := &opts{}
	cmd := &cobra.Command{
		Use:   "fetch-model",
		Short: "Download a verified model artefact from GitHub Releases",
		Long: `Fetch a model named in models/MANIFEST.json. Downloads from the
release URL listed in the manifest, verifies SHA-256 before placing,
and caches under $XDG_CACHE_HOME/real-esrgan-serve/models (or
~/.cache/real-esrgan-serve/models on systems without XDG).

Variants:
  --variant fp16    smaller, faster; default
  --variant fp32    higher precision; baseline
  --variant engine  TensorRT-compiled engine for a specific GPU. Pass
                    one of --gpu-class or --sm-arch to disambiguate.
                    --sm-arch matches more reliably (an RTX 4090 and
                    an L40S share sm89 and the same engine works on
                    both); --gpu-class is the human-friendly form.

If a model file is already present and matches the manifest hash,
fetch-model returns immediately without re-downloading.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return run(o)
		},
	}

	f := cmd.Flags()
	f.StringVar(&o.name, "name", "realesrgan-x4plus", "Model name as listed in MANIFEST.json")
	f.StringVar(&o.variant, "variant", "fp16", "Variant: fp16 | fp32 | engine")
	f.StringVar(&o.gpuClass, "gpu-class", "", "GPU class for engine variant (e.g. rtx-4090, a40)")
	f.StringVar(&o.smArch, "sm-arch", "", "SM compute capability for engine variant (e.g. sm89). Preferred over --gpu-class because one engine works for every GPU sharing the SM.")
	f.StringVar(&o.dest, "dest", "", "Override cache destination (default: XDG cache dir)")
	f.StringVar(&o.manifest, "manifest", "", "Override manifest path (default: built-in / repo-relative)")
	f.BoolVar(&o.noVerify, "no-verify", false, "Skip SHA-256 verification — DANGEROUS, dev only")
	f.BoolVar(&o.jsonEvts, "json-events", false, "Emit progress as JSON events on stdout")

	return cmd
}

func run(o *opts) error {
	mf, err := loadManifest(o.manifest)
	if err != nil {
		return fmt.Errorf("manifest: %w", err)
	}

	entry, err := mf.Find(o.name, o.variant, o.gpuClass, o.smArch)
	if err != nil {
		return err
	}

	dest, err := resolveDest(o.dest)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return fmt.Errorf("mkdir cache dir: %w", err)
	}
	target := filepath.Join(dest, entry.Filename)

	// Already cached + verified? Bail early — the common case.
	if ok, _ := verifyHash(target, entry.SHA256); ok {
		emit(o.jsonEvts, "cached", map[string]any{
			"path":   target,
			"sha256": entry.SHA256,
			"bytes":  entry.Bytes,
		})
		fmt.Fprintf(os.Stderr, "already cached + verified: %s\n", target)
		return nil
	}

	emit(o.jsonEvts, "downloading", map[string]any{
		"url":      entry.URL,
		"dest":     target,
		"expected": entry.SHA256,
	})
	fmt.Fprintf(os.Stderr, "fetching %s\n  → %s\n", entry.URL, target)

	tmp := target + ".part"
	defer os.Remove(tmp) // safe even when rename succeeds

	if err := download(entry.URL, tmp, o.jsonEvts); err != nil {
		return fmt.Errorf("download: %w", err)
	}

	if !o.noVerify {
		if entry.SHA256 == "" || strings.HasPrefix(entry.SHA256, "REPLACE_") {
			return fmt.Errorf("manifest hash for %s/%s is a placeholder (%q) — refusing to declare verified",
				entry.Name, entry.Variant, entry.SHA256)
		}
		ok, gotSum := verifyHash(tmp, entry.SHA256)
		if !ok {
			os.Remove(tmp)
			return fmt.Errorf("sha256 mismatch — got %s, manifest says %s. Partial file deleted.",
				gotSum, entry.SHA256)
		}
	} else {
		fmt.Fprintln(os.Stderr, "WARNING: --no-verify — skipping SHA-256 check.")
	}

	if err := os.Rename(tmp, target); err != nil {
		return fmt.Errorf("rename %s -> %s: %w", tmp, target, err)
	}

	emit(o.jsonEvts, "done", map[string]any{
		"path":   target,
		"sha256": entry.SHA256,
		"bytes":  entry.Bytes,
	})
	fmt.Fprintf(os.Stderr, "done: %s\n", target)
	return nil
}

// loadManifest tries override path → repo-relative (./models/MANIFEST.json)
// → embedded copy. Embedded is the runtime fallback when the binary
// runs detached from the source tree.
func loadManifest(override string) (*Manifest, error) {
	candidates := []string{}
	if override != "" {
		candidates = append(candidates, override)
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates,
			filepath.Join(filepath.Dir(exe), "models", "MANIFEST.json"),
			filepath.Join(filepath.Dir(exe), "..", "share", "real-esrgan-serve", "models", "MANIFEST.json"),
		)
	}
	candidates = append(candidates, "models/MANIFEST.json")

	for _, p := range candidates {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var m Manifest
		if err := json.Unmarshal(b, &m); err != nil {
			return nil, fmt.Errorf("parse %s: %w", p, err)
		}
		return &m, nil
	}

	// Embedded fallback
	b, err := embeddedManifest.ReadFile("manifest.json")
	if err != nil {
		return nil, errors.New("no manifest found on disk and no embedded copy")
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("parse embedded manifest: %w", err)
	}
	return &m, nil
}

// Find picks the manifest entry matching name+variant. For the
// "engine" variant, gpuClass OR smArch must match too. smArch is
// the preferred discriminator because multiple GPUs share an SM
// (RTX 4090 + L40S + L4 are all sm89 → one engine works for all).
func (m *Manifest) Find(name, variant, gpuClass, smArch string) (*Model, error) {
	for i := range m.Models {
		e := &m.Models[i]
		if e.Name != name || e.Variant != variant {
			continue
		}
		if variant == "engine" {
			if smArch == "" && gpuClass == "" {
				return nil, fmt.Errorf("--sm-arch or --gpu-class required when --variant engine")
			}
			if smArch != "" && e.SMArch != smArch {
				continue
			}
			if smArch == "" && gpuClass != "" && e.GPUClass != gpuClass {
				continue
			}
		}
		return e, nil
	}
	hint := ""
	if variant == "engine" {
		if smArch != "" {
			hint = fmt.Sprintf(" (sm-arch=%s)", smArch)
		} else if gpuClass != "" {
			hint = fmt.Sprintf(" (gpu-class=%s)", gpuClass)
		}
	}
	return nil, fmt.Errorf("no manifest entry for name=%s variant=%s%s — available: %s",
		name, variant, hint, m.summarise())
}

func (m *Manifest) summarise() string {
	var sb strings.Builder
	for i, e := range m.Models {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(e.Name)
		sb.WriteString("/")
		sb.WriteString(e.Variant)
		if e.SMArch != "" {
			sb.WriteString("@")
			sb.WriteString(e.SMArch)
		} else if e.GPUClass != "" {
			sb.WriteString("@")
			sb.WriteString(e.GPUClass)
		}
	}
	return sb.String()
}

// resolveDest returns the cache directory. Order: --dest flag,
// $XDG_CACHE_HOME/real-esrgan-serve/models, ~/.cache/real-esrgan-serve/models.
func resolveDest(override string) (string, error) {
	if override != "" {
		return override, nil
	}
	if x := os.Getenv("XDG_CACHE_HOME"); x != "" {
		return filepath.Join(x, "real-esrgan-serve", "models"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("locate home dir: %w", err)
	}
	return filepath.Join(home, ".cache", "real-esrgan-serve", "models"), nil
}

func download(url, dest string, jsonEvts bool) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", fmt.Sprintf(
		"real-esrgan-serve/dev (Go %s; %s/%s)",
		runtime.Version(), runtime.GOOS, runtime.GOARCH,
	))

	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<10))
		return fmt.Errorf("http %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	total := resp.ContentLength
	pw := &progressWriter{
		total:    total,
		jsonEvts: jsonEvts,
		started:  time.Now(),
		nextTick: time.Now().Add(2 * time.Second),
	}
	if _, err := io.Copy(io.MultiWriter(out, pw), resp.Body); err != nil {
		return err
	}
	pw.finish()
	return nil
}

type progressWriter struct {
	total    int64
	written  int64
	jsonEvts bool
	started  time.Time
	nextTick time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n := len(b)
	p.written += int64(n)
	if time.Now().After(p.nextTick) {
		p.tick()
		p.nextTick = time.Now().Add(2 * time.Second)
	}
	return n, nil
}

func (p *progressWriter) tick() {
	if p.jsonEvts {
		emit(true, "progress", map[string]any{
			"bytes":   p.written,
			"total":   p.total,
			"elapsed": time.Since(p.started).Seconds(),
		})
		return
	}
	if p.total > 0 {
		fmt.Fprintf(os.Stderr, "  %s / %s  (%.1f %%)\n",
			human(p.written), human(p.total),
			100*float64(p.written)/float64(p.total))
	} else {
		fmt.Fprintf(os.Stderr, "  %s\n", human(p.written))
	}
}

func (p *progressWriter) finish() {
	if p.jsonEvts {
		emit(true, "downloaded", map[string]any{
			"bytes":   p.written,
			"elapsed": time.Since(p.started).Seconds(),
		})
	}
}

func verifyHash(path, expected string) (bool, string) {
	f, err := os.Open(path)
	if err != nil {
		return false, ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, ""
	}
	got := hex.EncodeToString(h.Sum(nil))
	return got == expected, got
}

func emit(on bool, event string, fields map[string]any) {
	if !on {
		return
	}
	fields["event"] = event
	b, _ := json.Marshal(fields)
	fmt.Println(string(b))
}

func human(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n2 := n / unit; n2 >= unit; n2 /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}
