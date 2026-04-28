.PHONY: build clean test docker docker-runpod fmt vet prep-embed \
        artifacts artifacts-onnx artifacts-engine manifest manifest-check \
        remote-build-engine
BIN_DIR ?= bin
VERSION ?= dev

# `go:embed` directives can only reference paths inside the package
# they live in. The manifest's source of truth is models/MANIFEST.json
# at the repo root; this target copies it into internal/modelfetch/
# so the runtime binary embeds the latest snapshot without us having
# to commit the duplicate (it's in .gitignore).
prep-embed:
	@cp models/MANIFEST.json internal/modelfetch/manifest.json

# Pure-Go build. The CGO=0 here matters: the previous version of this
# repo required CGO + TensorRT system libs to compile. The rebuild is
# Go-native, all GPU work happens in the Python runtime helper invoked
# as a subprocess.
build: prep-embed
	@mkdir -p $(BIN_DIR)
	CGO_ENABLED=0 go build \
		-trimpath \
		-ldflags "-s -w -X main.version=$(VERSION)" \
		-o $(BIN_DIR)/real-esrgan-serve \
		./cmd/real-esrgan-serve

clean:
	rm -rf $(BIN_DIR)

test:
	go test ./...

fmt:
	go fmt ./...

vet:
	go vet ./...

# Base runtime image — license-clean (CUDA EULA only, no nvcr.io/...)
docker:
	docker build --build-arg VERSION=$(VERSION) -t real-esrgan-serve:$(VERSION) .

# RunPod serverless image — layers on top of the base
docker-runpod: docker
	docker build \
		--build-arg BASE_TAG=$(VERSION) \
		-f providers/runpod/Dockerfile \
		-t real-esrgan-serve:runpod-$(VERSION) \
		.

# ─── Artefact pipeline (build/ — produces release artefacts) ─────────
# Stage A runs anywhere; Stage B requires the target GPU. CI runs both
# in a matrix on tag push (.github/workflows/release.yml). Locals run
# them ad-hoc to add a new GPU class without cutting a release.

artifacts: artifacts-onnx artifacts-engine

# Stage A runs in the dedicated export container — the basicsr
# dependency stack is brittle on modern Python, see
# build/Dockerfile.export's header for the full reasoning. Do NOT
# replace this with a local `python3 build/export_onnx.py` invocation
# unless you've already wrestled the deps into a working set
# yourself; you'll spend a day on it otherwise.
artifacts-onnx:
	docker build -f build/Dockerfile.export -t real-esrgan-serve-export build/
	mkdir -p build/dist
	docker run --rm -v $(PWD)/build/dist:/output real-esrgan-serve-export

artifacts-engine:
	cd build && python3 compile_engine.py \
		--onnx dist/realesrgan-x4plus_fp16.onnx \
		--auto-detect-gpu

manifest:
	python3 build/update_manifest.py

manifest-check:
	python3 build/update_manifest.py --check

# ─── Remote engine compile via RunPod ───────────────────────────────
# Spins up a temp RunPod GPU pod, runs `make artifacts-engine` (the
# same target as the local one above) over SSH on the pod, pulls the
# resulting .engine back, terminates the pod. The remote script
# tarballs your working tree, so uncommitted changes to scripts or
# Make targets take effect on the remote run.
#
# This means: if remote build works, local build is guaranteed to
# work — same code, just running on a different host.
#
# Set GPU_CLASS to one of: rtx-4090, l40s, a100-40, h100, ...
#
# Auth: $RUNPOD_API_KEY in env. Maintainer convenience:
#   prefix with `build/.with-iosuite-key` to source from
#   ~/Projects/iosuite.io/.env without echoing the value.
GPU_CLASS ?= rtx-4090

remote-build-engine:
	@if [ -z "$$RUNPOD_API_KEY" ]; then \
		echo "RUNPOD_API_KEY not set. Either:"; \
		echo "  export RUNPOD_API_KEY=..."; \
		echo "or use the maintainer wrapper:"; \
		echo "  build/.with-iosuite-key make remote-build-engine GPU_CLASS=$(GPU_CLASS)"; \
		exit 1; \
	fi
	python3 build/remote_build.py --gpu-class $(GPU_CLASS)

# ─── RunPod serverless deploy + cold-start smoke test ───────────────
# Deploys the runpod image to a RunPod serverless endpoint, runs a
# cold-start + warm-latency benchmark, and prints a summary. See
# build/runpod_deploy.py for the full flow + cold-start measurement
# methodology.
#
# IMAGE        — container image to deploy (must be in a registry
#                RunPod can pull, e.g. ghcr.io/...)
# ENDPOINT_NAME — RunPod serverless endpoint name (create or update)
# WARMUP_JOBS  — number of warm jobs to run after the cold one
# CLEANUP      — set to 1 to delete the endpoint after the test
IMAGE         ?= ghcr.io/ls-ads/real-esrgan-serve:runpod-test
ENDPOINT_NAME ?= real-esrgan-serve-test
WARMUP_JOBS   ?= 5

docker-push: docker docker-runpod
	docker tag real-esrgan-serve:runpod-$(VERSION) ghcr.io/ls-ads/real-esrgan-serve:runpod-$(VERSION)
	docker push ghcr.io/ls-ads/real-esrgan-serve:runpod-$(VERSION)
	@echo "pushed: ghcr.io/ls-ads/real-esrgan-serve:runpod-$(VERSION)"

deploy-runpod:
	@if [ -z "$$RUNPOD_API_KEY" ]; then \
		echo "RUNPOD_API_KEY not set. Use build/.with-iosuite-key, or:"; \
		echo "  export RUNPOD_API_KEY=..."; \
		exit 1; \
	fi
	python3 build/runpod_deploy.py \
		--image $(IMAGE) \
		--gpu-class $(GPU_CLASS) \
		--endpoint-name $(ENDPOINT_NAME) \
		--warmup-jobs $(WARMUP_JOBS) \
		$(if $(CLEANUP),--cleanup,)

# Full pipeline: build → push → deploy → cold-start smoke. Convenient
# for cutting a release-candidate run through real RunPod hardware
# before tagging.
e2e-runpod: docker-push deploy-runpod
