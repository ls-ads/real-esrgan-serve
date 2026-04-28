.PHONY: build clean test docker docker-runpod fmt vet prep-embed \
        artifacts manifest manifest-check docker-push deploy-runpod e2e-runpod
BIN_DIR ?= bin
VERSION ?= dev

# `go:embed` directives can only reference paths inside the package
# they live in. The manifest's source of truth is models/MANIFEST.json
# at the repo root; this target copies it into internal/modelfetch/
# so the runtime binary embeds the latest snapshot without us having
# to commit the duplicate (it's in .gitignore).
prep-embed:
	@cp models/MANIFEST.json internal/modelfetch/manifest.json

# Pure-Go build. CGO=0 here matters: the previous version of this
# repo required CGO + TensorRT system libs to compile. The rebuild is
# Go-native; all GPU work happens in the Python runtime helper
# invoked as a subprocess (CUDA EP under onnxruntime-gpu).
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
# One-stage now: export the .pth → .onnx in a deps-frozen container.
# Runs anywhere with Docker; doesn't require a GPU. CI runs this on
# tag push (.github/workflows/release.yml).
#
# `make artifacts` builds the export image and runs it, dropping the
# resulting .onnx files into build/dist/. The base+runpod images COPY
# from build/dist/ at docker-build time.
#
# Stage A's basicsr stack is brittle on modern Python — see
# build/Dockerfile.export's header for the dependency-archeology
# notes. Do NOT replace the docker-run with a local
# `python3 build/export_onnx.py` invocation unless you're prepared
# to wrestle the deps into a working set yourself.
artifacts:
	docker build -f build/Dockerfile.export -t real-esrgan-serve-export build/
	mkdir -p build/dist
	docker run --rm -v $(PWD)/build/dist:/output real-esrgan-serve-export

manifest:
	python3 build/update_manifest.py

manifest-check:
	python3 build/update_manifest.py --check

# ─── RunPod serverless deploy + cold-start smoke test ───────────────
# Deploys the runpod image to a RunPod serverless endpoint, runs a
# cold-start + warm-latency benchmark, and prints a summary. See
# build/runpod_deploy.py for the full flow + cold-start measurement
# methodology.
#
# IMAGE        — container image to deploy (must be in a registry
#                RunPod can pull, e.g. ghcr.io/...)
# GPU_CLASS    — kebab-case GPU class (rtx-4090, rtx-3090, l40s, ...)
#                mapped to a RunPod GPU pool inside the deploy script
# ENDPOINT_NAME — RunPod serverless endpoint name (create or update)
# WARMUP_JOBS  — number of warm jobs to run after the cold one
# CLEANUP      — set to 1 to delete the endpoint after the test
#
# Auth: $RUNPOD_API_KEY in env. Maintainer convenience:
#   prefix with `build/.with-iosuite-key` to source from
#   ~/Projects/iosuite.io/.env without echoing the value.
IMAGE         ?= ghcr.io/ls-ads/real-esrgan-serve:runpod-test
GPU_CLASS     ?= rtx-4090
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
