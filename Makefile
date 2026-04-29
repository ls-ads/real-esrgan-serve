.PHONY: build clean test test-unit test-py test-go test-live fmt vet prep-embed \
        docker-cpu docker-cuda docker-trt \
        docker-runpod-cpu docker-runpod-cuda docker-runpod-trt \
        docker-push-cpu docker-push-cuda docker-push-trt \
        artifacts artifacts-onnx artifacts-engine manifest manifest-check \
        remote-build-engine deploy-runpod e2e-runpod
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

# `make test` = unit-only sweep across both languages. Live tests are
# excluded; run `make test-live` after deploying a long-lived endpoint
# (see tests/README.md). The python tests run inside the cpu image so
# the host machine doesn't need pytest/hypothesis/Pillow installed —
# the same image is what we ship as the cpu flavor, so this also
# verifies tests work in the runtime environment.
TEST_IMAGE ?= real-esrgan-serve:cpu-dev

test: test-go test-py

test-unit: test

test-go:
	go test ./...

test-py:
	@docker image inspect $(TEST_IMAGE) >/dev/null 2>&1 || { \
	  echo "test-py requires the cpu image; run 'make docker-cpu' first."; \
	  echo "  (it pre-bakes Python + numpy + Pillow + onnxruntime so we"; \
	  echo "   only need to layer pytest+hypothesis at run time)"; \
	  exit 1; \
	}
	docker run --rm -v $(PWD):/repo -w /repo --entrypoint /bin/bash $(TEST_IMAGE) -c '\
	  pip3 install --quiet -r tests/requirements.txt 2>&1 | tail -1; \
	  python3 -m pytest tests/ -m "not live" -v --tb=short \
	'

# Live tests require an active RunPod endpoint and a real API key.
# Skips cleanly if env vars aren't set; otherwise submits real jobs
# (each test = ~one GPU-second of cost). See tests/README.md for setup.
test-live:
	@if [ -z "$$RUNPOD_API_KEY" ]; then \
	  echo "test-live requires RUNPOD_API_KEY (use build/.with-iosuite-key wrapper)."; \
	  echo "  RUNPOD_ENDPOINT_ID also required — see tests/README.md."; \
	  exit 1; \
	fi
	docker run --rm -v $(PWD):/repo -w /repo \
	  -e RUNPOD_API_KEY -e RUNPOD_ENDPOINT_ID \
	  --entrypoint /bin/bash $(TEST_IMAGE) -c '\
	  pip3 install --quiet -r tests/requirements.txt 2>&1 | tail -1; \
	  python3 -m pytest tests/ -m live -v --tb=short \
	'

fmt:
	go fmt ./...

vet:
	go vet ./...

# Three image flavors, one per execution mode. Each is a complete
# image that consumers pull as `real-esrgan-serve:<flavor>-VERSION`.
# Pick the flavor that matches the workload's needs:
#
#   cpu  — smallest (~280 MB). ORT CPU wheel, no GPU libs at all.
#   cuda — medium (~2.4 GB). ORT CUDA EP via cuDNN 9. Loads .onnx.
#   trt  — TRT runtime (~2 GB). TensorRT Python directly, .engine
#          files only. No ORT.
#
# Each flavor has a corresponding `docker-runpod-<flavor>` target
# that produces the RunPod-specific image layered on top of it.

docker-cpu:
	docker build --build-arg VERSION=$(VERSION) \
		-f Dockerfile.cpu \
		-t real-esrgan-serve:cpu-$(VERSION) .

docker-cuda:
	docker build --build-arg VERSION=$(VERSION) \
		-f Dockerfile.cuda \
		-t real-esrgan-serve:cuda-$(VERSION) .

docker-trt:
	docker build --build-arg VERSION=$(VERSION) \
		-f Dockerfile.trt \
		-t real-esrgan-serve:trt-$(VERSION) .

# RunPod serverless images — one per flavor, layered on top of the
# matching base. Set FLAVOR=cpu|cuda|trt at the command line, or use
# the convenience targets below.
docker-runpod-cpu: docker-cpu
	docker build \
		--build-arg BASE_TAG=cpu-$(VERSION) \
		-f providers/runpod/Dockerfile \
		-t real-esrgan-serve:runpod-cpu-$(VERSION) \
		.

docker-runpod-cuda: docker-cuda
	docker build \
		--build-arg BASE_TAG=cuda-$(VERSION) \
		-f providers/runpod/Dockerfile \
		-t real-esrgan-serve:runpod-cuda-$(VERSION) \
		.

docker-runpod-trt: docker-trt
	docker build \
		--build-arg BASE_TAG=trt-$(VERSION) \
		-f providers/runpod/Dockerfile \
		-t real-esrgan-serve:runpod-trt-$(VERSION) \
		.

# ─── Artefact pipeline (build/ — produces release artefacts) ─────────
# Stage A (artifacts-onnx) runs anywhere; produces .onnx weights.
# Stage B (artifacts-engine) requires the target GPU; produces a
# .engine pinned to that GPU's SM arch + the host's TRT version.
# Run them ad-hoc to add a new GPU class without cutting a release;
# CI runs the full matrix on tag push (.github/workflows/release.yml).

artifacts: artifacts-onnx artifacts-engine

# Stage A: ONNX export. Containerised because basicsr is unmaintained
# and modern PyPI breaks it — see build/Dockerfile.export's header
# for the full reasoning. Do NOT replace this with a local
# `python3 build/export_onnx.py` invocation unless you've already
# wrestled the deps into a working set yourself.
artifacts-onnx:
	docker build -f build/Dockerfile.export -t real-esrgan-serve-export build/
	mkdir -p build/dist
	docker run --rm -v $(PWD)/build/dist:/output real-esrgan-serve-export

# Stage B: TensorRT engine compile. Requires a real GPU with TRT
# Python bindings installed. `--auto-detect-gpu` reads nvidia-smi to
# bake the GPU class + SM arch into the output filename.
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
# GPU_CLASS    — kebab-case GPU class (rtx-4090, rtx-3090, l40s, ...)
#                mapped to a RunPod GPU pool inside the deploy script
# ENDPOINT_NAME — RunPod serverless endpoint name (create or update)
# WARMUP_JOBS  — number of warm jobs to run after the cold one
# CLEANUP      — set to 1 to delete the endpoint after the test
#
# Auth: $RUNPOD_API_KEY in env. Maintainer convenience:
#   prefix with `build/.with-iosuite-key` to source from
#   ~/Projects/iosuite.io/.env without echoing the value.
IMAGE         ?= ghcr.io/ls-ads/real-esrgan-serve:runpod-cuda-test
GPU_CLASS     ?= rtx-4090
ENDPOINT_NAME ?= real-esrgan-serve-test
WARMUP_JOBS   ?= 5

# Three push targets, one per flavor.
docker-push-cpu: docker-runpod-cpu
	docker tag real-esrgan-serve:runpod-cpu-$(VERSION) ghcr.io/ls-ads/real-esrgan-serve:runpod-cpu-$(VERSION)
	docker push ghcr.io/ls-ads/real-esrgan-serve:runpod-cpu-$(VERSION)
	@echo "pushed: ghcr.io/ls-ads/real-esrgan-serve:runpod-cpu-$(VERSION)"

docker-push-cuda: docker-runpod-cuda
	docker tag real-esrgan-serve:runpod-cuda-$(VERSION) ghcr.io/ls-ads/real-esrgan-serve:runpod-cuda-$(VERSION)
	docker push ghcr.io/ls-ads/real-esrgan-serve:runpod-cuda-$(VERSION)
	@echo "pushed: ghcr.io/ls-ads/real-esrgan-serve:runpod-cuda-$(VERSION)"

docker-push-trt: docker-runpod-trt
	docker tag real-esrgan-serve:runpod-trt-$(VERSION) ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-$(VERSION)
	docker push ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-$(VERSION)
	@echo "pushed: ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-$(VERSION)"

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
