package server

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"real-esrgan-serve/pkg/tensorrt"
)

// Server handles HTTP requests and manages the TensorRT engine context.
type Server struct {
	port       int
	engine     *tensorrt.EngineContext
	gpuID      int
	workerChan chan struct{} // limits concurrency to 1 for TensorRT thread safety
	mu         sync.Mutex
}

// NewServer initializes a new server and loads the TensorRT engine into VRAM.
func NewServer(port int, enginePath string, gpuID int) (*Server, error) {
	log.Printf("Loading TensorRT engine from '%s' onto GPU %d...", enginePath, gpuID)

	engine, err := tensorrt.LoadEngine(enginePath, gpuID)
	if err != nil {
		return nil, fmt.Errorf("failed to load engine: %w", err)
	}

	return &Server{
		port:       port,
		engine:     engine,
		gpuID:      gpuID,
		workerChan: make(chan struct{}, 1), // Only 1 inference process at a time
	}, nil
}

// Start launches the HTTP server and blocks.
func (s *Server) Start() error {
	defer s.engine.Free()

	mux := http.NewServeMux()
	mux.HandleFunc("/upscale", s.handleUpscale)
	mux.HandleFunc("/health", s.handleHealth)

	addr := fmt.Sprintf(":%d", s.port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	log.Printf("REST API listening on http://localhost%s", addr)
	return srv.ListenAndServe()
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func (s *Server) handleUpscale(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 1. Parse Image
	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Failed to read image from form. Key must be 'image'", http.StatusBadRequest)
		return
	}
	defer file.Close()

	imgBytes, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Failed to read image data", http.StatusInternalServerError)
		return
	}

	log.Printf("Received image %s (%d bytes) for upscaling", header.Filename, len(imgBytes))

	// Acquire concurrency token for thread-safe TensorRT execution
	select {
	case s.workerChan <- struct{}{}:
		// Worker acquired
	case <-r.Context().Done():
		http.Error(w, "Client disconnected", http.StatusRequestTimeout)
		return
	}

	defer func() {
		<-s.workerChan // Release worker token
	}()

	start := time.Now()
	// TODO: Replace this with actual image decoding -> tensor float[] serialization -> inference byte[]
	// For scoping we are stubbing the inference IO structure.

	// Stub input/output sizes
	inputBuffer := make([]float32, 10)
	outputBuffer := make([]float32, 40) // 4x for x4plus

	err = s.engine.RunInference(inputBuffer, outputBuffer, 64, 64)
	if err != nil {
		log.Printf("Inference failed: %v", err)
		http.Error(w, "Inference processing failed", http.StatusInternalServerError)
		return
	}

	log.Printf("Inference completed in %v", time.Since(start))

	// Stub: return the raw output bytes (or a placeholder image byte array)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("upscaled_image_bytes_placeholder"))
}
