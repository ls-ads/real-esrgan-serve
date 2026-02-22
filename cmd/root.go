package cmd

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"real-esrgan-serve/pkg/imageutil"
	"real-esrgan-serve/pkg/tensorrt"

	"github.com/spf13/cobra"
)

var (
	inputPath       string
	outputPath      string
	gpuID           int
	enginePath      string
	port            int
	continueOnError bool
)

var rootCmd = &cobra.Command{
	Use:   "real-esrgan-serve",
	Short: "Standalone Go CLI tool bridging to TensorRT Real-ESRGAN",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Local inference mode (realesrgan-x4plus)")

		info, err := os.Stat(inputPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error accessing input path: %v\n", err)
			os.Exit(1)
		}

		// Try HTTP first
		useHTTP := isServerHealthy(port)
		var engine *tensorrt.EngineContext
		if !useHTTP {
			if enginePath == "" {
				fmt.Fprintf(os.Stderr, "Error: HTTP server not running on port %d and --engine not provided for local fallback.\n", port)
				os.Exit(1)
			}
			fmt.Printf("HTTP server not found on port %d, falling back to local TensorRT engine: %s\n", port, enginePath)
			eng, err := tensorrt.LoadEngine(enginePath, gpuID)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error loading engine: %v\n", err)
				os.Exit(1)
			}
			engine = eng
			defer engine.Free()
		} else {
			fmt.Printf("HTTP server found on port %d, routing inference tasks to server.\n", port)
		}

		if info.IsDir() {
			fmt.Printf("Input %s is a directory. Processing all images...\n", inputPath)
			if outputPath == "" {
				outputPath = filepath.Clean(inputPath) + "_out"
			}

			if err := os.MkdirAll(outputPath, 0755); err != nil {
				fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
				os.Exit(1)
			}

			files, err := os.ReadDir(inputPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading input directory: %v\n", err)
				os.Exit(1)
			}

			for _, file := range files {
				if file.IsDir() {
					continue
				}
				ext := strings.ToLower(filepath.Ext(file.Name()))
				if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" {
					inPath := filepath.Join(inputPath, file.Name())
					outPath := filepath.Join(outputPath, file.Name())
					if err := processFile(inPath, outPath, gpuID, useHTTP, engine); err != nil {
						fmt.Fprintf(os.Stderr, "Error processing %s\n", file.Name())
						if !continueOnError {
							os.Exit(1)
						}
					}
				}
			}
		} else {
			fmt.Printf("Input %s is a single file.\n", inputPath)
			if outputPath == "" {
				ext := filepath.Ext(inputPath)
				// If the extension is empty, default to .png
				if ext == "" {
					ext = ".png"
				}
				// e.g. path/to/image.jpg -> path/to/image_out.jpg
				outputPath = strings.TrimSuffix(inputPath, ext) + "_out" + ext
			}
			if err := processFile(inputPath, outputPath, gpuID, useHTTP, engine); err != nil {
				os.Exit(1)
			}
		}
	},
}

func isServerHealthy(port int) bool {
	resp, err := http.Get(fmt.Sprintf("http://localhost:%d/health", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func processFile(in string, out string, gpu int, useHTTP bool, engine *tensorrt.EngineContext) error {
	fmt.Printf("Processing %s -> %s\n", in, out)

	imgBytes, err := os.ReadFile(in)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read input file: %v\n", err)
		return err
	}

	var outputBytes []byte

	if useHTTP {
		var b bytes.Buffer
		w := multipart.NewWriter(&b)
		fw, err := w.CreateFormFile("image", filepath.Base(in))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to create form file: %v\n", err)
			return err
		}
		if _, err = io.Copy(fw, bytes.NewReader(imgBytes)); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to copy image to form: %v\n", err)
			return err
		}
		w.Close()

		ext := filepath.Ext(out)
		reqURL := fmt.Sprintf("http://localhost:%d/upscale?ext=%s", port, ext)
		req, err := http.NewRequest("POST", reqURL, &b)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to create request: %v\n", err)
			return err
		}
		req.Header.Set("Content-Type", w.FormDataContentType())

		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "HTTP request failed: %v\n", err)
			return err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			fmt.Fprintf(os.Stderr, "Server returned error %d: %s\n", resp.StatusCode, string(body))
			return fmt.Errorf("server HTTP %d", resp.StatusCode)
		}

		outputBytes, err = io.ReadAll(resp.Body)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to read server response: %v\n", err)
			return err
		}
	} else {
		// Local Engine inference
		tensor, width, height, err := imageutil.DecodeAndPreprocess(imgBytes)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Image decoding failed: %v\n", err)
			return err
		}

		outWidth := width * 4
		outHeight := height * 4
		outputBuffer := make([]float32, 3*outWidth*outHeight)

		err = engine.RunInference(tensor, outputBuffer, width, height)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Inference failed: %v\n", err)
			return err
		}

		ext := filepath.Ext(out)
		outputBytes, err = imageutil.PostprocessAndEncode(outputBuffer, outWidth, outHeight, ext)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Image encoding failed: %v\n", err)
			return err
		}
	}

	if err := os.WriteFile(out, outputBytes, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to write output file: %v\n", err)
		return err
	}
	return nil
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.Flags().StringVarP(&inputPath, "input", "i", "", "Input image path")
	rootCmd.Flags().StringVarP(&outputPath, "output", "o", "", "Output image path")
	rootCmd.Flags().IntVarP(&gpuID, "gpu-id", "g", 0, "GPU device to use")
	rootCmd.Flags().StringVarP(&enginePath, "engine", "e", "", "Path to TensorRT engine file (used if HTTP server is not running)")
	rootCmd.Flags().IntVarP(&port, "port", "p", 8080, "Port of the local HTTP server to route inference to")
	rootCmd.Flags().BoolVarP(&continueOnError, "continue-on-error", "c", false, "Continue processing batch if an individual file fails")
}
