package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var (
	inputPath  string
	outputPath string
	gpuID      int
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
					processFile(inPath, outPath, gpuID)
				}
			}
		} else {
			fmt.Printf("Input %s is a single file.\n", inputPath)
			if outputPath == "" {
				ext := filepath.Ext(inputPath)
				outputPath = strings.TrimSuffix(inputPath, ext) + "_out" + ext
			}
			processFile(inputPath, outputPath, gpuID)
		}
	},
}

func processFile(in string, out string, gpu int) {
	fmt.Printf("Processing %s -> %s on GPU %d\n", in, out, gpu)
	// TODO: Implement actual inference integration here
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
}
