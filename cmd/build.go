package cmd

import (
	"fmt"
	"os"
	"real-esrgan-serve/pkg/tensorrt"

	"github.com/spf13/cobra"
)

var (
	onnxPath string
)

var buildCmd = &cobra.Command{
	Use:   "build",
	Short: "Compile ONNX model to TensorRT engine",
	Run: func(cmd *cobra.Command, args []string) {
		if onnxPath == "" || enginePath == "" {
			fmt.Println("Error: --onnx and --engine flags are required")
			cmd.Help()
			os.Exit(1)
		}

		fmt.Printf("Building TensorRT engine...\n")
		fmt.Printf("ONNX Input: %s\n", onnxPath)
		fmt.Printf("Engine Output: %s\n", enginePath)
		fmt.Printf("Warning: Target shapes are hardcoded in the C++ backend for the realesrgan-x4plus model.\n\n")

		err := tensorrt.BuildEngineFromONNX(onnxPath, enginePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error building engine: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("Engine built successfully!")
	},
}

func init() {
	buildCmd.Flags().StringVarP(&onnxPath, "onnx", "x", "", "Path to the input ONNX model")
	buildCmd.Flags().StringVarP(&enginePath, "engine", "e", "", "Path to the output TensorRT engine file")
	buildCmd.MarkFlagRequired("onnx")
	buildCmd.MarkFlagRequired("engine")

	rootCmd.AddCommand(buildCmd)
}
