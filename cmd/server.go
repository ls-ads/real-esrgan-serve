package cmd

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"syscall"

	"real-esrgan-serve/pkg/server"

	"github.com/spf13/cobra"
)

const pidFile = "/tmp/realesrgan.pid"

var serverCmd = &cobra.Command{
	Use:   "server",
	Short: "HTTP server operations for real-esrgan-serve",
}

var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start HTTP server",
	Run: func(cmd *cobra.Command, args []string) {
		port, _ := cmd.Flags().GetInt("port")
		enginePath, _ := cmd.Flags().GetString("engine")
		gpuID, _ := cmd.Flags().GetInt("gpu-id")

		if enginePath == "" {
			fmt.Println("Error: --engine flag is required to start the server")
			cmd.Help()
			os.Exit(1)
		}

		// Write PID file
		pid := os.Getpid()
		if err := os.WriteFile(pidFile, []byte(strconv.Itoa(pid)), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to write PID file: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("Starting HTTP server on port %d... (PID %d)\n", port, pid)

		srv, err := server.NewServer(port, enginePath, gpuID)
		if err != nil {
			log.Fatalf("Server initialization failed: %v", err)
		}

		if err := srv.Start(); err != nil {
			log.Fatalf("Server error: %v", err)
		}
	},
}

var stopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop HTTP server",
	Run: func(cmd *cobra.Command, args []string) {
		data, err := os.ReadFile(pidFile)
		if err != nil {
			fmt.Printf("Error reading PID file (is server running?): %v\n", err)
			return
		}

		pid, err := strconv.Atoi(string(data))
		if err != nil {
			fmt.Printf("Invalid PID file content: %v\n", err)
			return
		}

		process, err := os.FindProcess(pid)
		if err != nil {
			fmt.Printf("Failed to find process: %v\n", err)
			return
		}

		if err := process.Signal(syscall.SIGTERM); err != nil {
			fmt.Printf("Failed to kill server (it may already be dead): %v\n", err)
		} else {
			fmt.Println("Server stopped.")
			os.Remove(pidFile)
		}
	},
}

func init() {
	startCmd.Flags().IntP("port", "p", 8080, "HTTP server port")
	startCmd.Flags().StringP("engine", "e", "", "Path to the TensorRT engine file (required)")
	startCmd.Flags().IntP("gpu-id", "g", 0, "GPU device ID to use for inference")

	serverCmd.AddCommand(startCmd)
	serverCmd.AddCommand(stopCmd)
	rootCmd.AddCommand(serverCmd)
}
