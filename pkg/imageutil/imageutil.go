package imageutil

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"strings"
)

// DecodeAndPreprocess reads an image byte slice, decodes it, and returns
// a flat float32 NCHW tensor formatted for Real-ESRGAN TensorRT inference.
// Pixels are normalized to [0, 1].
func DecodeAndPreprocess(imgBytes []byte) ([]float32, int, int, error) {
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to decode image: %w", err)
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// NCHW format: [batch, channel, height, width]
	// where batch = 1, channel = 3 (RGB)
	tensorSize := 3 * width * height
	tensor := make([]float32, tensorSize)

	// In NCHW, the channels are planar.
	// Index = c * (height * width) + y * width + x
	planeSize := width * height

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := img.At(bounds.Min.X+x, bounds.Min.Y+y)
			r, g, b, _ := c.RGBA()

			// RGBA() returns [0, 65535]. We need [0.0, 1.0]
			rFloat := float32(r) / 65535.0
			gFloat := float32(g) / 65535.0
			bFloat := float32(b) / 65535.0

			idxBase := y*width + x
			tensor[0*planeSize+idxBase] = rFloat // Red plane
			tensor[1*planeSize+idxBase] = gFloat // Green plane
			tensor[2*planeSize+idxBase] = bFloat // Blue plane
		}
	}

	return tensor, width, height, nil
}

// PostprocessAndEncode takes a float32 NCHW tensor from Real-ESRGAN TensorRT
// inference and encodes it directly back into a byte slice of the requested format.
// Supported extensions: .png, .jpg, .jpeg
// Pixels are denormalized from [0, 1].
func PostprocessAndEncode(tensor []float32, width int, height int, ext string) ([]byte, error) {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	planeSize := width * height

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idxBase := y*width + x

			rFloat := tensor[0*planeSize+idxBase]
			gFloat := tensor[1*planeSize+idxBase]
			bFloat := tensor[2*planeSize+idxBase]

			// Clamp to [0.0, 1.0]
			if rFloat < 0 {
				rFloat = 0
			} else if rFloat > 1 {
				rFloat = 1
			}
			if gFloat < 0 {
				gFloat = 0
			} else if gFloat > 1 {
				gFloat = 1
			}
			if bFloat < 0 {
				bFloat = 0
			} else if bFloat > 1 {
				bFloat = 1
			}

			// Convert back to 8-bit [0, 255]
			r := uint8(rFloat * 255.0)
			g := uint8(gFloat * 255.0)
			b := uint8(bFloat * 255.0)

			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	var buf bytes.Buffer
	ext = strings.ToLower(ext)

	switch ext {
	case ".jpg", ".jpeg":
		// Fast JPEG encoding with high quality
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95}); err != nil {
			return nil, fmt.Errorf("failed to encode jpeg: %w", err)
		}
	default:
		// Fallback to PNG
		if err := png.Encode(&buf, img); err != nil {
			return nil, fmt.Errorf("failed to encode png: %w", err)
		}
	}

	return buf.Bytes(), nil
}
