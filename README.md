# Image Style Transfer Application

A powerful and flexible application for applying artistic styles to images using state-of-the-art diffusion models. Optimized for Apple Silicon (M-series) and CPUs.

## Features

- **Adjustable Style Transfer**: Control the strength of the applied style
- **Hardware Acceleration**:
  - Apple Silicon optimization using Metal (MPS)
  - NVIDIA GPU support via CUDA
  - Automatic fallback to CPU when needed
- **Interactive Controls**:
  - Adjust style strength from subtle to dramatic
  - Control quality vs. speed with the steps slider
  - Choose which hardware to use for processing
- **User-Friendly Interface**:
  - Upload, paste, or drag-and-drop images
  - Real-time feedback on processing status
  - Detailed information about system capabilities

## Recent Updates

- Fixed API response format handling in the frontend
- Improved error messaging for better debugging
- Enhanced dummy inference handling to improve startup reliability
- Added system information endpoint
- Updated Dockerfile and docker-compose.yml for better containerization


## Requirements

- Python 3.10 or higher
- macOS, Linux, or Windows
- For optimal performance:
  - Apple Silicon Mac (M1/M2/M3 series), or
  - 8+ GB RAM for CPU processing

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image_stylist.git
   cd image_stylist
   ```

2. Run the application:
   ```bash
   ./run.sh
   ```
   
3. Open your browser and navigate to:
   ```
   http://localhost:8081
   ```

The run script will automatically:
- Create a Python virtual environment
- Install the appropriate PyTorch version for your hardware
- Install all required dependencies
- Launch the application

### Additional Options

You can run the application with extra options:

```bash
# Clean up temporary files and optionally clear the model cache
./run.sh --clean

# Show help information
./run.sh --help
```

## Usage Guide

1. **Upload an Image**: 
   - Click on the upload area or drag and drop an image
   - You can also paste an image from your clipboard

2. **Describe Style**: 
   - Enter a text description of the desired style, such as:
     - "In the style of Van Gogh"
     - "Make it look like a watercolor painting"
     - "Convert to cyberpunk aesthetic"

3. **Adjust Parameters**:
   - **Style Strength**: Control how strongly the style is applied
     - Lower values (0.1-0.4) preserve more of the original image
     - Higher values (0.6-0.9) create more dramatic stylistic changes
     
   - **Quality Steps**: Balance quality vs. speed
     - Lower values (20-30) generate faster results
     - Higher values (70-100) produce higher quality images
     
   - **Processing Device**: Choose which hardware to use
     - "Auto" selects the best available option
     - Specific options for Apple Silicon, NVIDIA, or CPU

4. **Apply Style**: 
   - Click the "Apply Style" button
   - Processing time varies from 20 seconds to 2 minutes depending on hardware

## Advanced Options

### Models

The default model is `runwayml/stable-diffusion-v1-5`, which works well for most style transfers. You can also try:
- `CompVis/stable-diffusion-v1-4` (smaller, faster)
- `stabilityai/stable-diffusion-2-1-base` (higher quality)

### Using Docker

If you prefer to use Docker:

```bash
docker-compose up --build
```

The application will be available at http://localhost:8081

## Troubleshooting

### Image Quality Issues

- **Black Images**: The application now includes built-in detection and correction for black or very dark images. If you still get a black image:
  - Try running with another device (CPU instead of GPU or vice versa)
  - Reduce the style strength to 0.2-0.3
  - Try a different model or prompt

- **Subject Gets Lost**: Reduce the style strength to 0.3-0.4
- **Not Enough Style Effect**: Increase the style strength to 0.6-0.8
- **Poor Quality Results**: Increase the quality steps to 70+

### Performance Issues

- **Slow Processing on Apple Silicon**: Make sure "MPS" is selected in the device dropdown
- **Out of Memory Errors**: The application automatically resizes large images, but you might need to manually resize very large images before uploading
- **Model Download Failures**: Check your internet connection; models are downloaded automatically on first use

### System-Specific Issues

- **Apple Silicon**: Make sure you have macOS 12.3+ and a recent version of PyTorch
- **NVIDIA GPU**: Ensure you have updated NVIDIA drivers installed
- **CPU Only**: Expect slower processing; reducing quality steps can help

## License

MIT License
