# Digital Image Processing

A comprehensive collection of digital image processing implementations and examples in Python, focusing on both theoretical foundations and practical applications.

## Overview

This repository contains various image processing algorithms and techniques implemented as part of a digital image processing course. The project includes implementations ranging from basic image operations to advanced processing techniques, with a special emphasis on image transformation, filtering, restoration, and feature extraction.

## Project Structure

- `00_OpenCV_and_Pillow/`: Basic image processing operations using OpenCV and Pillow
  - `simpleBlackAndWhite.py`: GUI tool for converting images to grayscale
- `01_Foundations/`: Fundamental image processing concepts and algorithms
  - `SVD.py`: Singular Value Decomposition for image compression with visualization
- `P1_Spatial_properties_and_filters/`: Spatial domain filtering and properties
  - `spatial_properties_and_filters.py`: Implementation of mean, median, maximum, and minimum filters
- `P2_Border_detection/`: Edge detection algorithms
  - `main.ipynb`: Jupyter notebook implementing Gaussian smoothing, CLAHE, and Canny edge detection
- `P3_Affine_transformations/`: Image transformation techniques
  - `affine_transformations.py`: Implementation of rotation transformation with bilinear interpolation
- `P4_Frequency_domain_filtering/`: Frequency domain analysis and filtering
  - `frequency_domain_filtering.py`: Interactive FFT-based filtering with notch filters
- `P5_Image_restoration/`: Image restoration techniques
  - `image_restoration.py`: Adaptive noise reduction, median filtering, and Wiener filter restoration
- `P6A_Segmentation/`: Image segmentation algorithms
- `P6B_Feature_extraction_and_descriptors/`: Feature extraction and description methods

## Key Features

### Basic Operations
- Grayscale conversion with GUI interface
- Image normalization and visualization

### Image Compression
- SVD-based compression with adjustable quality levels
- Compression ratio visualization

### Spatial Domain Filtering
- Mean filter implementation
- Median filter for noise reduction
- Maximum and minimum filters
- Adaptive noise reduction

### Edge Detection
- Manual and OpenCV Gaussian blur implementation
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Canny edge detection

### Image Transformations
- Rotation with custom center point
- Bilinear interpolation for smooth transformations
- Output dimension calculation

### Frequency Domain Processing
- Fast Fourier Transform (FFT) visualization
- Interactive noise point selection
- Notch filter implementation
- Symmetric point addition

### Image Restoration
- Adaptive noise reduction filter
- Adaptive median filter
- Motion blur simulation
- Wiener filter restoration with adjustable parameters
- PSNR and SSIM quality metrics

## Requirements

### Core Dependencies
- Python 3.12 or higher
- NumPy: Array operations and mathematical computations
- OpenCV (cv2): Image processing operations
- Matplotlib: Data visualization and image display
- Pillow (PIL): Image processing and file format support
- scikit-image: Advanced image processing algorithms

### GUI Dependencies
- tkinter: GUI development for desktop applications

### Additional Libraries
- os: Operating system interface
- pathlib: Object-oriented filesystem paths
- pickle: Python object serialization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ignacio-ireta/digital_image_processing.git
```

2. Install required dependencies:
```bash
pip install numpy opencv-python matplotlib pillow scikit-image
```

## Usage Examples

### Grayscale Conversion Using GUI
```python
# Run the grayscale converter GUI
python 00_OpenCV_and_Pillow/simpleBlackAndWhite.py
```

### SVD Image Compression
```python
# Run the SVD compression demo
python 01_Foundations/SVD.py
```

### Spatial Filtering
```python
# Apply spatial filters to images in the Input folder
python P1_Spatial_properties_and_filters/spatial_properties_and_filters.py
```

### Affine Transformations
```python
# Apply rotation transformation to an image
python P3_Affine_transformations/affine_transformations.py
```

### Image Restoration
```python
# Test adaptive filters and Wiener restoration
python P5_Image_restoration/image_restoration.py
```

## Input/Output Structure

Most modules follow a consistent directory structure:
- Place input images in the `Input` folder within each module directory
- Processed images will be saved to the `Output` folder
- Some modules generate reports or comparison visualizations

## Author

José Ignacio Esparza Ireta

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing, please:

1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request with a clear description of your improvements
