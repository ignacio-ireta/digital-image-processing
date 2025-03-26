# Image Restoration and Adaptive Filtering

This project implements various image processing techniques for noise reduction and image restoration:

1. **Adaptive Filters**
   - Adaptive Noise Reduction Filter
   - Adaptive Median Filter

2. **Wiener Filter Restoration**
   - Motion blur degradation simulation
   - Image restoration using Wiener filter with different K parameters

## Implementation Details

### Adaptive Noise Reduction Filter

This filter uses local statistics (mean and variance) to adjust the filtering strength based on the noise characteristics of the local region. The filter adapts to different regions of the image, applying stronger filtering in homogeneous areas and preserving edges in high-detail areas.

### Adaptive Median Filter

This filter dynamically adjusts the window size based on local noise characteristics. It's particularly effective at removing impulse noise (salt and pepper noise) while preserving edges and details. The filter starts with a small window and increases it if necessary until reaching a maximum size.

### Motion Blur Operator

The implementation generates a motion blur degradation operator H(u,v) in the frequency domain using the formula:

```
H(u,v) = (T/π(ua+vb)) * sin[π(ua+vb)] * e^(-jπ(ua+vb))
```

where T, a, and b are parameters controlling the blur characteristics.

### Wiener Filter Restoration

The Wiener filter is implemented to restore images degraded by motion blur and additive noise. The restoration formula is:

```
F̂(u,v) = [H*(u,v)/(|H(u,v)|² + K)] * G(u,v)
```

where H* is the complex conjugate of H, |H|² is the squared magnitude, K is a parameter controlling the restoration, and G is the Fourier transform of the degraded image.

## Usage

### Running the Demo

```bash
python image_restoration.py
```

This will:
1. Apply adaptive filters to the Lena image
2. Perform Wiener filter restoration on the book cover and boy images with different K values
3. Save all results to the Output directory
4. Display comparison images

### Using Individual Functions

You can import and use individual functions from `image_restoration_consolidated.py` in your own code:

```python
from image_restoration_consolidated import adaptive_noise_reduction, adaptive_median_filter, wiener_filter_restoration

# Apply adaptive noise reduction
filtered_image = adaptive_noise_reduction(image, window_size=5, k=2.0)

# Apply adaptive median filter
median_filtered = adaptive_median_filter(image, max_window_size=7)

# Restore image using Wiener filter
restored_image = wiener_filter_restoration(degraded_image, H, K=0.01)
```

## Results

The implementation generates several output files in the Output directory:

- `lena_noise_reduced.png` - Lena image processed with adaptive noise reduction
- `lena_median_filtered.png` - Lena image processed with adaptive median filter
- `adaptive_filters_comparison.png` - Comparison of original and filtered Lena images
- `book_cover_restored_k*.png` - Book cover restored with different K values
- `boy_restored_k*.png` - Boy image restored with different K values
- `*_all_k_values.png` - Comparison of restoration with different K values
- `*_restoration_comparison.png` - Comparison of original, degraded, and best restored images

## Evaluation

The quality of restoration is evaluated using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) metrics. Higher values indicate better restoration quality.