import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, 'Input')
output_dir = os.path.join(script_dir, 'Output')

os.makedirs(output_dir, exist_ok=True)

def adaptive_noise_reduction(image, window_size=5, k=2.0):
    output = np.zeros_like(image, dtype=np.float64)
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    global_mean = np.mean(image)
    global_variance = np.var(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+window_size, j:j+window_size]
            local_mean = np.mean(window)
            local_variance = np.var(window)
            
            if local_variance > 0:
                factor = 1 - (k * global_variance) / (local_variance + 0.000001)
                factor = max(0, factor)
                output[i, j] = local_mean + factor * (padded_image[i+pad_size, j+pad_size] - local_mean)
            else:
                output[i, j] = local_mean
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def adaptive_median_filter(image, max_window_size=7):
    output = np.zeros_like(image)
    pad_size = max_window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window_size = 3
            center_value = padded_image[i+pad_size, j+pad_size]
            
            while window_size <= max_window_size:
                half_window = window_size // 2
                window = padded_image[i+pad_size-half_window:i+pad_size+half_window+1,
                                     j+pad_size-half_window:j+pad_size+half_window+1]
                
                window_min = np.min(window)
                window_max = np.max(window)
                window_median = np.median(window)
                
                if window_min < window_median < window_max:
                    if window_min < center_value < window_max:
                        output[i, j] = center_value
                    else:
                        output[i, j] = window_median
                    break
                else:
                    window_size += 2
                    
                    if window_size > max_window_size:
                        output[i, j] = window_median
    
    return output

def generate_motion_blur_operator(shape, a=0.1, b=0.1, T=1.0):
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    freq_term = np.pi * (u_grid * a + v_grid * b)
    H = np.zeros((M, N), dtype=np.complex128)
    epsilon = 1e-10
    mask = np.abs(freq_term) > epsilon
    H[mask] = T * np.sin(freq_term[mask]) / freq_term[mask] * np.exp(-1j * freq_term[mask])
    H[~mask] = T
    return H

def degrade_image(image, H, noise_variance=0.001):
    normalized_image = image.astype(np.float64) / 255.0
    F = np.fft.fftshift(np.fft.fft2(normalized_image))
    G = F * H
    blurred_image = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    noise = np.random.normal(0, np.sqrt(noise_variance), blurred_image.shape)
    degraded_image = blurred_image + noise
    degraded_image = np.clip(degraded_image, 0, 1)
    return degraded_image

def wiener_filter_restoration(degraded_image, H, K):
    G = np.fft.fftshift(np.fft.fft2(degraded_image))
    H_conj = np.conjugate(H)
    H_mag_squared = np.abs(H) ** 2
    F_hat = H_conj / (H_mag_squared + K) * G
    restored_image = np.real(np.fft.ifft2(np.fft.ifftshift(F_hat)))
    restored_image = np.clip(restored_image, 0, 1)
    return restored_image

def evaluate_restoration(original, restored):
    if original.max() > 1.0:
        original = original.astype(np.float64) / 255.0
    if restored.max() > 1.0:
        restored = restored.astype(np.float64) / 255.0
    
    psnr_value = psnr(original, restored)
    ssim_value = ssim(original, restored, data_range=1.0)
    
    return psnr_value, ssim_value

def display_images(images, titles, figsize=(15, 10)):
    plt.close('all')
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        if image.max() <= 1.0:
            axes[i].imshow(image, cmap='gray')
        else:
            axes[i].imshow(image, cmap='gray', vmin=0, vmax=255)
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_adaptive_filters():
    print("Testing Adaptive Filters...")
    lena_path = os.path.join(input_dir, 'Lena.png')
    lena = cv2.imread(lena_path, cv2.IMREAD_GRAYSCALE)
    lena_noise_reduced = adaptive_noise_reduction(lena, window_size=5, k=2.0)
    lena_median_filtered = adaptive_median_filter(lena, max_window_size=7)
    
    cv2.imwrite(os.path.join(output_dir, 'lena_noise_reduced.png'), lena_noise_reduced)
    cv2.imwrite(os.path.join(output_dir, 'lena_median_filtered.png'), lena_median_filtered)
    display_images(
        [lena, lena_noise_reduced, lena_median_filtered],
        ['Original', 'Adaptive Noise Reduction', 'Adaptive Median Filter'],
        figsize=(15, 5)
    )
    plt.savefig(os.path.join(output_dir, 'adaptive_filters_comparison.png'))

def test_wiener_restoration():
    print("Testing Wiener Filter Restoration...")
    test_wiener_on_image('book_cover')
    test_wiener_on_image('boy')

def test_wiener_on_image(image_name):
    original_path = os.path.join(input_dir, f'{image_name.replace("_", "-")}.png')
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    blurred_path = os.path.join(input_dir, f'{image_name}_blurred_image.png')
    operator_path = os.path.join(input_dir, f'{image_name}_operator_H.png')
    
    if os.path.exists(blurred_path) and os.path.exists(operator_path):
        blurred_image = cv2.imread(blurred_path, cv2.IMREAD_GRAYSCALE)
        blurred_image = blurred_image.astype(np.float64) / 255.0
        h_pickle_path = os.path.join(input_dir, f'{image_name}_operator_H.pkl')
        if os.path.exists(h_pickle_path):
            with open(h_pickle_path, 'rb') as f:
                H = pickle.load(f)
        else:
            H = generate_motion_blur_operator(original.shape, a=0.1, b=0.1, T=1.0)
    else:
        H = generate_motion_blur_operator(original.shape, a=0.1, b=0.1, T=1.0)
        blurred_image = degrade_image(original, H, noise_variance=0.001)
        cv2.imwrite(os.path.join(output_dir, f'{image_name}_degraded.png'), 
                   (blurred_image * 255).astype(np.uint8))
    
    k_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    restored_images = []
    psnr_values = []
    ssim_values = []
    
    original_normalized = original.astype(np.float64) / 255.0
    
    for k in k_values:
        restored = wiener_filter_restoration(blurred_image, H, k)
        restored_images.append(restored)
        psnr_val, ssim_val = evaluate_restoration(original_normalized, restored)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        cv2.imwrite(os.path.join(output_dir, f'{image_name}_restored_k{k}.png'), 
                   (restored * 255).astype(np.uint8))
    
    best_k_idx = np.argmax(psnr_values)
    best_k = k_values[best_k_idx]
    
    display_images(
        [original_normalized, blurred_image, restored_images[best_k_idx]],
        ['Original', 'Degraded', f'Restored (K={best_k})'],
        figsize=(15, 5)
    )
    plt.savefig(os.path.join(output_dir, f'{image_name}_restoration_comparison.png'))
    
    n_images = 2 + len(k_values)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_normalized, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(blurred_image, cmap='gray')
    plt.title('Degraded')
    plt.axis('off')
    
    for i, (k, restored) in enumerate(zip(k_values, restored_images)):
        plt.subplot(n_rows, n_cols, i+3)
        plt.imshow(restored, cmap='gray')
        plt.title(f'K={k} (PSNR: {psnr_values[i]:.2f})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{image_name}_all_k_values.png'))

def main():
    test_adaptive_filters()
    test_wiener_restoration()

if __name__ == "__main__":
    main()