import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_and_convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def get_pixel_range(image):
    min_value = np.min(image)
    max_value = np.max(image)
    return min_value, max_value

def normalize_image(image):
    min_val, max_val = get_pixel_range(image)
    if max_val > min_val:
        normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = image.copy()
    return normalized

def apply_mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def SubMatriz(img, center, k):
    height, width = img.shape
    x, y = center
    half_k = k // 2
    start_row = max(0, y - half_k)
    start_col = max(0, x - half_k)
    end_row = min(height, y + half_k + 1)
    end_col = min(width, x + half_k + 1)
    return img[start_row:end_row, start_col:end_col]

def apply_maximum_filter(image, kernel_size=3):
    result = np.zeros_like(image)
    height, width = image.shape
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                       cv2.BORDER_REFLECT)
    for y in range(height):
        for x in range(width):
            submatrix = SubMatriz(padded_image, (x + pad_size, y + pad_size), kernel_size)
            max_value = np.max(submatrix)
            result[y, x] = max_value
    return result

def apply_minimum_filter(image, kernel_size=3):
    result = np.zeros_like(image)
    height, width = image.shape
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                       cv2.BORDER_REFLECT)
    for y in range(height):
        for x in range(width):
            submatrix = SubMatriz(padded_image, (x + pad_size, y + pad_size), kernel_size)
            min_value = np.min(submatrix)
            result[y, x] = min_value
    return result

def display_image_comparison(original, processed, title="Image Comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title("Processed")
    axes[1].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_processed_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

def process_images(image_paths, kernel_size=3, output_dir="results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images_data = []
    
    for image_path in image_paths:
        image_name = Path(image_path).stem
        print(f"\nProcessing {image_name}...")
        gray_image = load_and_convert_to_grayscale(image_path)
        save_processed_image(gray_image, f"{output_dir}/{image_name}_gray.jpg")
        min_val, max_val = get_pixel_range(gray_image)
        print(f"Pixel range: [{min_val}, {max_val}]")
        normalized_image = normalize_image(gray_image)
        save_processed_image(normalized_image, f"{output_dir}/{image_name}_normalized.jpg")
        mean_filtered = apply_mean_filter(normalized_image, kernel_size)
        save_processed_image(mean_filtered, f"{output_dir}/{image_name}_mean.jpg")
        median_filtered = apply_median_filter(normalized_image, kernel_size)
        save_processed_image(median_filtered, f"{output_dir}/{image_name}_median.jpg")
        max_filtered = apply_maximum_filter(normalized_image, kernel_size)
        save_processed_image(max_filtered, f"{output_dir}/{image_name}_max.jpg")
        min_filtered = apply_minimum_filter(normalized_image, kernel_size)
        save_processed_image(min_filtered, f"{output_dir}/{image_name}_min.jpg")
        images_data.append({
            "name": image_name,
            "pixel_range": (min_val, max_val),
            "kernel_size": kernel_size
        })

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "Input")
    output_dir = os.path.join(current_dir, "Output")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory at {input_dir}")
        print("Please place your images in the Input folder and run the script again.")
        return
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]
    if not image_paths:
        print("No valid images found in the Input folder.")
        print("Please add some images and run the script again.")
        return
    kernel_size = 3
    process_images(image_paths, kernel_size, output_dir)
    print("\nImage processing completed successfully!")

if __name__ == "__main__":
    main()