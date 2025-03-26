import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import math

def load_grayscale_image(image_path):
    img = Image.open(image_path).convert('L')
    return np.array(img)

def bilinear_interpolation(image, x, y):
    height, width = image.shape

    if x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
        return 0 

    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    x1 = min(x1, width - 1)
    y1 = min(y1, height - 1)

    f00 = image[y0, x0]
    f01 = image[y0, x1]
    f10 = image[y1, x0]
    f11 = image[y1, x1]

    wx = x - x0
    wy = y - y0

    top = f00 * (1 - wx) + f01 * wx
    bottom = f10 * (1 - wx) + f11 * wx
    return int(top * (1 - wy) + bottom * wy)

def calculate_output_dimensions(image_shape, matrix):
    height, width = image_shape

    corners = np.array([
        [0, 0, 1],
        [width - 1, 0, 1],
        [0, height - 1, 1],
        [width - 1, height - 1, 1]
    ])

    transformed_corners = np.dot(corners, matrix.T)

    min_x = np.floor(np.min(transformed_corners[:, 0])).astype(int)
    max_x = np.ceil(np.max(transformed_corners[:, 0])).astype(int)
    min_y = np.floor(np.min(transformed_corners[:, 1])).astype(int)
    max_y = np.ceil(np.max(transformed_corners[:, 1])).astype(int)

    new_width = max_x - min_x + 1
    new_height = max_y - min_y + 1
    
    return new_height, new_width, min_x, min_y

def affine_transform(image, matrix, output_size=None):
    height, width = image.shape

    if output_size is None:
        new_height, new_width, min_x, min_y = calculate_output_dimensions(image.shape, matrix)

        adjustment_matrix = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])
        matrix = np.dot(adjustment_matrix, matrix)
    else:
        new_height, new_width = output_size

    output = np.zeros((new_height, new_width), dtype=np.uint8)
    inv_matrix = np.linalg.inv(matrix)

    for j in range(new_height):
        for i in range(new_width):
            source = np.dot(inv_matrix, np.array([i, j, 1]))
            x, y = source[0], source[1]
            output[j, i] = bilinear_interpolation(image, x, y)
    
    return output

def create_rotation_matrix(angle_degrees, center=None):
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    if center is None:
        return np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
    else:
        cx, cy = center
        return np.array([
            [cos_theta, -sin_theta, cx * (1 - cos_theta) + cy * sin_theta],
            [sin_theta, cos_theta, cy * (1 - cos_theta) - cx * sin_theta],
            [0, 0, 1]
        ])

def display_images(original, transformed, title="Affine Transformation"):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(transformed, cmap='gray')
    plt.title('Transformed Image')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_image(image, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    Image.fromarray(image).save(output_path)
    print(f"Transformed image saved to {output_path}")

def main():
    current_dir = os.path.dirname(os.path.abspath('__file__'))
    input_dir = os.path.join(current_dir, r"P3_Affine_transformations\Input")
    output_dir = os.path.join(current_dir, r"P3_Affine_transformations\Output")
    input_image = os.listdir(input_dir)[0]
    input_path = os.path.join(input_dir, input_image)
    output_path = output_dir + "/transformed_image.jpg"
    
    image = load_grayscale_image(input_path)
    height, width = image.shape
    center = (width // 2, height // 2)
    
    matrix = create_rotation_matrix(45, center=center)
    title = "Rotation (45 degrees) around center"

    transformed_image = affine_transform(image, matrix)
    display_images(image, transformed_image, title)
    save_image(transformed_image, output_path)

if __name__ == "__main__":
    main()