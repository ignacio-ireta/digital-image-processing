import cv2
import numpy as np
import matplotlib.pyplot as plt

def reconstruct_image(U, S, Vt, num_singular_values):
    U_reduced = U[:, :num_singular_values]
    S_reduced = np.diag(S[:num_singular_values])
    Vt_reduced = Vt[:num_singular_values, :]
    reconstructed_image = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))
    return reconstructed_image

def calculate_compression_rate(original_size, U, S, Vt, num_singular_values):
    compressed_size = (U[:, :num_singular_values].nbytes +
                       S[:num_singular_values].nbytes +
                       Vt[:num_singular_values, :].nbytes)
    compression_rate = compressed_size / original_size
    return compression_rate

image_path = 'bearTesselation.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

U, S, Vt = np.linalg.svd(gray_image, full_matrices=False)

original_size = gray_image.nbytes

s_values = np.logspace(0, np.log10(min(gray_image.shape)), num=12, dtype=int)

reconstructed_images = []
compression_rates = []

for s in s_values:
    reconstructed_image = reconstruct_image(U, S, Vt, s)
    compression_rate = calculate_compression_rate(original_size, U, S, Vt, s)
    reconstructed_images.append(reconstructed_image)
    compression_rates.append(compression_rate)

plt.figure(figsize=(15, 10))
for i, (s, img, rate) in enumerate(zip(s_values, reconstructed_images, compression_rates)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'S = {s} Comp.: {rate:.2%}')
    plt.axis('off')
# plt.tight_layout()
plt.show()
