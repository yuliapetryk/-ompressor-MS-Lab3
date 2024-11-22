import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.util import img_as_float


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def alpha(val, n):
    return np.sqrt(1 / n) if val == 0 else np.sqrt(2 / n)


def dct_2d(block):
    n = block.shape[0]
    dct = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            dct[u, v] = alpha(u, n) * np.cos((np.pi / n) * (v + 0.5) * u)
    return dct @ block @ dct.T


def dct_compression(img, block_size=8):
    h, w = img.shape
    compressed_img = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i + block_size, j:j + block_size]
            dct_block = dct_2d(block)
            compressed_img[i:i + block_size, j:j + block_size] = dct_block
    return compressed_img


def idct_2d(block):
    n = block.shape[0]
    idct = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            idct[u, v] = alpha(u, n) * np.cos((np.pi / n) * (v + 0.5) * u)
    return idct.T @ block @ idct


def dct_decompression(compressed_img, block_size=8):
    h, w = compressed_img.shape
    decompressed_img = np.zeros_like(compressed_img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = compressed_img[i:i + block_size, j:j + block_size]
            idct_block = idct_2d(block)
            decompressed_img[i:i + block_size, j:j + block_size] = idct_block
    return decompressed_img


def dwt_compression(img, lev):
    h, w = img.shape
    compressed_img = img.copy().astype('float32')

    for level in range(lev):
        temp = compressed_img[:h, :w].copy()

        for i in range(h):
            row = temp[i, :w]
            compressed_img[i, :w] = np.concatenate([
                (row[0::2] + row[1::2]) / np.sqrt(2),
                (row[0::2] - row[1::2]) / np.sqrt(2)
            ])

        for j in range(w):
            col = compressed_img[:h, j]
            compressed_img[:h, j] = np.concatenate([
                (col[0::2] + col[1::2]) / np.sqrt(2),
                (col[0::2] - col[1::2]) / np.sqrt(2)
            ])

        h //= 2
        w //= 2

    return compressed_img


def dwt_decompression(compressed_img, lev, original_shape):
    h, w = original_shape
    decompressed_img = compressed_img.copy().astype('float32')

    for level in range(lev):
        h //= 2
        w //= 2

    for level in range(lev):
        h *= 2
        w *= 2

        temp = decompressed_img[:h, :w].copy()

        for j in range(w):
            col = temp[:h, j]
            even = (col[:h // 2] + col[h // 2:]) / np.sqrt(2)
            odd = (col[:h // 2] - col[h // 2:]) / np.sqrt(2)
            decompressed_img[:h, j] = np.ravel(np.column_stack((even, odd)))

        for i in range(h):
            row = decompressed_img[i, :w]
            even = (row[:w // 2] + row[w // 2:]) / np.sqrt(2)
            odd = (row[:w // 2] - row[w // 2:]) / np.sqrt(2)
            decompressed_img[i, :w] = np.ravel(np.column_stack((even, odd)))

    return decompressed_img[:original_shape[0], :original_shape[1]]


def calculate_psnr(original, compressed):
    mse = mean_squared_error(original, compressed)
    return 10 * np.log10(255 ** 2 / mse)


def main():
    image_path = "lena.png"
    img = read_image(image_path)
    img = img_as_float(img)

    levels = 1
    dct_img_compress = dct_compression(img)
    dwt_img_compress = dwt_compression(img, levels)

    dct_img_decompress = dct_decompression(dct_img_compress)
    dwt_img_decompress = dwt_decompression(dwt_img_compress, levels, img.shape)

    psnr_dct = calculate_psnr(img, dct_img_decompress)
    # psnr_dct = calculate_psnr(img, dct_img_compress)

    psnr_dwt = calculate_psnr(img, dwt_img_decompress)
    # psnr_dwt = calculate_psnr(img, dwt_img_compress)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title(f"DCT Compressed")
    plt.imshow(dct_img_compress, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title(f"DCT Decompressed\nPSNR: {psnr_dct:.2f}")
    plt.imshow(dct_img_decompress, cmap='gray')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title(f"DWT Compressed")
    plt.imshow(dwt_img_compress, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title(f"DWT Decompressed\nPSNR: {psnr_dwt:.2f} dB")
    plt.imshow(dwt_img_decompress, cmap='gray')
    plt.tight_layout()
    plt.show()


main()
