import cv2
import numpy as np

def apply_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image.
    Works for both grayscale and color images (by converting to LAB).
    """
    if image is None:
        return None
    
    # Check if image is color or grayscale
    if len(image.shape) == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    else:
        # Color - Convert to LAB, apply to L channel, merge back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def convert_color_space(image, target_space):
    """
    Converts image to different color spaces.
    target_space: 'HSV', 'YUV', 'GRAY'
    """
    if image is None: return None
    
    # Input is BGR
    if target_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif target_space == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif target_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_blur(image, kernel_size=5):
    """
    Applies Gaussian Blur to reduce noise.
    Wrapper for cv2.GaussianBlur.
    """
    if image is None:
        return None
    # Kernel size must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_blur(image, kernel_size=5):
    """
    Applies Median Blur to reduce salt-and-pepper noise.
    Wrapper for cv2.medianBlur.
    """
    if image is None:
        return None
    # Kernel size must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)

def apply_sobel(image):
    """
    Applies Sobel Edge Detection.
    """
    if image is None: return None
    
    # Convert to gray if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def apply_laplacian(image):
    """
    Applies Laplacian Edge Detection.
    """
    if image is None: return None
    
    # Convert to gray if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(lap)
