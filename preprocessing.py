import cv2
import numpy as np

def to_gray(image):
    """Convert RGB image to Grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def to_hsv(image):
    """Convert RGB image to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def equalize_histogram(image):
    """
    Apply histogram equalization.
    If image is RGB, convert to YCrCb, equalize Y channel, and convert back.
    If grayscale, apply directly.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    channels = cv2.split(ycrcb)
    
    # Equalize the Y channel (luminance)
    cv2.equalizeHist(channels[0], channels[0])
    
    # Merge and convert back
    ycrcb = cv2.merge(channels)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def gaussian_blur(image, kernel_size=5):
    """Apply Gaussian Blur."""
    k = (kernel_size, kernel_size)
    return cv2.GaussianBlur(image, k, 0)

def median_blur(image, kernel_size=5):
    """Apply Median Blur."""
    return cv2.medianBlur(image, kernel_size)

def extract_color_histogram(image):
    """
    Extract color histogram features.
    Concatenates histograms for R, G, B channels.
    Returns: flattened normalized histogram vector.
    """
    # Calculate histogram for each channel
    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
    
    # Normalize
    cv2.normalize(hist_r, hist_r)
    cv2.normalize(hist_g, hist_g)
    cv2.normalize(hist_b, hist_b)
    
    # Concatenate
    hist_features = np.concatenate((hist_r, hist_g, hist_b)).flatten()
    return hist_features

def sharpen_image(image):
    """
    Sharpen the image using a kernel.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def morphological_ops(image, op_type='dilation', kernel_size=5):
    """
    Perform morphological operations.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if op_type == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif op_type == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif op_type == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif op_type == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image
