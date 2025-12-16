import cv2
import numpy as np
from sklearn.cluster import KMeans

def otsu_thresholding(image):
    """
    Apply Otsu's binarization.
    Expects grayscale image. If RGB, converts to gray first.
    Returns the binary image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to remove noise before thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def kmeans_segmentation(image, k=3):
    """
    Apply K-Means clustering for color segmentation.
    Returns the segmented image.
    """
    # Reshape image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria (stop when epsilon <= 1.0 or max_iter == 10)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Apply K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # Reshape back to original image shape
    segmented_image = segmented_data.reshape(image.shape)
    return segmented_image

def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    """
    Apply Canny Edge Detection.
    """
    # Canny expects grayscale usually, but can handle others. Better to ensure inputs.
    if len(image.shape) == 3:
        # Often better on blurred gray image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    return edges
