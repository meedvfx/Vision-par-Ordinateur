import cv2
import numpy as np

def apply_simple_threshold(image, thresh_val):
    """
    Applies simple binary thresholding.
    """
    if image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return binary

def apply_otsu_threshold(image):
    """
    Applies Otsu's binarization.
    Returns the binary image.
    """
    if image is None:
        return None
    
    # Needs grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Otsu's thresholding
    # Returns (threshold_value, binary_image)
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def apply_gmm_segmentation(image, n_components=3):
    """
    Applies GMM (Gaussian Mixture Model) for segmentation.
    n_components: Number of Gaussian components.
    """
    if image is None: return None
    from sklearn.mixture import GaussianMixture
    
    # Reshape and normalize
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', max_iter=20)
    labels = gmm.fit_predict(pixel_values)
    centers = gmm.means_
    
    # Map pixels to centers
    segmented_image = centers[labels]
    segmented_image = np.uint8(segmented_image)
    return segmented_image.reshape(image.shape)

def apply_kmeans_segmentation(image, k=3):
    """
    Applies K-Means Clustering for color segmentation.
    k: Number of clusters (colors)
    """
    if image is None:
        return None
    
    # Reshape image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    # Convert to float
    pixel_values = np.float32(pixel_values)
    
    # Define criteria = ( type, max_iter = 100, epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Apply KMeans
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    
    # Flatten the labels array
    labels = labels.flatten()
    
    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

def apply_watershed_segmentation(image):
    """
    Applies Watershed algorithm to separate touching objects.
    Assumes objects are light on dark background roughly.
    """
    if image is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize (Otsu)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply Watershed
    # Watershed modifies the image in-place (plotting boundaries in red usually)
    # We work on a copy to not destroy original
    img_copy = image.copy()
    markers = cv2.watershed(img_copy, markers)
    
    # Mark boundaries in Red
    img_copy[markers == -1] = [0, 0, 255]
    
    return img_copy, markers
