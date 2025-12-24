import cv2
import numpy as np

def apply_canny_edge_detection(image, low_threshold=100, high_threshold=200):
    """
    Applies Canny Edge Detection.
    """
    if image is None:
        return None
    return cv2.Canny(image, low_threshold, high_threshold)

def extract_geometric_features(binary_image):
    """
    Extracts geometric features from a binary image (Area, Perimeter, Circularity).
    Returns a list of dictionaries for each contour found.
    """
    if binary_image is None:
        return []
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features_list = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Circularity = 4 * pi * Area / (Perimeter^2)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
        features_list.append({
            "contour": cnt,
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity
        })
        
    return features_list, contours

def extract_color_histogram(image):
    """
    Computes the color histogram for RGB channels.
    """
    if image is None: return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([rgb], [2], None, [256], [0, 256])
    return hist_r, hist_g, hist_b

def draw_features(image, features, contours):
    """
    Helper to visualize contours and their features on an image.
    """
    img_copy = image.copy()
    
    # Draw all contours
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
    
    for i, data in enumerate(features):
        cnt = data["contour"]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Label index
            cv2.putText(img_copy, f"#{i}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    return img_copy
