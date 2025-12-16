import cv2
import numpy as np

def load_image(image_file):
    """
    Load an image from a file buffer (Streamlit UploadedFile) or path.
    For Streamlit, we usually convert the file buffer to numpy array.
    """
    if isinstance(image_file, str):
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's a Streamlit UploadedFile
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize the image while maintaining aspect ratio.
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
