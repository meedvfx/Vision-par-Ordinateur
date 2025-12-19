import tensorflow as tf
# MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_mobilenet, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet

import numpy as np
import cv2

def load_dl_model(model_name="MobileNetV2"):
    """
    Load a pre-trained Deep Learning model.
    """
    if model_name == "MobileNetV2":
        # Load with ImageNet weights, include_top=True for classification
        model = MobileNetV2(weights='imagenet', include_top=True)
    elif model_name == "ResNet50":
        model = ResNet50(weights='imagenet', include_top=True)
    else:
        return None
    return model

def load_feature_extractor(model_name="MobileNetV2"):
    """
    Load model for feature extraction (without top layer).
    """
    if model_name == "MobileNetV2":
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == "ResNet50":
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    else:
        return None
    return model

def predict_dl_class(image, model):
    """
    Predict the class of an image using a pre-trained DL model (MobileNetV2 or ResNet50).
    Returns: list of (class_name, probability) tuples.
    """
    import numpy as np
    import cv2
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

    # Redimensionner l'image à la taille attendue par le modèle
    img_resized = cv2.resize(image, (224, 224))
    
    # Ajouter la dimension batch pour que le modèle accepte une seule image
    img_array = np.expand_dims(img_resized, axis=0)

    # Appliquer le prétraitement approprié selon le modèle
    model_name = model.name.lower()
    
    if "mobilenetv2" in model_name:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
        img_array = preprocess_mobilenet(img_array)
    elif "resnet50" in model_name:
        from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
        img_array = preprocess_resnet(img_array)
    else:
        raise ValueError("Unsupported model for preprocessing")

    # Effectuer la prédiction
    preds = model.predict(img_array)

    # Décoder les 3 classes les plus probables
    decoded_preds = decode_predictions(preds, top=3)[0]
    results = [(pred[1], float(pred[2])) for pred in decoded_preds]

    return results

# --- Custom Model Logic ---

def load_custom_model(model_path):
    """Load a custom trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_labels(label_path):
    """Load class labels from a JSON file."""
    try:
        import json
        with open(label_path, 'r') as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

def predict_custom_model(image, model, labels):
    """Predict class using the custom trained model."""
    img_resized = cv2.resize(image, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    top_idx = np.argmax(preds[0])
    top_prob = preds[0][top_idx]
    
    if labels and len(labels) > top_idx:
        class_name = labels[top_idx]
    else:
        class_name = f"Class {top_idx}"
    return class_name, float(top_prob)

    return class_name, float(top_prob)
