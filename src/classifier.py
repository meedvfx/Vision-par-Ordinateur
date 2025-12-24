import numpy as np
import cv2
import torch
from transformers import AutoImageProcessor, ViTForImageClassification

class ImageClassifier:
    def __init__(self, model_name='MobileNetV2', model_path=None, dataset='imagenet'):
        self.model_name = model_name
        self.model = None
        self.labels = None
        self.target_size = (224, 224)
        self.preprocess = None
        self.decode = None
        self.dataset = dataset # 'imagenet', 'caltech', 'food101'
        
        self.load_model(model_name, model_path, dataset)
        
    def load_model(self, model_name, model_path=None, dataset='imagenet'):
        self.model_name = model_name
        self.dataset = dataset
        
        # Determine architecture and defaults
        if 'MobileNet' in model_name:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions as mobilenet_decode
            base_class = MobileNetV2
            self.preprocess = mobilenet_preprocess
            self.decode = mobilenet_decode
            self.target_size = (224, 224)
        elif 'ResNet' in model_name:
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
            base_class = ResNet50
            self.preprocess = resnet_preprocess
            self.decode = resnet_decode
            self.target_size = (224, 224)
        elif 'Inception' in model_name:
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess, decode_predictions as inception_decode
            base_class = InceptionV3
            self.preprocess = inception_preprocess
            self.decode = inception_decode
            self.target_size = (299, 299)
        elif 'ViT' in model_name:
            self.target_size = None # Handled by processor
            self.preprocess = None
            self.decode = self.vit_decode
            base_class = 'ViT'
        elif 'Hybrid' in model_name or 'Custom' in model_name:
             from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
             self.preprocess = mobilenet_preprocess
             self.decode = self.custom_decode
             self.target_size = (224, 224)
             base_class = None 
        else:
             from tensorflow.keras.applications import MobileNetV2
             from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess, decode_predictions as mobilenet_decode
             base_class = MobileNetV2 
             self.preprocess = mobilenet_preprocess
             self.decode = mobilenet_decode


        # Load Model
        if model_path:
            # Load custom/fine-tuned weights
            import tensorflow as tf
            print(f"Loading custom model from {model_path} for {dataset}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Load Labels based on dataset
            # Load Labels based on dataset
            if dataset == 'cifar10':
                try:
                    from cifar10_labels import CIFAR_10_CLASSES
                    self.labels = CIFAR_10_CLASSES
                except ImportError:
                    print("Could not load cifar10_labels.py")
                    self.labels = None
            else:
                 self.labels = None 

            self.decode = self.custom_decode
            
            # IMPORTANT: For Hybrid or specific custom models, we might need to override input shape
            if getattr(self.model, 'input_shape', None):
                 try:
                     self.target_size = self.model.input_shape[1:3]
                 except:
                     pass

        else:
            if base_class == 'ViT':
                print("Loading Vision Transformer (ViT) from Hugging Face Hub...")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", low_cpu_mem_usage=False).to(self.device)
            elif base_class:
                # Load basic ImageNet weights
                import tensorflow as tf
                print(f"Loading basic {model_name} with ImageNet weights")
                self.model = base_class(weights='imagenet')
            else:
                raise ValueError(f"No base class found for {model_name} and no model_path provided.")

    def vit_decode(self, outputs, top=3):
        # ViT classification decoding
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        
        top_probs, top_indices = torch.topk(probs, k=top)
        
        results = []
        for i in range(top):
            idx = top_indices[i].item()
            score = top_probs[i].item()
            label = self.model.config.id2label[idx]
            results.append((str(idx), label, score))
        return results

    def custom_decode(self, preds, top=3):
        # preds shape (1, num_classes)
        if self.labels is None:
            # Fallback: Generate "Class N"
            self.labels = [f"Class {i}" for i in range(preds.shape[1])]
            
        top_indices = preds[0].argsort()[-top:][::-1]
        
        results = []
        for i in top_indices:
            label = str(i)
            if i < len(self.labels):
                label = self.labels[i]
            score = preds[0][i]
            results.append((str(i), label, score))
        return results
            
    def predict(self, image_array):
        """
        Runs prediction on a single image (numpy array BGR from cv2).
        Returns top 3 decoded predictions.
        """
        if self.model is None:
            return None

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        if 'ViT' in self.model_name:
            inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return self.vit_decode(outputs, top=3)
            
        # Standard Keras flow
        # Resize image to target size
        img = cv2.resize(image_array, self.target_size)
        
        # Convert BGR to RGB (Keras models usually expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Expand dims to batch (1, H, W, 3)
        x = np.expand_dims(img, axis=0)
        
        # Preprocess
        x = x.astype(np.float32)
        if self.preprocess:
            x = self.preprocess(x)
        
        # Predict
        preds = self.model.predict(x)
        
        # Decode
        if self.decode:
            results = self.decode(preds, top=3)
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                return results[0]
            return results
        return []

