import sys
import os
import cv2
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from classifier import ImageClassifier

def test_vit():
    print("Testing Vision Transformer (ViT) integration...")
    try:
        classifier = ImageClassifier(model_name='Vision Transformer (ViT)')
        
        # Create a more realistic dummy image (random colors)
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        print(f"Device being used: {classifier.device}")
        print("Running prediction...")
        preds = classifier.predict(dummy_img)
        print("Top Predictions:")
        for idx, label, score in preds:
            print(f"- {label}: {score:.4f}")
        
        if len(preds) > 0:
            print("Test PASSED: Predictions returned.")
        else:
            print("Test FAILED: No predictions returned.")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vit()
