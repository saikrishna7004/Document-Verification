import cv2
import easyocr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

def detect_logo(*args, **kwargs):
    return True, True

# Paths to your images
document_path = 'document.jpg'
logo_template_path = 'logo_template.jpg'

# Load and preprocess the document image
document_image = cv2.imread(document_path)
document_image = cv2.cvtColor(document_image, cv2.COLOR_BGR2RGB)

# Detect the logo in the document
logo_detected, logo_bbox = detect_logo(document_path, logo_template_path, 0.5)

if logo_detected:
    print("The document contains the logo.")
    print("Logo Bounding Box:", logo_bbox)
else:
    print("The document does not contain the logo.")

# Text extraction using easyocr
reader = easyocr.Reader(['en'])
results = reader.readtext(document_image)

document_image_rgb = cv2.cvtColor(document_image, cv2.COLOR_BGR2RGB)

# Highlight text and display the image
for detection in results:
    bbox = detection[0]
    bbox = [[int(x), int(y)] for x, y in bbox]
    pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(document_image_rgb, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

print("Extracted Text from Document:")
for detection in results:
    print(f"Text: {detection[1]}, Confidence: {detection[2]:.2f}")

cv2.imshow('Document with Text Highlights and Logo Detection', document_image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
