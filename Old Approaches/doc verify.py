import cv2
import easyocr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def detect_logo(image_path, logo_template_path, threshold=0.8):
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    image = Image.open(image_path)
    logo_template = Image.open(logo_template_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    logo_template = preprocess(logo_template)
    with torch.no_grad():
        image_feature = resnet(image.unsqueeze(0))
        logo_feature = resnet(logo_template.unsqueeze(0))
    similarity = torch.nn.functional.cosine_similarity(image_feature, logo_feature)
    return similarity.item() >= threshold

document_path = 'document.jpg'

document_image = cv2.imread(document_path)
logo_template_path = 'logo_template.jpg'

logo_detected = detect_logo(document_path, logo_template_path, 0.5)

if logo_detected:
    print("The document contains the logo.")
else:
    print("The document does not contain the logo.")

reader = easyocr.Reader(['en'])
results = reader.readtext(document_image)

document_image_rgb = cv2.cvtColor(document_image, cv2.COLOR_BGR2RGB)

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
