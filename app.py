from flask import Flask, render_template, request, jsonify
import cv2
import easyocr
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import easyocr
from roboflow import Roboflow

app = Flask(__name__)

# Define 'model' in the global scope
rf = Roboflow(api_key="unMsLfhQFFysKy7Fahc0")
project = rf.workspace().project("document-verification-rbdur")
model = project.version(3).model
reader = easyocr.Reader(['en', 'hi'])

d = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], 
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], 
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], 
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], 
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1], 
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], 
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], 
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], 
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

p = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], 
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], 
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], 
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], 
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1], 
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], 
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def validate_aadhar(aadhar_number):
    aadhar_number = str(aadhar_number)
    if len(aadhar_number) != 12:
        print("Aadhar should be 12 digits")
        return False

    if not aadhar_number.isdigit():
        print("Aadhaar must be a number")
        return False

    c = 0
    inverted_list = list(map(int, aadhar_number[::-1]))
    for i in range(len(inverted_list)):
        c = d[c][p[(i % 8)][inverted_list[i]]]

    if c == 0:
        print("Aadhaar number is valid")
        return True
    else:
        print("Aadhaar number is invalid")
        return False
    
# Function to process the uploaded image
def process_uploaded_image(image):
    try:
        nparr = np.frombuffer(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cv2.imwrite('./images/test_file.png', img_np)
        print('Model:\tPrediction started')
        result = model.predict('./images/test_file.png', confidence=40, overlap=30).json()
        print('Model:\tPrediction done')

        image = cv2.imread('./images/test_file.png')

        l = []
        classes = []
        aadhar_texts = ['']

        for prediction in result['predictions']:
            center_x, center_y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            roi = image[y:y2, x:x2]

            ocr_result = reader.readtext(roi)

            if 'aadhar no' in prediction['class'].lower():
                aadhar_texts.extend([text for (_, text, _) in ocr_result])

            l.extend([text for (_, text, _) in ocr_result])

            class_name = prediction['class']
            classes.append(class_name.lower())
            
            class_name_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x, y), (x + class_name_size[0] + 10, y + class_name_size[1] + 10), (0, 255, 0), -1)
            cv2.putText(image, prediction['class'].capitalize(), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

        auth = False
        if 'gov' in classes and 'logo' in classes:
            auth = True

        print("aadhar_texts", aadhar_texts)
        
        valid_aadhar_no = False

        for text in aadhar_texts:
            text_without_spaces = ''.join(text.split())

            if len(text_without_spaces) == 12 and text_without_spaces.isdigit() and validate_aadhar(text_without_spaces):
                valid_aadhar_no = True
                break

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            'processed_image': img_b64,
            'ocr_text': str(l),
            'authentic': auth,
            'aadhar_no': aadhar_texts[-1],
            'aadhar_no_valid': valid_aadhar_no
        }
    except Exception as e:
        print(f"Error occurred: {e.with_traceback(None)}")
        return {'error': 'Error processing image'}

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image processing
@app.route('/process_image', methods=['POST'])
def upload_file():
    try:
        file = request.files['file'].read()
        processed_data = process_uploaded_image(file)
        
        return jsonify(processed_data)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Error processing image'})


if __name__ == '__main__':
    app.run(debug=True)
