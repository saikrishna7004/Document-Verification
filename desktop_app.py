import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import cv2
import easyocr
from roboflow import Roboflow

# Define 'model' in the global scope
rf = Roboflow(api_key="unMsLfhQFFysKy7Fahc0")
project = rf.workspace().project("document-verification-rbdur")
model = project.version(1).model

# Function to process the selected image
def process_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected. Exiting...")
        return

    loading_label.config(text="Processing Image...")
    loading_label.update()

    def process_image_thread():
        try:
            # Load the image and perform prediction and OCR
            result = model.predict(file_path, confidence=40, overlap=30).json()
            loading_label.config(text="Prediction done")
            image = cv2.imread(file_path)
            reader = easyocr.Reader(['en'])
            loading_label.config(text="OCR Loading")

            for prediction in result['predictions']:
                center_x, center_y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                x2 = int(center_x + width / 2)
                y2 = int(center_y + height / 2)

                roi = image[y:y2, x:x2]
                ocr_result = reader.readtext(roi)

                for (bbox, text, prob) in ocr_result:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = (int(top_left[0]) + x, int(top_left[1]) + y)
                    bottom_right = (int(bottom_right[0]) + x, int(bottom_right[1]) + y)
                    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
                    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                class_name = prediction['class'].capitalize()
                class_name_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x, y), (x + class_name_size[0] + 10, y + class_name_size[1] + 10), (0, 255, 0), -1)
                cv2.putText(image, prediction['class'].capitalize(), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

            cv2.imwrite("result_image.png", image)
            img = Image.open("result_image.png")
            img.thumbnail((600, 600))
            img = ImageTk.PhotoImage(img)

            image_label.config(image=img)
            image_label.image = img
            loading_label.config(text="Processing Complete")
        except Exception as e:
            loading_label.config(text="Error occurred")

    # Start the image processing in a separate thread
    processing_thread = threading.Thread(target=process_image_thread)
    processing_thread.start()

# Create a Tkinter window
root = tk.Tk()
root.title("Document Verification App")

# Increase the window size
root.geometry("800x600")

# Add a label for the app title
title_label = tk.Label(root, text="Document Verification App", font=("Helvetica", 20))
title_label.pack(pady=10)

# Create a frame for result sections
result_frame = tk.Frame(root)
result_frame.pack(pady=20)

# Create labels for different sections
prediction_label = tk.Label(result_frame, text="Prediction Results", font=("Helvetica", 16))
prediction_label.pack(pady=20)
prediction_label.grid(row=0, column=0)

ocr_label = tk.Label(result_frame, text="OCR Text", font=("Helvetica", 16))
ocr_label.grid(row=0, column=1)

# Create a button to select and process an image
select_button = tk.Button(root, text="Select Image", command=process_image, font=("Helvetica", 14))
select_button.pack(pady=10)

# Add a label for loading animation
loading_label = tk.Label(root, text="", font=("Helvetica", 16))
loading_label.pack(pady=20)
loading_label.pack()

# Add a label for displaying the processed image
image_label = tk.Label(root)
image_label.pack()

# Start the Tkinter main loop
root.mainloop()
