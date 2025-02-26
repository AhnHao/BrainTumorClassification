# app.py
import os
import torch
import numpy as np
from flask import Flask, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import BrainTumorCNN  # Import your model class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN(4)  # Adjust the number of classes as needed
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Define class labels
class_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and make prediction
            image = preprocess_image(file_path)
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
                pred_label = class_labels[pred.item()]

            # Render the result
            return render_template('result.html', filename=filename, label=pred_label, correct=True)  # Adjust 'correct' based on your logic

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)