# app.py
import os
import torch
from flask import Flask, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from model import BrainTumorCNN
import atexit
import glob

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
class_labels = ["Glioma", "Meningioma" , "No Tumor", "Pituitary"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert("L")
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

def cleanup_uploads():
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        os.remove(f)
    print("Uploads folder cleaned up.")

atexit.register(cleanup_uploads)

if __name__ == '__main__':
    app.run(debug=True)