from flask_cors import CORS
import os
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
from ImageClassification import OptiSA, device
import tempfile

app = Flask(__name__)
CORS(app)

# Load the model
model = OptiSA(num_classes=5).to(device)
model.load_state_dict(torch.load('best_opti_sa.pth', map_location=device))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Use system's temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'temp_image.jpg')
        file.save(temp_path)
        
        try:
            # Make prediction
            predicted_class, confidence = predict_image(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': f'{confidence:.2%}'
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)})

# For local development
if __name__ == '__main__':
    app.run(debug=True) 