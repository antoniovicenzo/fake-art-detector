import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load model
model = torch.load("models/fake_art_classifier_full.pt", 
map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

labels = ["AI-Generated", "Real Artwork"]

st.title("\U0001F3A8 Fake Art Detector")
image_file = st.file_uploader("Upload an image to check:", type=["jpg", "png"])

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    confidence = torch.softmax(output, dim=1)
    class_idx = torch.argmax(confidence).item()

    label = labels[class_idx]
    confidence_score = confidence[0][class_idx].item() * 100

    st.markdown(f"### Result: {label} ({confidence_score:.2f}% confidence)")
