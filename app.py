import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
import torch.nn as nn

# Load model
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/fake_art_classifier.pt", map_location=torch.device('cpu')))
model.eval()

# Transform input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("🎨 Fake Art Detector")
image_file = st.file_uploader("Upload an image to check:", type=["jpg", "png"])

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    confidence = torch.softmax(output, dim=1)
    class_idx = torch.argmax(confidence).item()

    label = "Real Artwork" if class_idx == 0 else "AI-Generated"
    confidence_score = confidence[0][class_idx].item() * 100

    st.markdown(f"### Result: {label} ({confidence_score:.2f}% confidence)")
