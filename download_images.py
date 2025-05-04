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
(base) tony@TonysMacPro fake-art-detector % cat download_images.py 
from duckduckgo_search import DDGS
import requests
import os

def download_images(query, folder, max_images=500):
    os.makedirs(folder, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        for i, result in enumerate(results):
            try:
                img_url = result["image"]
                img_data = requests.get(img_url).content
                with open(os.path.join(folder, f"{query.replace(' ', '_')}_{i}.jpg"), 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                print(f"Failed to download image {i}: {e}")

# AI-generated art (unchanged, still good)
download_images("AI-generated art", "data/train/ai", 500)
download_images("AI-generated art", "data/val/ai", 100)

# Real artwork: better, more classical sources
download_images("famous classical paintings", "data/train/real", 200)
download_images("museum oil paintings", "data/train/real", 200)
download_images("Renaissance artwork", "data/train/real", 100)

download_images("famous classical paintings", "data/val/real", 50)
download_images("museum oil paintings", "data/val/real", 50)
