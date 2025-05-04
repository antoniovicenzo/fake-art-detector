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

# Training images
download_images("AI-generated art", "data/train/ai", 500)
download_images("oil painting human art", "data/train/real", 500)

# Validation images
download_images("AI-generated art", "data/val/ai", 100)
download_images("oil painting human art", "data/val/real", 100)
