from duckduckgo_search import ddg_images
import requests
import os

def download_images(query, folder, max_images=500):
    os.makedirs(folder, exist_ok=True)
    results = ddg_images(query, max_results=max_images)
    
    for i, result in enumerate(results):
        try:
            img_data = requests.get(result["image"]).content
            with open(os.path.join(folder, f"{query.replace(' ', '_')}_{i}.jpg"), 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f"Failed to download image {i}: {e}")

# Example usage
download_images("AI-generated art", "data/train/ai", 500)
download_images("oil painting human art", "data/train/real", 500)
