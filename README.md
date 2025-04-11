# 🎨 Fake Art or AI-Generated Image Detection

This project uses deep learning to classify images as either **human-made artwork** or **AI-generated art**. It's designed to support art buyers, galleries, and the general public in verifying the authenticity of digital art.

---

## 🚀 Project Overview

- **Goal**: Detect if an artwork was created by a human or an AI model like Midjourney, DALL·E, or Stable Diffusion.  
- **Model**: ResNet18 (transfer learning)  
- **Framework**: PyTorch  
- **Interface**: Streamlit web app  
- **Input**: JPG or PNG image  
- **Output**: Class label ("Real" or "AI-Generated") + confidence score  

---

## 🗂 Folder Structure

```
fake-art-detector/
├── app.py                  # Streamlit app for uploading and classifying images
├── train.py                # Model training script (PyTorch)
├── requirements.txt        # Project dependencies
├── README.md               # You're here!
├── models/                 # Saved trained models (.pt files)
├── data/                   # Real and AI-generated images (not included)
├── src/                    # Utilities and helper functions (optional)
└── notebooks/              # Jupyter experiments (optional)
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/antoniovicenzo/fake-art-detector.git
cd fake-art-detector
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎛️ How to Use

### To Train the Model

1. Prepare your data in this structure:

```
data/
├── train/
│   ├── real/
│   └── ai/
├── val/
│   ├── real/
│   └── ai/
```

2. Run the training script:

```bash
python train.py
```

The trained model will be saved to the `models/` folder.

---

### To Launch the Streamlit App

```bash
streamlit run app.py
```

Then upload any image to check if it was made by an AI or a human.

---

## 📦 Requirements

```
torch
torchvision
streamlit
Pillow
matplotlib
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 📊 Example Output

Upload an image and receive something like:

```
Result: Real Artwork (93.2% confidence)
```

---

## 📜 License & Usage

This project is for academic and educational purposes only.  
All artworks remain property of their respective creators.

---

## 🙏 Acknowledgements

- WikiArt and Behance for real art data  
- Hugging Face & PyTorch for modeling  
- Midjourney, DALL·E, and Stable Diffusion for generated samples  

---

## 📬 Contact

Questions or ideas? Message [antoniovicenzo](https://github.com/antoniovicenzo) on GitHub.
