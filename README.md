# Fingerprint Reconstruction using Diffusion Models

**Fingerprint Reconstruction using Diffusion Models** is a project that leverages Denoising Diffusion Probabilistic Models (DDPMs) to reconstruct damaged or partial fingerprint images. By utilizing the power of diffusion models, this approach aims to restore fingerprint patterns with high fidelity, enhancing the performance of biometric systems.

---

## 🧠 Overview

This project focuses on reconstructing degraded fingerprint images using a U-Net-based DDPM architecture. The model learns to reverse the degradation process by iteratively denoising the input, resulting in a restored fingerprint image that closely resembles the original.

---

## 📁 Project Structure

```
Fingerprint-Reconstruction-using-Diffusion-Models/
├── Test_Images/             # Directory containing test fingerprint images
├── results/                 # Directory to save reconstructed images
├── dataset.py               # Script for data loading and preprocessing
├── evaluate.py              # Script to evaluate the model on test data
├── model.py                 # Definition of the U-Net-based DDPM model
├── train.py                 # Script to train the model
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch
- NumPy
- OpenCV
- Matplotlib

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Qadeer2syed/Fingerprint-Reconstruction-using-Diffusion-Models.git
cd Fingerprint-Reconstruction-using-Diffusion-Models
```

2. **Install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Usage

### Training

Train the model using the provided training script:

```bash
python train.py
```

### Evaluation

Evaluate the trained model on test images:

```bash
python evaluate.py
```

*The reconstructed images will be saved in the `results/` directory.*

---

## 📊 Results

The model demonstrates effective reconstruction of degraded fingerprint images, preserving critical features necessary for identification.

---


