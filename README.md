# weed-detection
# 🌿 Weed Detection Using Deep Learning (YOLOv8 + ResNet-50 + SAM + XAI)

This project implements a robust pipeline for detecting, classifying, and segmenting **weeds and crops** using a combination of **YOLOv8**, **ResNet-50**, **Segment Anything Model (SAM)**, and **Explainable AI (XAI)** techniques.

---

## 📂 Project Structure

---

## 🚀 Project Pipeline

### 1. **Dataset Preparation**
- **Classes:** `crop`, `weed`
- **Format:** YOLO + COCO
- **Augmentation:** Canopy-style augmentations to simulate real agricultural environments.

### 2. **YOLOv8 Detection**
- **Model:** YOLOv8n (Ultralytics)
- **Task:** Detect all plant instances (weed or crop).
- **Output:** Bounding boxes

### 3. **ResNet-50 Classification**
- **Input:** Cropped detections from YOLO
- **Model:** Fine-tuned `ResNet-50`
- **Accuracy:** Achieved **94% classification accuracy** on distinguishing weeds from crops
- **Framework:** PyTorch + torchvision

### 4. **SAM Segmentation**
- **Library:** Meta AI’s [Segment Anything](https://github.com/facebookresearch/segment-anything)
- **Purpose:** Pixel-level segmentation of only plants (background ignored)
- **Input:** Regions classified as weed or crop

### 5. **Explainable AI (XAI)**
- **Tool:** Grad-CAM (Visual explanation for ResNet-50 decisions)
- **Purpose:** Understand model predictions and highlight image regions influencing classification
- **Notebook:** `explainable_ai.ipynb`

---

## 🛠 Setup

### 🔧 Requirements

```bash
pip install ultralytics opencv-python torch torchvision matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install grad-cam

