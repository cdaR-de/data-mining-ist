---

# Wonders of the World Image Classification with YOLO v8l

This repository demonstrates the use of the **Wonders of the World Image Classification** dataset from [Kaggle](https://www.kaggle.com/datasets/balabaskar/wonders-of-the-world-image-classification) to develop an object detection model using **YOLO v8l**. Additionally, a **Streamlit app** is provided for real-time classification of images based on this model.

---

## 🗂 Dataset

The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/balabaskar/wonders-of-the-world-image-classification), includes images of various wonders of the world. It is structured for use in image classification and object detection tasks.

### Key Features:
- High-quality labeled images.
- Categories include famous wonders such as the **Great Wall of China**, **Taj Mahal**, **Machu Picchu**, and more.
- Suitable for training and testing deep learning models.

---

## 🛠 Model Development with YOLO v8l

The object detection pipeline is built using **YOLO v8l** (You Only Look Once, version 8 - large model) for robust and efficient image classification.

### Steps:
1. **Preprocessing**: 
   - Images were resized and annotated for YOLO v8 format.
   - Data augmentation techniques were applied to improve generalization.

2. **Model Training**:
   - YOLO v8l was fine-tuned on the dataset using transfer learning.
   - Hyperparameters such as learning rate, batch size, and epochs were optimized for best performance.

3. **Evaluation**:
   - The model was tested on a separate validation set.
   - Metrics such as mAP (mean Average Precision) and F1-score were used to assess performance.

---

## 🚀 Streamlit App

A **Streamlit app** has been deployed to allow users to interact with the trained model. The app enables users to upload images and receive real-time predictions about which wonder is detected in the image.

### App Features:
- **Simple Interface**: Upload an image and get instant results.
- **High Accuracy**: Powered by YOLO v8l for precise detection.
- **Live Demo**: Access the app [here](https://data-mining-ist-dtizaizdajbxzpg7phumnl.streamlit.app).

---

---

## 📊 Results

The YOLO v8l model achieved the following results on the validation set:
- **mAP@50**: 96.8%
- **Precision**: 0.93
- **Recall**: 0.91

These results demonstrate the model's effectiveness in accurately identifying wonders from images.

## Contributors  
- [Senin Adın](https://github.com/cdaR-de)  
- [Arkadaşının Adı](https://github.com/mmelzey)  


---
