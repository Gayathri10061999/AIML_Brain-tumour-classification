# 🧠 Brain Tumor Classification using Deep Learning

## 📌 Overview

This project focuses on building a **deep learning-based system** to classify brain MRI images into multiple tumor categories. It combines a **custom Convolutional Neural Network (CNN)** and **transfer learning models** to achieve high accuracy and robustness.

The solution is deployed using a **Streamlit web application**, enabling real-time predictions from uploaded MRI scans.

---

## 🎯 Objectives

* Classify MRI images into tumor types:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor
* Compare performance of:

  * Custom CNN
  * Pretrained models (EfficientNetB0, ResNet50, MobileNet)
* Deploy an interactive application for real-time inference

---

## 💼 Business Use Cases

### 1. AI-Assisted Diagnosis

Supports radiologists with automated tumor classification, reducing diagnostic time and improving accuracy.

### 2. Early Detection & Triage

Identifies high-risk cases for priority review in hospitals.

### 3. Research & Clinical Trials

Enables dataset segmentation based on tumor types for medical research.

### 4. Telemedicine / Second Opinion

Provides AI-based diagnostic assistance in remote or resource-limited regions.

---

## 📂 Project Structure

```
brain-tumor-classification/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── augmentations.py
│   ├── model_custom.py
│   ├── model_transfer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│
├── models/
│   ├── best_model.h5
│
├── app/
│   ├── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

* Brain MRI image dataset containing labeled tumor categories
* Images resized to **224x224 pixels**
* Data split:

  * Training set
  * Validation set

> ⚠️ Ensure dataset is organized in directory format compatible with `ImageDataGenerator`.

---

## ⚙️ Data Preprocessing

* Image resizing → 224x224
* Pixel normalization → [0,1]
* Label encoding → categorical

---

## 🔄 Data Augmentation

Applied using `ImageDataGenerator`:

* Rotation
* Horizontal flipping
* Zoom
* Brightness adjustment

Purpose:

* Improve generalization
* Reduce overfitting

---

## 🧠 Models

### 1. Custom CNN

* Multiple Conv2D + MaxPooling layers
* Fully connected dense layers
* Dropout for regularization

### 2. Transfer Learning Models

* EfficientNetB0 (primary model)
* ResNet50
* MobileNet

Approach:

* Load pretrained weights (ImageNet)
* Freeze base layers
* Add custom classification head
* Fine-tune top layers

---

## 🏋️ Model Training

* Loss Function: `categorical_crossentropy`
* Optimizer: `Adam`
* Metrics: `Accuracy`

### Callbacks

* EarlyStopping
* ModelCheckpoint

---

## 📈 Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📊 Model Comparison

| Model          | Accuracy | Remarks           |
| -------------- | -------- | ----------------- |
| Custom CNN     | Medium   | Lightweight       |
| EfficientNetB0 | High     | Best performer    |
| ResNet50       | High     | Stable but slower |

---

## 🌐 Streamlit Web App

### Features

* Upload MRI image
* Real-time tumor prediction
* Confidence score display

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 🚀 Deployment Options

* Streamlit Cloud
* AWS EC2
* Docker containers

---

## 📦 Requirements

```
tensorflow
numpy
pandas
matplotlib
scikit-learn
streamlit
Pillow
```

---

## 🔍 Future Enhancements

* Grad-CAM visualization (model explainability)
* MLflow integration for experiment tracking
* FastAPI backend for scalable deployment
* Hyperparameter tuning (Optuna)
* CI/CD pipeline with GitHub Actions

---

## 🧪 How to Train the Model

```bash
cd src
python train.py
```

---

## 📷 Sample Output

* Predicted Class: Glioma
* Confidence: 0.94

---

## 🤝 Contributing

Contributions are welcome. Please follow standard Git workflow:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 📧 Contact

For queries or collaboration:

* Name: Gayathri M
* Role: Data Scientist / ML Engineer

---

## ⭐ Acknowledgements

* TensorFlow / Keras
* Open-source medical imaging datasets
* Streamlit for UI deployment

---
