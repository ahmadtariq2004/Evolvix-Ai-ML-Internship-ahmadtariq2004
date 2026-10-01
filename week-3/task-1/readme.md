# Week 3, Task 1: Neural Network for Image Classification 🧠🖼️

## 📝 Task Overview
This project is part of the **Evolvix AI/ML Internship (Week 3: Deep Learning & NLP)**. The objective was to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification, stepping up from classical Machine Learning to Deep Learning. 

The dataset chosen for this challenge is **CIFAR-10**, which consists of 60,000 color images in 10 classes, presenting a solid stretch challenge compared to MNIST.

---

## 🚀 What Was Done
1. **Model Architecture:** Built a CNN using TensorFlow/Keras with multiple `Conv2D` and `MaxPooling2D` layers, followed by `Dense` fully connected layers.
2. **Training:** The model was trained on the CIFAR-10 training dataset for 12 epochs.
3. **Evaluation Curves:** Plotted the training vs. validation accuracy and loss curves to monitor for overfitting during the training process.
4. **Testing & Visualization:** Tested the trained model on sample images from the test set, visualizing the actual vs. predicted labels.

---

## 📊 Results & Performance
- **Final Test Accuracy:** **71.77%**
- The model successfully learned to identify objects across 10 complex categories despite the low resolution of the dataset.

### 📉 Evaluation Curves
*(Insert screenshot of your Accuracy & Loss graphs here)*
> **Note to myself:** Add the image file (e.g., `curves.png`) to this folder and replace this line with `![Training Curves](curves.png)`

### 🎯 Sample Predictions
*(Insert screenshot of your 5 sample test images with Pred/Act labels here)*
> **Note to myself:** Add the image file (e.g., `predictions.png`) to this folder and replace this line with `![Predictions](predictions.png)`

---

## 🛠️ Technologies Used
- **Language:** Python
- **Libraries:** TensorFlow, Keras, Matplotlib, NumPy
- **Environment:** Google Colab

---
*Created by **Muhammad Ahmad** during the Evolvix AI/ML Internship.*
