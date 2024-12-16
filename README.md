# VGG16-LoRA-Brain-Tumor-Classifier
The project utilizes Convolutional Neural Networks (CNNs), specifically the VGG16 architecture, for tumor classification and Low-Rank Adaptation (LoRA) to optimize computational efficiency. The dataset used for training and testing includes MRI scans from different tumor types, such as glioma, meningioma, pituitary tumor, and no tumor.

# Brain Tumor Classification using VGG16 and LoRA

This repository contains a Python-based deep learning project for classifying brain tumors from MRI images using Convolutional Neural Networks (CNNs). The project uses the VGG16 model architecture for feature extraction and classification, along with the Low-Rank Adaptation (LoRA) technique to optimize training efficiency.

## Table of Contents

- [VGG16-LoRA-Brain-Tumor-Classifier](#vgg16-lora-brain-tumor-classifier)
- [Brain Tumor Classification using VGG16 and LoRA](#brain-tumor-classification-using-vgg16-and-lora)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Requirements](#requirements)
  - [Dataset](#dataset)
  - [how-to-run](#how-to-run)

## Overview

Brain tumor classification plays a crucial role in early diagnosis and treatment planning. This repository provides an implementation for classifying four categories of brain tumors: glioma, meningioma, pituitary tumor, and no tumor, using MRI scans. The model is trained with a modified VGG16 architecture to handle the complexity and variety of MRI images.

## Installation

To get started, clone this repository:

```bash
git clone https://github.com/once28/VGG16-LoRA-Brain-Tumor-Classifier.git
cd VGG16-LoRA-Brain-Tumor-Classifier
```

## Requirements
Python 3.x
TensorFlow (>= 2.0)
Keras
NumPy
Matplotlib
pandas
scikit-learn
You can install these dependencies using the following command:

```bash
pip install tensorflow keras numpy matplotlib pandas scikit-learn
```

## Dataset
The dataset used in this project is the Kaggle Brain Tumor Classification dataset, which contains MRI images labeled as:

- glioma_tumor
- meningioma_tumor
- pituitary_tumor
- no_tumor
These images are preprocessed to enhance contrast and standardize dimensions for better feature detection. You can download the dataset from Kaggle [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).

## how-to-run
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/once28/VGG16-LoRA-Brain-Tumor-Classifier.git
cd VGG16-LoRA-Brain-Tumor-Classifier
```
Download the dataset from Kaggle and place the images in the appropriate folder (e.g., archive/training/ and archive/testing/).

If you don't have access to the dataset, you can create an account on Kaggle and download it [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).
Open the notebook model.ipynb using Jupyter:

```bash
jupyter notebook model.ipynb 
```

Run the cells in the notebook to:
Load and preprocess the data.
Train the VGG16 model with LoRA.
Evaluate the model using performance metrics (accuracy, precision, recall, etc.).
The training process may take some time depending on your machine's computational resources.
