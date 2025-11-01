# Animal10 Image Classification

This repository presents a deep learning model for classifying animal images using the Animals-10 dataset.  
The project is implemented in PyTorch and trained on the ResNet-50 architecture with transfer learning.  
It includes data preprocessing, model training, evaluation, and a Gradio-based demo for real-time predictions.

---

## Overview

The model classifies images into ten animal categories:

- Dog  
- Cat  
- Elephant  
- Cow  
- Horse  
- Sheep  
- Chicken  
- Butterfly  
- Spider  
- Squirrel  

The training process fine-tunes a pretrained ResNet-50 model on this dataset and achieves strong performance.

---

## Key Features

- Transfer learning using **ResNet-50** pretrained on ImageNet  
- Data preprocessing and augmentation with **torchvision.transforms**  
- End-to-end training and validation pipeline  
- Real-time accuracy and loss tracking per epoch  
- Gradio web interface for easy inference  
- Fully compatible with **Google Colab**

---

## Dataset

Dataset: [Animals-10 Dataset on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

- ~28,000 labeled images across 10 categories  
- Images are resized to 224×224 and normalized  
- Dataset is split into training and testing sets  

---

## Model Details

- **Architecture:** ResNet-50  
- **Framework:** PyTorch  
- **Optimizer:** Adam (learning rate = 0.0001)  
- **Loss Function:** CrossEntropyLoss  
- **Batch Size:** 32  
- **Epochs:** 5  
- **Input Shape:** 3×224×224  

---

## Results

The model achieved approximately **97–99% training accuracy** after five epochs on GPU runtime.  
Accuracy may vary slightly depending on random seeds and dataset splits.

---

## Running the Notebook

1. Open the notebook in Google Colab:  
   [Open in Colab](https://colab.research.google.com/github/<your-username>/<your-repo-name>/blob/main/Animal10_Ai_Recognition.ipynb)

2. Upload your Kaggle API key (`kaggle.json`) when prompted.  
3. Enable GPU acceleration:  
   - **Runtime → Change runtime type → GPU**  
4. Run all cells sequentially to download the dataset, train the model, and launch the Gradio demo.

## File Structure

Animal10-Image-Classification/
│
├── Animal10_Ai_Recognition.ipynb # Main Jupyter/Colab notebook
├── README.md # Project documentation
└── animal_classifier.pth # (Optional) Saved model weights



---

## File Structure

