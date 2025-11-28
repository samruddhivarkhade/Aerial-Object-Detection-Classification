# Aerial-Object-Detection-Classification


This project is an **Aerial Object Classification System** that identifies whether an uploaded aerial image contains a **Bird** or a **Drone** using a deep learning model built with **PyTorch**. The dataset is preprocessed with normalization, resizing, and augmentation to improve accuracy, and a fine-tuned **ResNet-18** model performs the classification.

The system is deployed through an easy-to-use **Streamlit web app**, where users can upload images and instantly view predictions with confidence scores. Designed to be simple, accurate, and fast, this project demonstrates a complete pipeline from dataset preparation to model training and real-time deployment.

# 🚀 Features
 
 Upload aerial images directly from your device
 Deep learning–based image classification
 Fast and interactive Streamlit interface
 Displays prediction label & probability score
 Trained model stored and loaded automatically
 Deployable on Streamlit Cloud

# Model Overview

The model was trained using a custom aerial dataset containing images categorized into multiple classes.
Training involved:
Data preprocessing (resize, normalization)
CNN-based architecture
Train/validation splits
Performance evaluation using accuracy & loss metrics

# ▶️ How to Run Locally

1️⃣ Clone the repository
git clone https://github.com/your-username/aerial-object-classification.git
cd aerial-object-classification

2️⃣ Create a virtual environment
python -m venv venv
Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run streamlit_app.py

