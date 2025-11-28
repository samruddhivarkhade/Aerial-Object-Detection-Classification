import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os

# -----------------------------------------------------
#               STREAMLIT PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="🛰️",
    layout="centered"
)

# -----------------------------------------------------
#              DEVICE CONFIGURATION
# -----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------
#              LOAD TRAINED RESNET MODEL
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: Bird, Drone

    model_path = "saved_models/resnet_best.pth"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# -----------------------------------------------------
#                  IMAGE TRANSFORMS
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = ["Bird", "Drone"]

# -----------------------------------------------------
#            PREDICTION FUNCTION
# -----------------------------------------------------
def predict(image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    return CLASS_NAMES[predicted.item()], confidence.item()

# -----------------------------------------------------
#               STREAMLIT FRONTEND UI
# -----------------------------------------------------
st.title("🛰️ Aerial Object Classifier")
st.write("Upload an image to classify whether it's a **Bird** or a **Drone**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        label, confidence = predict(image)

        st.success(f"### 🏷 Prediction: **{label}**")
        st.info(f"### 🔥 Confidence: **{confidence*100:.2f}%**")

        # Emoji indicators
        if label == "Bird":
            st.write("🕊 The model predicts a **Bird**.")
        else:
            st.write("🛸 The model predicts a **Drone**.")

# Footer
st.markdown("---")
st.markdown("Developed by **Samruddhi Varkhade** | Deep Learning Project")