import os
import streamlit as st
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the image transformation
def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)

# Load dataset images
def load_dataset_images(dataset_path):
    images = []
    image_paths = []
    for file in os.listdir(dataset_path):
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(dataset_path, file)
            images.append(Image.open(image_path).convert("RGB"))
            image_paths.append(image_path)
    return images, image_paths

# Generate embeddings for all dataset images
def generate_embeddings(images, model, transform, device):
    embeddings = []
    for image in images:
        image_tensor = preprocess_image(image, transform).to(device)
        with torch.no_grad():
            embedding = model(image_tensor).cpu().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

# Streamlit app
def main():
    st.title("Image Similarity Finder")

    # Load the model (preloaded on the server)
    model_path = r"C:\Users\Sarthak Santra\Desktop\jewel-detection\mobilenet_jewels_model_v2_state_dict.pth"  # Replace with the actual path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=True).to(device)
    model.eval()
    st.success("Model loaded successfully!")

    # Assume the dataset is in the same folder as the model file
    dataset_path = r"C:\Users\Sarthak Santra\Desktop\jewel-detection\data"
    images, image_paths = load_dataset_images(dataset_path)
    if len(images) == 0:
        st.warning("No images found in the dataset folder!")
        return

    # Define image transformation (adjust as needed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
    ])

    # Generate embeddings for dataset images
    st.info("Generating embeddings for dataset images...")
    dataset_embeddings = generate_embeddings(images, model, transform, device)
    st.success("Embeddings generated!")

    # Upload input image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        # Generate embedding for input image
        input_embedding = preprocess_image(input_image, transform).to(device)
        with torch.no_grad():
            input_embedding = model(input_embedding).cpu().numpy()

        # Compute similarity
        similarities = cosine_similarity(input_embedding, dataset_embeddings)[0]
        top_5_indices = np.argsort(similarities)[-5:][::-1]

        # Display top 5 similar images
        st.header("Top 5 Similar Images")
        for idx in top_5_indices:
            st.image(image_paths[idx], caption=f"Similarity: {similarities[idx]:.4f}", use_column_width=True)


if __name__ == "__main__":
    main()
