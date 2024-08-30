import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import requests
import streamlit as st

# Create directories to store images
os.makedirs('pan_card_tampering', exist_ok=True)
os.makedirs('pan_card_tampering/Image', exist_ok=True)

# Streamlit UI setup
st.title("PAN Card Tampering Detection")

st.write("""
### Upload PAN Card Images to Compare
Upload the original and potentially tampered PAN card images to detect tampering.
""")

# Allow the user to upload images
uploaded_original = st.file_uploader("Upload Original PAN Card", type=["jpg", "png"])
uploaded_tampered = st.file_uploader("Upload Tampered PAN Card", type=["jpg", "png"])

if uploaded_original and uploaded_tampered:
    # Load images from the uploaded files
    original = Image.open(uploaded_original)
    tampered = Image.open(uploaded_tampered)

    # Display image formats and sizes
    st.write("**Original Image Format:** ", original.format)
    st.write("**Tampered Image Format:** ", tampered.format)
    st.write("**Original Image Size:** ", original.size)
    st.write("**Tampered Image Size:** ", tampered.size)

    # Resize images to a standard size
    original = original.resize((250, 160))
    tampered = tampered.resize((250, 160))

    # Save the images to the directories
    original.save('pan_card_tampering/Image/original.png')
    tampered.save('pan_card_tampering/Image/tampered.png')

    # Display the images in the Streamlit app
    st.image(original, caption="Original PAN Card", use_column_width=True)
    st.image(tampered, caption="Tampered PAN Card", use_column_width=True)

    # Convert images to grayscale using OpenCV
    original_cv2 = cv2.imread('pan_card_tampering/Image/original.png')
    tampered_cv2 = cv2.imread('pan_card_tampering/Image/tampered.png')
    original_gray = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered_cv2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the images
    (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")
    st.write(f"**SSIM Score:** {score:.4f}")

    # Apply threshold to the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours to highlight differences
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original_cv2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(tampered_cv2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the images with detected differences
    st.image(original_cv2, caption="Detected differences on Original PAN Card", use_column_width=True)
    st.image(tampered_cv2, caption="Detected differences on Tampered PAN Card", use_column_width=True)
    st.image(diff, caption="Difference Image", use_column_width=True)
    st.image(thresh, caption="Threshold Image", use_column_width=True)
