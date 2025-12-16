import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import preprocessing
import segmentation
import classifier

# Set page config
st.set_page_config(page_title="Object Recognition System", layout="wide")

st.markdown("""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #2d3436;
    }

    /* Background Logic */
    .reportview-container {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar Glassmorphism */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Main Content Styling */
    h1 {
        font-weight: 700;
        color: #2d3436;
        text-align: center;
        padding: 20px;
        background: rgba(255,255,255,0.7);
        border-radius: 15px;
        backdrop-filter: blur(5px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 30px;
    }

    h2, h3, h4 {
        color: #2d3436;
        font-weight: 600;
    }

    /* Cards / Containers */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stImage:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.25);
    }

    /* Buttons with Pulse Animation */
    .stButton>button {
        background: linear-gradient(45deg, #6c5ce7, #a29bfe);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.6);
        background: linear-gradient(45deg, #5b4cc4, #8e7ce6);
    }

    /* File Uploader */
    .stFileUploader > div > div > button {
        background-color: #00cec9;
        color: white;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 36px;
        background: -webkit-linear-gradient(#00b894, #00cec9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.8);
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0,0,0,0.1);
        margin: 30px 0;
    }

</style>
""", unsafe_allow_html=True)

st.title("🔎 Object Recognition System (CV Project)")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load Image
    image = utils.load_image(uploaded_file)
    
    # Display Original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Pipeline Selection
    option = st.sidebar.selectbox(
        "Select Pipeline Stage",
        ("Preprocessing", "Segmentation", "Image Analysis", "Classification")
    )
    
    if option == "Preprocessing":
        st.header("1. Preprocessing")
        
        # Grayscale
        gray = preprocessing.to_gray(image)
        st.subheader("Grayscale")
        st.image(gray, clamp=True, channels='GRAY', use_container_width=True)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            # HSV
            hsv = preprocessing.to_hsv(image)
            st.subheader("HSV Color Space")
            st.image(hsv, use_container_width=True)
            
        with col_p2:
            # Equalized
            eq = preprocessing.equalize_histogram(image)
            st.subheader("Histogram Equalization")
            st.image(eq, use_container_width=True)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            # Gaussian
            st.subheader("Gaussian Blur")
            k_gauss = st.slider("Kernel Size (Gaussian)", 1, 15, 5, step=2)
            gauss = preprocessing.gaussian_blur(image, k_gauss)
            st.image(gauss, use_container_width=True)
            
        with col_f2:
            # Median
            st.subheader("Median Blur")
            k_median = st.slider("Kernel Size (Median)", 1, 15, 5, step=2)
            median = preprocessing.median_blur(image, k_median)
            st.image(median, use_container_width=True)
            
        st.divider()
        st.subheader("Advanced Processing")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.write("**Sharpening**")
            sharp = preprocessing.sharpen_image(image)
            st.image(sharp, use_container_width=True)
        with col_a2:
            st.write("**Morphological (Dilation)**")
            dilated = preprocessing.morphological_ops(image, 'dilation')
            st.image(dilated, use_container_width=True)

    elif option == "Segmentation":
        st.header("2. Segmentation")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.subheader("Otsu Thresholding")
            thresh = segmentation.otsu_thresholding(image)
            st.image(thresh, clamp=True, channels='GRAY', use_container_width=True)
            
        with col_s2:
            st.subheader("K-Means Clustering")
            k_value = st.slider("Clusters (k)", 2, 10, 3)
            kmeans = segmentation.kmeans_segmentation(image, k=k_value)
            st.image(kmeans, use_container_width=True)
            
        with col_s3:
            st.subheader("Canny Edge Detection")
            t1 = st.slider("Threshold 1", 0, 255, 100)
            t2 = st.slider("Threshold 2", 0, 255, 200)
            edges = cv2.Canny(image, t1, t2)
            st.image(edges, clamp=True, channels='GRAY', use_container_width=True)
            
    elif option == "Image Analysis":
        st.header("3. Image Analysis")
        
        # Color Histogram
        st.subheader("Color Histogram")
        hist_features = preprocessing.extract_color_histogram(image)
        
        # Plot Histogram
        fig, ax = plt.subplots()
        ax.plot(hist_features)
        ax.set_title("Flattened Color Histogram")
        st.pyplot(fig)
        st.write(f"Feature Vector Size: {len(hist_features)}")



    elif option == "Classification":
        st.header("4. Classification")
        
        model_type = st.radio("Select Model Type", ("Pre-trained (MobileNetV2)", "Custom Model (Caltech-101)"))

        if st.button("Classify Object"):
            with st.spinner("Classifying..."):
                
                if model_type == "Pre-trained (MobileNetV2)":
                    model_cls = classifier.load_dl_model("MobileNetV2")
                    results = classifier.predict_dl_class(image, model_cls)
                    
                    st.success("Analysis Complete!")
                    for i, (label, prob) in enumerate(results):
                        st.write(f"**{i+1}. {label}**: {prob*100:.2f}%")
                        st.progress(prob)
                        
                else: # Custom Model
                    import os
                    model_path = "model/caltech101_model.h5"
                    label_path = "model/caltech101_labels.json"
                    
                    if os.path.exists(model_path):
                        model_custom = classifier.load_custom_model(model_path)
                        labels = classifier.load_labels(label_path)
                        
                        if model_custom:
                            label, prob = classifier.predict_custom_model(image, model_custom, labels)
                            st.success("Analysis Complete!")
                            st.metric(label="Predicted Class", value=label, delta=f"{prob*100:.2f}% Confidence")
                        else:
                            st.error("Failed to load custom model.")
                    else:
                        st.warning(f"Custom model not found at {model_path}. Please train it first using the notebook.")

else:
    st.info("Please upload an image to start.")
