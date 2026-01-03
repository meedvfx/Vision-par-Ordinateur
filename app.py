import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import apply_histogram_equalization, apply_gaussian_blur, apply_median_blur, convert_color_space, apply_sobel, apply_laplacian
from segmentation import apply_otsu_threshold, apply_kmeans_segmentation, apply_watershed_segmentation, apply_simple_threshold, apply_gmm_segmentation
from analysis import apply_canny_edge_detection, extract_geometric_features, draw_features, extract_color_histogram
from classifier import ImageClassifier

st.set_page_config(page_title="CV Mastery Project", layout="wide")

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('assets/style.css')

st.title("Vision Par Ordinateur - Du Pixel au Deep Learning")
st.markdown("""
Ce projet démontre les différentes étapes de la Vision par Ordinateur :
1. **Preprocessing** (Nettoyage)
2. **Segmentation** (Isolation)
3. **Analyse Classique** (Mesures)
4. **Deep Learning** (Classification)
""")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choisir le Niveau",
    ["1. Preprocessing (Nettoyer)", 
     "2. Segmentation (Isoler)", 
     "3. Analyse Classique (Mesurer)", 
     "4. Deep Learning (Classifier)"])

# File Uploader
uploaded_file = st.sidebar.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
else:
    st.info("Veuillez uploader une image pour commencer.")
    st.stop()

# Display Original
col1, col2 = st.columns(2)
with col1:
    st.subheader("Image Originale")
    st.image(image, use_container_width=True)

# Logic based on mode
if app_mode == "1. Preprocessing (Nettoyer)":
    st.sidebar.subheader("Hyperparamètres")
    
    op = st.sidebar.radio("Opération", ["Color Space Conversion", "Histogram Equalization (CLAHE)", "Gaussian Blur", "Median Blur"])
    
    processed_img = None
    
    if op == "Color Space Conversion":
        space = st.sidebar.selectbox("Espace", ["HSV", "YUV", "GRAY"])
        processed_img = convert_color_space(image_cv, space)
        
    elif op == "Histogram Equalization (CLAHE)":
        clip_limit = st.sidebar.slider("Clip Limit (Contraste)", 1.0, 10.0, 2.0)
        processed_img = apply_histogram_equalization(image_cv, clip_limit=clip_limit)
        
    elif op == "Gaussian Blur":
        k_size = st.sidebar.slider("Kernel Size (Impair)", 3, 21, 5, step=2)
        processed_img = apply_gaussian_blur(image_cv, kernel_size=k_size)
        
    elif op == "Median Blur":
        k_size = st.sidebar.slider("Kernel Size (Impair)", 3, 21, 5, step=2)
        processed_img = apply_median_blur(image_cv, kernel_size=k_size)

    elif op == "Sobel Filter":
        processed_img = apply_sobel(image_cv)
        
    elif op == "Laplacian Filter":
        processed_img = apply_laplacian(image_cv)
    
    if processed_img is not None:
        with col2:
            st.subheader("Résultat Preprocessing")
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

elif app_mode == "2. Segmentation (Isoler)":
    st.sidebar.subheader("Méthode")
    method = st.sidebar.selectbox("Algorithme", ["Simple Thresholding", "Otsu Thresholding", "K-Means Clustering", "GMM Clustering", "Watershed"])
    
    processed_img = None
    
    if method == "Simple Thresholding":
        thresh_val = st.sidebar.slider("Seuil", 0, 255, 127)
        processed_img = apply_simple_threshold(image_cv, thresh_val)

    elif method == "Otsu Thresholding":
        binary = apply_otsu_threshold(image_cv)
        processed_img = binary  # Grayscale result
        st.write("Otsu calcule automatiquement le seuil optimal.")
        
    elif method == "K-Means Clustering":
        k = st.sidebar.slider("Nombre de Clusters (K)", 2, 10, 3)
        processed_img = apply_kmeans_segmentation(image_cv, k=k)

    elif method == "GMM Clustering":
        n = st.sidebar.slider("Nombre de Composantes (GMM)", 2, 10, 3)
        processed_img = apply_gmm_segmentation(image_cv, n_components=n)
        
    elif method == "Watershed":
        img_w, markers = apply_watershed_segmentation(image_cv)
        processed_img = img_w
        st.sidebar.info("Les lignes rouges délimitent les objets.")

    if processed_img is not None:
        with col2:
            st.subheader("Résultat Segmentation")
            # Handle grayscale vs color
            if len(processed_img.shape) == 2:
                st.image(processed_img, use_container_width=True, channels="GRAY")
            else:
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

elif app_mode == "3. Analyse Classique (Mesurer)":
    st.sidebar.subheader("Détection de Contours")
    
    low_thresh = st.sidebar.slider("Seuil Bas (Canny)", 0, 255, 50)
    high_thresh = st.sidebar.slider("Seuil Haut (Canny)", 0, 255, 150)
    
    edges = apply_canny_edge_detection(image_cv, low_thresh, high_thresh)
    
    with col2:
        st.subheader("Contours (Canny)")
        st.image(edges, use_container_width=True, channels="GRAY")
        
    st.subheader("Extraction deactéristiques (Features)")
    if st.button("Calculer les propriétés"):
        features, contours = extract_geometric_features(edges)
        
        # Draw on original
        result_img = draw_features(image_cv, features, contours)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Contours détectés", use_container_width=True)
        
        st.write(f"Nombre d'objets détectés: {len(features)}")
        if len(features) > 0:
            import pandas as pd
            df = pd.DataFrame(features)
            # Remove the contour object column for display
            display_df = df.drop(columns=["contour"])
            st.dataframe(display_df)

    st.divider()
    st.subheader("Histogramme de Couleur")
    hists = extract_color_histogram(image_cv)
    if hists:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(hists[0], color='red', label='Red')
        ax.plot(hists[1], color='green', label='Green')
        ax.plot(hists[2], color='blue', label='Blue')
        ax.set_title("Répartition des intensités")
        st.pyplot(fig)

elif app_mode == "4. Deep Learning (Classifier)":
    st.sidebar.subheader("Configuration du Modèle")
    
    # NEW: Select Task
    task_type = st.sidebar.selectbox("Tâche / Dataset", 
                                     ["General (ImageNet)", 
                                      "CIFAR-10 Classification"])
    
    # 1. Choose Architecture
    if task_type == "CIFAR-10 Classification":
        # User requested ONLY Hybrid (MobileNet)
        arch_options = ["Hybrid (MobileNet)"]
    else:
        arch_options = ["ResNet50", "MobileNetV2", "InceptionV3", "Vision Transformer (ViT)"]
        
    model_arch = st.sidebar.selectbox("Architecture", arch_options)
    
    # 2. Choose Mode (Basic/ImageNet or Custom Weights/Local Models)
    # For CIFAR-10, we now ALWAYS look in 'models/' folder.
    model_path = None
    
    if task_type == "CIFAR-10 Classification":
        use_custom_weights = True # Implicitly true
        
        # Only one expected filename now
        expected_filename = "mobilenetv2_cifar10_animals.keras"
            
        model_path = os.path.join("models", expected_filename)
        
        # Check if exists
        if os.path.exists(model_path):
            st.sidebar.success(f"Modèle trouvé: {expected_filename}")
        else:
            st.sidebar.warning(f"Modèle non trouvé: {expected_filename}")
            st.sidebar.info(f"Veuillez placer le fichier '{expected_filename}' dans le dossier 'models'.")
            model_path = None # Prevent loading
            
    elif task_type == "General (ImageNet)" and (model_arch == "Hybrid (Custom)" or "Hybrid" in model_arch):
         # Legacy or fallback for ImageNet Hybrid if someone tries it
         use_custom_weights = True
         st.sidebar.info("Veuillez charger votre modèle.")
         # ... (fallthrough to existing uploader logic below for non-CIFAR cases or different setup)
    else:
        # Standard ImageNet cases
        use_custom_weights = st.sidebar.checkbox("Utiliser mes propres poids (.h5/.keras)", value=False)
    
    # Logic for File Uploader (ONLY if NOT CIFAR-10, as CIFAR-10 now uses local folder)
    if not (task_type == "CIFAR-10 Classification"):
        if use_custom_weights:
            uploaded_model = st.sidebar.file_uploader(f"Charger poids pour {model_arch}", type=["h5", "keras"])
            
            if uploaded_model:
                # Save temporarily, preserving extension
                file_ext = os.path.splitext(uploaded_model.name)[1]
                if not file_ext:
                     file_ext = ".h5" 
                
                # Sanitize filename components
                safe_arch = model_arch.replace(" ", "_").replace("(", "").replace(")", "")
                safe_task = task_type.split()[0]
                
                model_path = os.path.join("data", f"temp_{safe_arch}_{safe_task}{file_ext}")
                
                os.makedirs("data", exist_ok=True)
                
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                st.sidebar.success(f"Poids chargés: {uploaded_model.name}")
            else:
                 st.sidebar.warning("Veuillez uploader un fichier.")
    
    st.subheader(f"{task_type} - {model_arch}")
    
    # Block execution if custom weights needed but missing
    blocked = False
    if use_custom_weights and not model_path:
        st.info("En attente du fichier de poids...")
        blocked = True
        
    if not blocked:
        if st.button("Lancer la Classification"):
            with st.spinner(f"Chargement {model_arch} et Prédiction..."):
                try:
                    # Determine dataset tag for classifier
                    dataset_tag = 'imagenet'
                    if task_type == "CIFAR-10 Classification":
                        dataset_tag = 'cifar10'
                    elif use_custom_weights: 
                        dataset_tag = 'imagenet' # Custom weights on general = imagenet classes usually, or we could handle generic
                    
                    # Initialize classifier
                    classifier = ImageClassifier(model_name=model_arch, 
                                               model_path=model_path,
                                               dataset=dataset_tag)
                    
                    predictions = classifier.predict(image_cv)
                    
                    if predictions:
                        with col2:
                            st.subheader("Résultats")
                            for i, (uid, label, prob) in enumerate(predictions):
                                st.write(f"**{label}**: {prob*100:.2f}%")
                                st.progress(float(prob))
                            
                            if not use_custom_weights and dataset_tag == 'imagenet':
                                st.caption("Résultats basés sur ImageNet (1000 classes).")
                            elif dataset_tag == 'cifar10':
                                st.caption("Résultats basés sur CIFAR-10.")

                except OSError as e:
                    if "file signature not found" in str(e) or "Unable to synchronously open file" in str(e):
                        st.error("Erreur: Le fichier de poids semble invalide ou corrompu. Assurez-vous d'uploader un fichier .h5 ou .keras valide.")
                    else:
                        st.error(f"Erreur d'ouverture du fichier: {e}")
                except Exception as e:
                    st.error(f"Erreur lors de la classification: {e}")
                    st.exception(e)


