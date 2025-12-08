import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os
import gdown
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bone Fracture Classification System",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = ['distal-fracture', 'non-fracture', 'post-fracture', 'proximal-fracture']
IMG_SIZE = (224, 224)

# 🔹 NEW: options for primary diagnosis (UI)
PRIMARY_DIAGNOSIS_OPTIONS = ["unknown"] + CLASS_NAMES

# Use relative path for deployment compatibility
BASE_DIR = Path(__file__).parent  # Same directory as app.py
MODELS_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Default Model URL (Configure your model URL here)
DEFAULT_MODEL_URL = "https://drive.google.com/file/d/1731iJjX5LsxeaoM37sUP2lKIxhcsnUEz/view?usp=drive_link"

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'metadata_feature_cols' not in st.session_state:
    st.session_state.metadata_feature_cols = None
if 'gdrive_model_path' not in st.session_state:
    st.session_state.gdrive_model_path = None

# Helper Functions
@st.cache_resource
def load_preprocessing_objects():
    """Load label encoders and scaler"""
    try:
        with open(BASE_DIR / 'label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return label_encoders, scaler
    except Exception as e:
        st.warning(f"⚠️ Preprocessing objects not found. Using defaults. ({str(e)})")
        return None, None

@st.cache_resource
def load_trained_model(model_path):
    """Load a trained model with TensorFlow version compatibility"""
    try:
        from tensorflow.keras.layers import DepthwiseConv2D

        class CompatibleDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        custom_objects = {
            'DepthwiseConv2D': CompatibleDepthwiseConv2D
        }

        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def download_model_from_url(url):
    """Download model from URL (Google Drive or direct link)"""
    try:
        import urllib.request

        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'model.h5')

        if 'drive.google.com' in url:
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                st.error("Invalid Google Drive URL format")
                return None

            download_url = f'https://drive.google.com/uc?id={file_id}'

            with st.spinner('Downloading model from Google Drive...'):
                gdown.download(download_url, output_path, quiet=False)
        else:
            with st.spinner('Downloading model from URL...'):
                def reporthook(count, block_size, total_size):
                    if total_size > 0:
                        percent = int(count * block_size * 100 / total_size)
                        st.write(f"Download progress: {percent}%")

                urllib.request.urlretrieve(url, output_path, reporthook)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error("Downloaded file is empty or doesn't exist")
            return None

    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return None

def preprocess_xray_image(image, apply_clahe=True, apply_blur=True):
    """Preprocess X-ray image with medical imaging techniques"""
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img = image.copy()

    original = img.copy()
    processed = img.copy()

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    if apply_blur:
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

    processed = processed.astype(np.float32) / 255.0

    return original, processed

def prepare_image_for_model(image):
    """Prepare image for model prediction"""
    _, processed = preprocess_xray_image(image)
    processed_uint8 = (processed * 255).astype(np.uint8)
    img_resized = cv2.resize(processed_uint8, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    return img_normalized, img_rgb

def generate_gradcam(model, img_array, meta_array, class_idx, layer_name=None):
    """Generate Grad-CAM heatmap"""
    try:
        if layer_name is None:
            for layer in reversed(model.layers):
                try:
                    if len(layer.output.shape) == 4:
                        layer_name = layer.name
                        break
                except:
                    continue

        if layer_name is None:
            st.warning("No convolutional layer found for Grad-CAM")
            return None

        try:
            grad_model = Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer_name).output, model.output]
            )
        except Exception as e:
            st.error(f"Error creating gradient model: {str(e)}")
            return None

        try:
            with tf.GradientTape() as tape:
                outputs = grad_model([img_array, meta_array], training=False)
                if isinstance(outputs, list):
                    conv_outputs = outputs[0]
                    predictions = outputs[1]
                else:
                    conv_outputs, predictions = outputs

                loss = predictions[0][class_idx]
        except Exception as e:
            st.error(f"Error in forward pass: {str(e)}")
            return None

        try:
            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                st.warning("Could not compute gradients for Grad-CAM")
                return None
        except Exception as e:
            st.error(f"Error computing gradients: {str(e)}")
            return None

        try:
            conv_outputs_val = conv_outputs.numpy()
            grads_val = grads.numpy()
            pooled_grads = np.mean(grads_val, axis=(1, 2))[0]
            conv_output = conv_outputs_val[0]

            heatmap = np.zeros(conv_output.shape[:2], dtype=np.float32)
            for i in range(len(pooled_grads)):
                heatmap += pooled_grads[i] * conv_output[:, :, i]

            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            return heatmap
        except Exception as e:
            st.error(f"Error processing heatmap: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None

def generate_lime_explanation(model, img_array, meta_array, num_samples=1000):
    """Generate LIME explanation"""
    try:
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            processed_images = []
            for img in images:
                img_norm = img.astype(np.float32) / 255.0 if img.max() > 1 else img
                processed_images.append(img_norm)

            processed_images = np.array(processed_images)
            meta_batch = np.repeat(meta_array, len(images), axis=0)
            predictions = model.predict([processed_images, meta_batch], verbose=0)
            return predictions

        img_for_lime = (img_array[0] * 255).astype(np.uint8)
        explanation = explainer.explain_instance(
            img_for_lime,
            predict_fn,
            top_labels=4,
            hide_color=0,
            num_samples=num_samples
        )

        return explanation
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")
        return None

def process_metadata_input(metadata_dict, label_encoders, scaler):
    """Process metadata input from user"""
    try:
        # NOTE: we keep fracture_type as in training code
        categorical_cols = ['gender', 'bone_type', 'left_right', 'gap_visibility', 'fracture_type']
        numerical_cols = ['age', 'bone_width', 'fracture_gap']

        features = []

        for col in categorical_cols:
            if col in metadata_dict:
                value = metadata_dict[col]

                if label_encoders and col in label_encoders:
                    if value in label_encoders[col].classes_:
                        encoded = label_encoders[col].transform([value])[0]
                    else:
                        encoded = 0
                else:
                    simple_mapping = {
                        'male': 0, 'female': 1, 'unknown': 2,
                        'humerus': 0, 'radius': 1, 'ulna': 2, 'femur': 3, 'tibia': 4, 'fibula': 5,
                        'left': 0, 'right': 1,
                        'visible': 1, 'not_visible': 0,
                        'distal-fracture': 0, 'proximal-fracture': 1, 'post-fracture': 2, 'non-fracture': 3
                    }
                    encoded = simple_mapping.get(value, 0)

                features.append(encoded)

        numerical_values = []
        for col in numerical_cols:
            if col in metadata_dict:
                numerical_values.append(metadata_dict[col])
            else:
                numerical_values.append(0)

        if scaler:
            numerical_scaled = scaler.transform([numerical_values])[0]
        else:
            numerical_scaled = [
                numerical_values[0] / 100.0,
                numerical_values[1] / 50.0,
                numerical_values[2] / 20.0
            ]

        features.extend(numerical_scaled)

        return np.array(features, dtype=np.float32)
    except Exception as e:
        st.error(f"Error processing metadata: {str(e)}")
        return None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🦴 Bone Fracture Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Multimodal Fracture Detection with Interpretability</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        st.subheader("1. Select Model Source")

        options = ["Local Files", "URL (Google Drive or Direct)"]
        if DEFAULT_MODEL_URL:
            options.insert(0, "Default Model (Hardcoded)")

        model_source = st.radio("Load model from:", options)

        if model_source == "Default Model (Hardcoded)":
            st.info(f"📌 Using configured default model")
            st.code(DEFAULT_MODEL_URL, language=None)

            if st.button("Load Default Model"):
                model_path = download_model_from_url(DEFAULT_MODEL_URL)

                if model_path:
                    st.session_state.gdrive_model_path = model_path

                    with st.spinner("Loading default model..."):
                        st.session_state.model = load_trained_model(model_path)
                        st.session_state.label_encoders, st.session_state.scaler = load_preprocessing_objects()

                        if st.session_state.model:
                            st.success("✅ Default model loaded successfully!")
                        else:
                            st.error("❌ Failed to load model")
                else:
                    st.error("❌ Failed to download default model")

        elif model_source == "Local Files":
            model_files = list(MODELS_DIR.glob("*.h5"))

            if model_files:
                model_names = [f.stem for f in model_files]
                selected_model = st.selectbox("Choose a trained model:", model_names)

                if st.button("Load Model"):
                    with st.spinner("Loading model..."):
                        model_path = MODELS_DIR / f"{selected_model}.h5"
                        st.session_state.model = load_trained_model(model_path)
                        st.session_state.label_encoders, st.session_state.scaler = load_preprocessing_objects()

                        if st.session_state.model:
                            st.success(f"✅ Model '{selected_model}' loaded successfully!")
                        else:
                            st.error("❌ Failed to load model")
            else:
                st.warning("⚠️ No trained models found in the models directory")

        else:  # URL
            st.info("📌 Paste your model URL below")
            st.markdown("**Supported:**")
            st.markdown("- Google Drive shareable links")
            st.markdown("- Direct download URLs (.h5 files)")

            model_url = st.text_input(
                "Model URL:",
                placeholder="https://example.com/model.h5 or Google Drive link",
                help="Paste a direct download URL or Google Drive shareable link"
            )

            if st.button("Download & Load Model"):
                if model_url:
                    model_path = download_model_from_url(model_url)

                    if model_path:
                        st.session_state.gdrive_model_path = model_path

                        with st.spinner("Loading downloaded model..."):
                            st.session_state.model = load_trained_model(model_path)
                            st.session_state.label_encoders, st.session_state.scaler = load_preprocessing_objects()

                            if st.session_state.model:
                                st.success("✅ Model downloaded and loaded successfully!")
                            else:
                                st.error("❌ Failed to load model")
                    else:
                        st.error("❌ Failed to download model from URL")
                else:
                    st.warning("⚠️ Please enter a model URL")

        st.divider()

        # Interpretability options
        st.subheader("2. Interpretability Options")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        show_lime = st.checkbox("Show LIME", value=True)
        lime_samples = st.slider("LIME Samples", 100, 2000, 1000, 100) if show_lime else 1000

    if st.session_state.model is None:
        st.info("👈 Please load a model from the sidebar to begin")
        return

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Batch Analysis", "ℹ️ About"])

    with tab1:
        st.header("Single Image Prediction")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Image Input")
            uploaded_file = st.file_uploader("Upload X-ray image", type=['jpg', 'jpeg', 'png'])

            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Uploaded X-ray", use_container_width=True)

        with col2:
            st.subheader("📋 Patient Metadata")

            with st.form("metadata_form"):
                col_a, col_b = st.columns(2)

                with col_a:
                    age = st.number_input("Age", min_value=0, max_value=120, value=50)
                    gender = st.selectbox("Gender", ["male", "female", "unknown"])
                    bone_type = st.selectbox(
                        "Bone Type",
                        ["humerus", "radius", "ulna", "femur", "tibia", "fibula", "elbow", "unknown"]
                    )

                with col_b:
                    left_right = st.selectbox("Side", ["left", "right", "unknown"])
                    bone_width = st.number_input(
                        "Bone Width (mm)", min_value=0.0, max_value=100.0, value=20.0, step=0.1
                    )
                    fracture_gap = st.number_input(
                        "Fracture Gap (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.1
                    )
                    gap_visibility = st.selectbox("Gap Visibility", ["visible", "not_visible", "unknown"])

                # 🔹 NEW: Primary Diagnosis field
                primary_diagnosis = st.selectbox(
                    "Primary Diagnosis (clinical / radiologist)",
                    PRIMARY_DIAGNOSIS_OPTIONS,
                    index=0,
                    help="Select the clinician's primary diagnosis (optional)."
                )

                submit_button = st.form_submit_button("🔍 Analyze")

        if uploaded_file and submit_button:
            with st.spinner("Analyzing..."):
                img_normalized, img_rgb = prepare_image_for_model(image_rgb)
                img_batch = np.expand_dims(img_normalized, axis=0)

                # 🔹 NEW: use primary_diagnosis to populate fracture_type metadata
                if primary_diagnosis in CLASS_NAMES:
                    fracture_type_meta = primary_diagnosis
                else:
                    fracture_type_meta = "unknown"

                metadata_dict = {
                    'age': age,
                    'gender': gender,
                    'bone_type': bone_type,
                    'left_right': left_right,
                    'bone_width': bone_width,
                    'fracture_gap': fracture_gap,
                    'gap_visibility': gap_visibility,
                    'fracture_type': fracture_type_meta,   # used by model
                    'primary_diagnosis': primary_diagnosis  # UI-only (ignored in process_metadata_input)
                }

                meta_features = process_metadata_input(
                    metadata_dict,
                    st.session_state.label_encoders,
                    st.session_state.scaler
                )

                if meta_features is not None:
                    meta_batch = np.expand_dims(meta_features, axis=0)

                    predictions = st.session_state.model.predict([img_batch, meta_batch], verbose=0)
                    pred_class_idx = np.argmax(predictions[0])
                    pred_class = CLASS_NAMES[pred_class_idx]
                    pred_confidence = predictions[0][pred_class_idx]

                    st.divider()
                    st.header("🎯 Prediction Results")

                    col1_m, col2_m, col3_m, col4_m = st.columns(4)

                    with col1_m:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Predicted Class</h3>
                            <h2>{pred_class}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2_m:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Confidence</h3>
                            <h2>{pred_confidence*100:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3_m:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Patient Age</h3>
                            <h2>{age} years</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    # 🔹 NEW: show primary diagnosis (if provided)
                    with col4_m:
                        display_pd = primary_diagnosis if primary_diagnosis != "unknown" else "Not provided"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Primary Diagnosis</h3>
                            <h2>{display_pd}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    st.subheader("📊 Class Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': CLASS_NAMES,
                        'Probability': predictions[0]
                    }).sort_values('Probability', ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.barh(
                        prob_df['Class'],
                        prob_df['Probability'],
                        color=['#667eea' if c == pred_class else '#cbd5e0' for c in prob_df['Class']]
                    )
                    ax.set_xlabel('Probability')
                    ax.set_title('Prediction Probabilities for All Classes')
                    ax.set_xlim(0, 1)

                    for i, (idx, row) in enumerate(prob_df.iterrows()):
                        ax.text(row['Probability'] + 0.02, i, f"{row['Probability']:.3f}",
                                va='center', fontweight='bold')

                    st.pyplot(fig)
                    plt.close()

                    st.divider()
                    st.header("🔍 Model Interpretability")

                    interp_cols = []
                    if show_gradcam:
                        interp_cols.append("Grad-CAM")
                    if show_lime:
                        interp_cols.append("LIME")

                    if interp_cols:
                        cols_interp = st.columns(len(interp_cols))

                        if show_gradcam:
                            with cols_interp[0]:
                                st.subheader("Grad-CAM Visualization")
                                with st.spinner("Generating Grad-CAM..."):
                                    heatmap = generate_gradcam(
                                        st.session_state.model,
                                        img_batch,
                                        meta_batch,
                                        pred_class_idx
                                    )

                                    if heatmap is not None:
                                        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

                                        fig, ax = plt.subplots(figsize=(6, 6))
                                        ax.imshow(img_rgb, cmap='gray')
                                        ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
                                        ax.set_title(f'Grad-CAM for {pred_class}')
                                        ax.axis('off')

                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.warning("Grad-CAM not available for this model")

                        if show_lime:
                            with cols_interp[-1]:
                                st.subheader("LIME Explanation")
                                with st.spinner(f"Generating LIME (using {lime_samples} samples)..."):
                                    explanation = generate_lime_explanation(
                                        st.session_state.model,
                                        img_batch,
                                        meta_batch,
                                        num_samples=lime_samples
                                    )

                                    if explanation is not None:
                                        temp, mask = explanation.get_image_and_mask(
                                            pred_class_idx,
                                            positive_only=False,
                                            num_features=10,
                                            hide_rest=False
                                        )

                                        fig, ax = plt.subplots(figsize=(6, 6))
                                        ax.imshow(mark_boundaries(temp / 255.0, mask))
                                        ax.set_title(f'LIME Explanation for {pred_class}')
                                        ax.axis('off')

                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.warning("LIME explanation failed")

    with tab2:
        st.header("Batch Analysis from CSV")
        st.info("📁 Upload a CSV file with patient metadata and image paths for batch prediction")

        csv_file = st.file_uploader("Upload CSV file", type=['csv'])

        if csv_file:
            df = pd.read_csv(csv_file)
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))

            if st.button("🚀 Run Batch Prediction"):
                st.warning("⚠️ Batch prediction feature coming soon!")

    with tab3:
        st.header("About This Application")

        st.markdown("""
        ### 🦴 Bone Fracture Classification System
        
        This application uses advanced deep learning models for multimodal bone fracture classification.
        
        #### 🎯 Features:
        - **Multimodal Input**: Combines X-ray images with patient metadata
        - **Multiple Model Support**: Choose from various trained architectures
        - **Interpretability**: Grad-CAM and LIME visualizations
        - **Real-time Prediction**: Instant classification results
        - **Batch Processing**: Analyze multiple cases at once
        - **Primary Diagnosis Field**: Optionally record clinician's primary diagnosis for comparison
        
        #### 🏥 Supported Fracture Types:
        1. **Distal Fracture**
        2. **Proximal Fracture**
        3. **Post-Fracture**
        4. **Non-Fracture**
        
        ---
        
        *This tool is for research purposes only and should not be used for clinical diagnosis without proper validation.*
        """)

if __name__ == "__main__":
    main()
