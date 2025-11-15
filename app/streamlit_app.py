"""Streamlit web interface for activity recognition."""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.models import CNNModel, CNNLSTMModel, CNNGRUModel
from src.data import DataPreprocessor


# Page configuration
st.set_page_config(
    page_title="Activity Recognition",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Activity labels mapping
ACTIVITY_LABELS = {
    0: "Still",
    1: "Walking",
    2: "Running",
    3: "Biking",
    4: "Car",
    5: "Bus",
    6: "Train",
    7: "Subway"
}


@st.cache_resource
def load_model(model_type: str):
    """Load and cache model."""
    config = get_config()

    if model_type == "CNN":
        model = CNNModel(config.model, num_classes=8)
        weights_path = config.data.models_dir / "CNN.h5"
    elif model_type == "CNN-LSTM":
        model = CNNLSTMModel(config.model, num_classes=8)
        weights_path = config.data.models_dir / "CNN_LSTM.h5"
    else:  # CNN-GRU
        model = CNNGRUModel(config.model, num_classes=8)
        weights_path = config.data.models_dir / "CNN_GRU.h5"

    if weights_path.exists():
        model.load_weights(weights_path)
    else:
        st.warning(f"Model weights not found at {weights_path}. Using untrained model.")

    return model


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">📱 Smartphone Activity Recognition</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    model_type = st.sidebar.selectbox(
        "Select Model",
        ["CNN", "CNN-LSTM", "CNN-GRU"],
        help="Choose the deep learning model architecture"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application uses deep learning to recognize human activities "
        "from smartphone sensor data (accelerometer, gyroscope, magnetometer)."
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📊 Data Input")

        input_method = st.radio(
            "Choose input method:",
            ["Upload CSV File", "Manual Input", "Use Sample Data"],
            horizontal=True
        )

        if input_method == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload sensor data (CSV with 9 features)",
                type=["csv"],
                help="CSV file with 9 columns: Accel(x,y,z), Gyro(x,y,z), Mag(x,y,z)"
            )

            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file, header=None)
                st.success(f"Loaded {len(data)} samples")
                st.dataframe(data.head(), use_container_width=True)

                if st.button("🔮 Predict Activity", type="primary"):
                    with st.spinner("Loading model and making predictions..."):
                        model = load_model(model_type)
                        preprocessor = DataPreprocessor(get_config().data)

                        # Normalize and reshape
                        X = preprocessor.normalize_features(data)
                        X = preprocessor.reshape_for_cnn(X)

                        # Predict
                        predictions = model.get_model().predict(X, verbose=0)
                        predicted_classes = np.argmax(predictions, axis=1)
                        confidences = np.max(predictions, axis=1)

                        # Display results
                        st.success("Predictions complete!")

                        # Show prediction distribution
                        unique, counts = np.unique(predicted_classes, return_counts=True)
                        pred_df = pd.DataFrame({
                            "Activity": [ACTIVITY_LABELS.get(c, f"Class {c}") for c in unique],
                            "Count": counts,
                            "Percentage": (counts / len(predicted_classes) * 100).round(2)
                        })

                        st.subheader("📈 Activity Distribution")
                        st.dataframe(pred_df, use_container_width=True)

                        # Plot
                        fig = go.Figure(data=[
                            go.Bar(x=pred_df["Activity"], y=pred_df["Count"])
                        ])
                        fig.update_layout(
                            title="Predicted Activities",
                            xaxis_title="Activity",
                            yaxis_title="Count",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

        elif input_method == "Manual Input":
            st.info("Enter 9 sensor values (Accelerometer, Gyroscope, Magnetometer - x, y, z each)")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.subheader("Accelerometer")
                accel_x = st.number_input("Accel X", value=0.0, format="%.4f")
                accel_y = st.number_input("Accel Y", value=0.0, format="%.4f")
                accel_z = st.number_input("Accel Z", value=9.81, format="%.4f")

            with col_b:
                st.subheader("Gyroscope")
                gyro_x = st.number_input("Gyro X", value=0.0, format="%.4f")
                gyro_y = st.number_input("Gyro Y", value=0.0, format="%.4f")
                gyro_z = st.number_input("Gyro Z", value=0.0, format="%.4f")

            with col_c:
                st.subheader("Magnetometer")
                mag_x = st.number_input("Mag X", value=0.0, format="%.4f")
                mag_y = st.number_input("Mag Y", value=0.0, format="%.4f")
                mag_z = st.number_input("Mag Z", value=0.0, format="%.4f")

            if st.button("🔮 Predict Activity", type="primary"):
                with st.spinner("Making prediction..."):
                    model = load_model(model_type)
                    preprocessor = DataPreprocessor(get_config().data)

                    # Create input array
                    X = np.array([[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]])
                    X = preprocessor.reshape_for_cnn(X)

                    # Predict
                    prediction = model.get_model().predict(X, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])

                    # Display result
                    activity = ACTIVITY_LABELS.get(predicted_class, f"Class {predicted_class}")

                    st.success(f"Predicted Activity: **{activity}**")
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                    # Show all probabilities
                    st.subheader("All Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Activity": [ACTIVITY_LABELS.get(i, f"Class {i}") for i in range(len(prediction[0]))],
                        "Probability": prediction[0] * 100
                    }).sort_values("Probability", ascending=False)

                    st.dataframe(prob_df, use_container_width=True)

        else:  # Use Sample Data
            st.info("Using sample data for demonstration")
            st.write("Sample data will be generated with typical sensor values")

            if st.button("🔮 Generate and Predict", type="primary"):
                with st.spinner("Generating sample data and making predictions..."):
                    # Generate sample data (100 samples)
                    np.random.seed(42)
                    sample_data = np.random.randn(100, 9)

                    model = load_model(model_type)
                    preprocessor = DataPreprocessor(get_config().data)

                    # Normalize and reshape
                    data_df = pd.DataFrame(sample_data)
                    X = preprocessor.normalize_features(data_df)
                    X = preprocessor.reshape_for_cnn(X)

                    # Predict
                    predictions = model.get_model().predict(X, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)

                    # Display results
                    unique, counts = np.unique(predicted_classes, return_counts=True)
                    pred_df = pd.DataFrame({
                        "Activity": [ACTIVITY_LABELS.get(c, f"Class {c}") for c in unique],
                        "Count": counts
                    })

                    st.subheader("Activity Distribution")
                    st.dataframe(pred_df, use_container_width=True)

    with col2:
        st.header("ℹ️ Model Info")

        st.markdown(f"**Selected Model:** {model_type}")

        st.markdown("---")

        st.subheader("📋 Features")
        st.markdown("""
        The model uses 9 sensor features:
        - **Accelerometer** (x, y, z)
        - **Gyroscope** (x, y, z)
        - **Magnetometer** (x, y, z)
        """)

        st.markdown("---")

        st.subheader("🎯 Activities")
        for idx, activity in ACTIVITY_LABELS.items():
            st.markdown(f"**{idx}:** {activity}")

        st.markdown("---")

        st.subheader("🏗️ Architecture")
        if model_type == "CNN":
            st.markdown("""
            - VGG-16 inspired CNN
            - 4 Conv1D layers
            - 2 MaxPool layers
            - Dense layers (512, 256)
            """)
        elif model_type == "CNN-LSTM":
            st.markdown("""
            - Hybrid architecture
            - CNN for feature extraction
            - LSTM for temporal modeling
            - Dropout for regularization
            """)
        else:
            st.markdown("""
            - Hybrid architecture
            - CNN for feature extraction
            - GRU for temporal modeling
            - Lighter than LSTM
            """)


if __name__ == "__main__":
    main()
