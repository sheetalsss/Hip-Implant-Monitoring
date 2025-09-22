import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="Corrosion Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and label encoder"""
    try:
        with open('model/corrosion_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load model
model, le = load_model()

# Header
st.markdown('<h1 class="main-header">🔬 Hip Implant Corrosion Detection</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Monitoring System for Medical Implants")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-heart.png", width=80)
    st.title("Settings")

    st.subheader("Model Information")
    if model and le:
        st.success("✅ Model loaded successfully")
        st.write(f"**Classes:** {', '.join(le.classes_)}")
    else:
        st.error("❌ Model not loaded")
        st.info("Using demo mode with sample predictions")

    st.subheader("Sample Values")
    if st.button("Load Sample Data"):
        st.session_state.acoustic = 123.45
        st.session_state.electrochemical = 67.89
        st.session_state.mechanical = 12.34

    st.subheader("History")
    if st.session_state.prediction_history:
        for i, pred in enumerate(st.session_state.prediction_history[-5:]):
            st.write(f"{i + 1}. {pred['prediction']} ({pred['timestamp']})")
    else:
        st.write("No predictions yet")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📊 Input Parameters")

    # Input fields
    acoustic = st.number_input(
        "Acoustic Emission Data",
        # min_value=0.0,
        # max_value=1000.0,
        value=123.45,
        step=0.01,
        key="acoustic",
        help="Vibration intensity measurements from sensors"
    )

    electrochemical = st.number_input(
        "Electrochemical Data",
        # min_value=0.0,
        # max_value=1000.0,
        value=67.89,
        step=0.01,
        key="electrochemical",
        help="Corrosion potential readings"
    )

    mechanical = st.number_input(
        "Mechanical Data",
        # min_value=0.0,
        # max_value=1000.0,
        value=12.34,
        step=0.01,
        key="mechanical",
        help="Stress and strain measurements"
    )

    # Predict button
    if st.button("🚀 Predict Corrosion Type", use_container_width=True):
        with st.spinner("Analyzing sensor data..."):
            time.sleep(0.5)  # Simulate processing

            # Prepare input data
            input_data = np.array([[acoustic, electrochemical, mechanical]])

            if model and le:
                # Real prediction
                try:
                    prediction = model.predict(input_data)[0]
                    prediction_label = le.inverse_transform([prediction])[0]

                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(input_data)[0]
                        confidence_scores = {
                            le.inverse_transform([i])[0]: float(prob)
                            for i, prob in enumerate(probabilities)
                        }
                    else:
                        confidence_scores = {prediction_label: 1.0}

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    prediction_label = "anodic"  # Fallback
                    confidence_scores = {"anodic": 0.8, "cathodic": 0.1, "corrosion": 0.1}
            else:
                # Demo prediction
                avg_value = np.mean([acoustic, electrochemical, mechanical])
                if avg_value > 100:
                    prediction_label = "anodic"
                    confidence_scores = {"anodic": 0.8, "cathodic": 0.1, "corrosion": 0.1}
                elif avg_value > 50:
                    prediction_label = "cathodic"
                    confidence_scores = {"anodic": 0.1, "cathodic": 0.8, "corrosion": 0.1}
                else:
                    prediction_label = "corrosion"
                    confidence_scores = {"anodic": 0.1, "cathodic": 0.1, "corrosion": 0.8}

            # Store prediction
            prediction_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_data": {
                    "acoustic": acoustic,
                    "electrochemical": electrochemical,
                    "mechanical": mechanical
                },
                "prediction": prediction_label,
                "confidence_scores": confidence_scores,
                "confidence": max(confidence_scores.values())
            }

            st.session_state.prediction_history.append(prediction_data)
            st.session_state.last_prediction = prediction_data

with col2:
    st.header("📈 Prediction Results")

    if 'last_prediction' in st.session_state:
        prediction_data = st.session_state.last_prediction

        # Display prediction
        st.markdown(f"""
        <div class="prediction-card">
            <h2>Prediction: {prediction_data['prediction'].upper()}</h2>
            <h3>Confidence: {prediction_data['confidence'] * 100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # Confidence scores
        st.subheader("Confidence Scores")

        # Create bar chart
        df_scores = pd.DataFrame({
            'Class': list(prediction_data['confidence_scores'].keys()),
            'Confidence': list(prediction_data['confidence_scores'].values())
        })

        fig = px.bar(df_scores, x='Class', y='Confidence',
                     color='Confidence', color_continuous_scale='Viridis',
                     title="Prediction Confidence by Class")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        st.subheader("Input Data Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Acoustic", f"{acoustic:.2f}")
        with col2:
            st.metric("Electrochemical", f"{electrochemical:.2f}")
        with col3:
            st.metric("Mechanical", f"{mechanical:.2f}")

        # Raw data
        with st.expander("View Raw Data"):
            st.json(prediction_data)

    else:
        st.info("👆 Enter values and click 'Predict' to see results")

# Additional features
st.markdown("---")
st.header("📊 Advanced Analysis")

tab1, tab2, tab3 = st.tabs(["History", "Statistics", "3D Visualization"])

with tab1:
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No prediction history yet")

with tab2:
    if st.session_state.prediction_history:
        # Count predictions
        pred_counts = pd.Series([p['prediction'] for p in st.session_state.prediction_history]).value_counts()
        fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index,
                         title="Prediction Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Collect some predictions to see statistics")

with tab3:
    # 3D scatter plot
    if st.session_state.prediction_history:
        plot_data = []
        for pred in st.session_state.prediction_history:
            plot_data.append({
                'Acoustic': pred['input_data']['acoustic'],
                'Electrochemical': pred['input_data']['electrochemical'],
                'Mechanical': pred['input_data']['mechanical'],
                'Prediction': pred['prediction'],
                'Confidence': pred['confidence']
            })

        df_plot = pd.DataFrame(plot_data)
        fig_3d = px.scatter_3d(df_plot, x='Acoustic', y='Electrochemical', z='Mechanical',
                               color='Prediction', size='Confidence',
                               title="3D Data Visualization",
                               hover_data=['Confidence'])
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Make some predictions to enable 3D visualization")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 <b>Hip Implant Monitoring System</b> | Powered by Machine Learning</p>
    <p>For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)