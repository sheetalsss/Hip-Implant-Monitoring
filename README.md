## 🔬 Hip Implant Monitoring System

A machine learning system for early detection of hip implant corrosion using multivariate sensor data

* https://img.shields.io/badge/Python-3.9%252B-blue
* https://img.shields.io/badge/LightGBM-4.1.0-green
* https://img.shields.io/badge/Accuracy-99.96%2525-brightgreen
* https://img.shields.io/badge/Streamlit-1.32.0-red

### 📖 Overview

This project develops a machine learning system for monitoring hip implant conditions using real-time sensor data from the University of Illinois College of Medicine. The system analyzes acoustic emission, electrochemical, and mechanical sensor data to detect early signs of implant corrosion with 99.96% accuracy.

### 🎯 Key Features

* 99.96% Accuracy: LightGBM model outperformed initial MLP approach (99%)
* Real-time Monitoring: Streamlit dashboard for clinical applications
* Multivariate Analysis: Processes acoustic, electrochemical, and mechanical sensor data
* Temporal Validation: Handles 293,635 time groups preventing data leakage
* Production Ready: End-to-end pipeline from data cleaning to deployment

### 📊 Performance Metrics

* Metric -	Score
* Accuracy -	99.96%
* Precision -	99.94%
* Recall -	99.94%
* F1-Score -	99.94%
* False Positives -	< 0.1%

### 🏥 Research Background

This project was developed in collaboration with the University of Illinois College of Medicine under the mentorship of Dr. Mathew T. Mathew. The work addresses critical needs in orthopedic implant monitoring using advanced machine learning techniques on real biomedical sensor data.

### 🛠️ Installation

```commandline
# Clone the repository
git clone https://github.com/yourusername/hip-implant-monitoring.git
cd hip-implant-monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

###  Quick Start

1. Install dependencies:
```commandline
pip install -r requirements.txt
```
2. Run the Streamlit app:
```commandline
streamlit run app.py
```
3. Access the dashboard:
   Open http://localhost:8501 in your browser

4. Input sensor data:
Enter acoustic, electrochemical, and mechanical values to get real-time predictions

### 📈 Usage Examples

#### Real-time Prediction
```commandline
# Sample input data
input_data = {
    "acoustic_emission": 123.45,
    "electrochemical": 67.89, 
    "mechanical": 12.34
}

# Get prediction
prediction = model.predict(input_data)
# Returns: {"prediction": "anodic", "confidence": 0.9996}
```

#### Batch Processing 
```commandline
# Process multiple samples
batch_data = [
    {"acoustic": 123.45, "electrochemical": 67.89, "mechanical": 12.34},
    {"acoustic": 234.56, "electrochemical": 78.90, "mechanical": 23.45}
]

predictions = model.predict_batch(batch_data)
```

### 🔧 Model Architecture

#### Data Processing Pipeline:

* Sensor Data Acquisition: Real-time data from hip implants
* Data Cleaning: Handle missing values and outliers
* Feature Engineering: Time-series feature extraction
* Temporal Validation: Group-aware cross-validation
* Model Training: LightGBM with hyperparameter optimization

#### Key Technical Features

* Temporal Validation: 293,635 time groups handled without data leakage
* Feature Importance: Acoustic emission data most predictive feature
* Class Balance: Handled imbalanced data through strategic sampling
* Real-time Processing: < 100ms prediction latency

### 📊 Dataset Information

* Source: University of Illinois College of Medicine
* Samples: 1,111,564 time series observations
* Time Groups: 293,635 unique temporal groups

Features:
1. Acoustic Emission Data
2. Electrochemical Data
3. Mechanical Data

Target Classes:
1. Anodic Corrosion
2. Cathodic Corrosion
3. General Corrosion

### 🎓 Research Methodology

* Model: LightGBM Classifier
* Accuracy: 99.96% (+0.96% improvement)
* Focus: Multivariate sensor fusion
* Innovation: Advanced temporal validation and data cleaning

### 📋 Requirements

```commandline
streamlit==1.32.0
lightgbm==4.1.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
plotly==5.18.0
matplotlib==3.7.0
```

### 📜 License

This project is for academic and research purposes. Data provided by University of Illinois College of Medicine.

### 🙏 Acknowledgments

Dr. Mathew T. Mathew - University of Illinois College of Medicine
University of Illinois for providing the sensor data
Research Team for initial collaboration and foundation

### 📞 Contact

* Sheetal
* Email: sheetalshenoy02@gmail.com

### 🎯 Future Enhancements

* Real-time sensor integration
* Mobile application development
* Cloud deployment for scalability
* Additional implant types support
* Clinical trial integration