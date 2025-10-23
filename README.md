# Cropable - The Crop Protection App

An AI-powered web application for automated crop disease detection and treatment recommendations using Deep Convolutional Neural Networks (CNN).

## 🌾 Overview

Cropable addresses the critical challenge of crop disease detection in agriculture, where an estimated 10% of global production is lost annually to pests and pathogens. The application empowers farmers to identify crop diseases simply by uploading images of affected leaves and provides actionable treatment recommendations.

### Key Features

- **Automated Disease Detection**: Uses deep CNN models to identify 10 different tomato leaf diseases with high accuracy
- **Treatment Recommendations**: Content-based recommendation system provides targeted solutions for detected diseases
- **User-Friendly Interface**: Simple web interface designed following Norman's heuristics and Schneiderman's 8 Golden Rules
- **Real-Time Analysis**: Instant disease diagnosis from uploaded crop images

## 🎯 Project Objectives

- Develop an AI-powered web application for accurate crop disease identification
- Provide solution-based treatment measures for detected diseases
- Empower farmers to improve crop quality and reduce yield losses
- Make disease detection accessible without requiring laboratory equipment or expert knowledge

## 🏗️ Architecture

### System Components

1. **Frontend**: Flask-based web application with user authentication
2. **Backend**: Deep learning models (CNN-based) for image classification
3. **Recommendation Engine**: Content-based filtering using TF-IDF for treatment suggestions
4. **Data Pipeline**: Image preprocessing, augmentation, and validation

### Disease Detection Models

- **Sequential CNN Model**: Custom 2-layer convolutional architecture with ReLU activation
- **VGG16 Model**: Pre-trained 16-layer CNN for transfer learning
- Both models trained with:
  - Input shape: (64, 64, 3)
  - Batch size: 32
  - Epochs: 25
  - Optimizer: Adam
  - Loss function: Categorical cross-entropy

## 📊 Dataset

**Source**: Tomato Leaf Disease Detection Dataset (Kaggle)

### Disease Classes (10 total)
- Tomato Mosaic Virus
- Target Spot
- Bacterial Spot
- Tomato Yellow Leaf Curl Virus
- Late Blight
- Leaf Mold
- Early Blight
- Spider Mites (Two-spotted spider mite)
- Septoria Leaf Spot
- Healthy

**Data Split**:
- Training: ~67% (with augmentation)
- Validation: ~20%
- Testing: ~13%

## 🔧 Technical Implementation

### Data Processing Pipeline

1. **Data Augmentation**
   - Rotation, flipping, scaling, and translation
   - Prevents overfitting and improves model generalization

2. **Preprocessing**
   - Image normalization
   - Grayscale conversion and histogram equalization
   - Noise reduction using filters

3. **Model Training**
   - Categorical classification across 10 disease classes
   - Softmax activation in output layer
   - Early stopping and validation monitoring

### Recommendation System

- **Approach**: Content-based filtering
- **Method**: TF-IDF vectorization with cosine similarity
- **Output**: Ranked treatment recommendations based on detected disease

## 🚀 Installation & Setup

### Prerequisites

```bash
Python 3.7+
pip
virtualenv (recommended)
```

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/Anannya09/Cropable.git
cd Cropable

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### Required Dependencies

```txt
Flask>=2.0.0
tensorflow>=2.6.0
keras>=2.6.0
numpy>=1.19.0
pandas>=1.3.0
Pillow>=8.0.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
matplotlib>=3.4.0
```

## 💻 Usage

### Running the Application

```bash
# Start Flask server
python apps.py

# Access the application
# Navigate to http://localhost:5000 in your browser
```

### Using Cropable

1. **Sign Up/Sign In**: Create an account or log in
2. **Upload Image**: Select and upload an image of the affected crop leaf
3. **Get Diagnosis**: The system analyzes the image and identifies the disease
4. **View Treatment**: Review recommended treatment measures for the detected disease

## 📁 Project Structure

```
Cropable/
├── apps.py                 # Main Flask application
├── models.py              # CNN model definitions
├── train_model.py         # Model training script
├── leaf.py                # Leaf processing utilities
├── forms.py               # Web form definitions
├── admin.py               # Admin panel functionality
├── asgi.py                # ASGI configuration
├── wsgi.py                # WSGI configuration
├── settings.py            # Application settings
├── tests.py               # Unit tests
├── urls.py                # URL routing
├── views.py               # View controllers
├── model1.json            # Model architecture (JSON)
├── __init__.py           # Package initialization
├── Cropable.pdf          # Project documentation
├── Example.py            # Usage examples
└── README.md             # This file
```

## 📈 Model Performance

### Sequential CNN Model
- **Training Accuracy**: ~88.80%
- **Test Accuracy**: ~88.80%
- **No overfitting observed**

### VGG16 Transfer Learning
- **Training Accuracy**: ~92%+
- **Improved feature extraction** through pre-trained weights

## 🔬 Research Foundation

This project is based on extensive literature review covering:
- Computer vision techniques for plant disease detection
- Deep learning architectures (CNN, VGG, ResNet)
- Image processing methods (histogram equalization, segmentation)
- Recommendation systems for agricultural applications

**Key References**: 10+ peer-reviewed papers on ML/DL approaches to crop disease detection (2015-2020)

## 🛣️ Future Enhancements

- [ ] Expand to multiple crop types (rice, wheat, cotton, sugarcane)
- [ ] Mobile application development (iOS/Android)
- [ ] Multi-language support for regional accessibility
- [ ] Integration with IoT sensors for real-time monitoring
- [ ] Weather and soil condition integration
- [ ] Fertilizer optimization recommendations
- [ ] Community features for farmer knowledge sharing

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- UCI Machine Learning Repository for datasets
- Kaggle community for the tomato leaf disease dataset
- VIT University for academic support
- Open-source deep learning community (TensorFlow, Keras)

## 📊 Impact

By reducing crop disease-related losses and making expert-level diagnosis accessible to all farmers, Cropable aims to:
- Improve food security
- Increase farmer income through better crop quality
- Reduce unnecessary pesticide usage
- Enable data-driven agricultural decisions

**Built with ❤️ for Indian farmers**
