🧠 Gender Classification from Facial Images
A machine learning project that detects faces in images and classifies gender using a PCA-reduced SVM pipeline. Built with OpenCV, scikit-learn, and Flask for local deployment.

📸 Project Overview
This project performs gender classification from facial images using a combination of:
- HAAR Cascade Classifier for face detection
- Principal Component Analysis (PCA) for dimensionality reduction
- Support Vector Machine (SVM) for classification
The final model is deployed as a Flask web application that allows users to upload images and view predictions with annotated results.



🧪 Model Pipeline
- Face Detection
    - Uses OpenCV’s haarcascade_frontalface_default.xml to detect faces in grayscale images.
- Preprocessing
    - Resizes detected faces to 100×100 pixels
    - Normalizes pixel values
    - Flattens into 10,000-dimensional vectors
- Dimensionality Reduction
    - Applies PCA to extract top 250 principal components
    - Subtracts the mean face before transformation
- Classification
    - Trained an SVM classifier on PCA-transformed data
    - Evaluated using metrics like:
        - AUC-ROC
        - Cohen’s Kappa Score





The link to the dataset used in the model creation:
    https://drive.google.com/file/d/1bNVQoQZYnfK1rb2AeRquGRKEDVVvJIRR/view?usp=sharing 




