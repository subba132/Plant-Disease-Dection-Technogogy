# 🌿 Plant Disease Detection System

This project presents a **model-based AI system** that utilizes **Logistic Regression** to classify plant leaf images into three categories: **Healthy**, **Powdery**, and **Rust**. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) and contains 1,530 images divided into training, testing, and validation sets.

---

## 📌 Project Objective

To develop an AI model capable of accurately detecting and classifying plant diseases from leaf images, aiding in early diagnosis and treatment.

---

## 📂 Dataset

- **Source:** [Plant Disease Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)
- **Total Images:** 1,530
- **Classes:**
  - Healthy
  - Powdery
  - Rust
- **Structure:** The dataset is organized into `train`, `test`, and `validation` directories, each containing subfolders for the three classes.

---

## 🛠️ Technologies & Libraries

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 🧹 Data Preprocessing

- **Image Resizing:** All images resized to a uniform size (e.g., 128x128 pixels).
- **Grayscale Conversion:** Converted RGB images to grayscale to simplify processing.
- **Flattening:** Transformed 2D image arrays into 1D feature vectors.
- **Normalization:** Scaled pixel values to a range of 0 to 1.
- **Label Encoding:** Converted class labels to numerical format.

---

## 🧠 Model Development

- **Algorithm:** Multinomial Logistic Regression
- **Training:** Model trained on the preprocessed training dataset.
- **Hyperparameter Tuning:** Utilized GridSearchCV to find the optimal regularization parameter.
- **Evaluation:** Assessed model performance on the test dataset.

---

## 📈 Evaluation Metrics

- **Accuracy:** Achieved an accuracy of **<insert accuracy>%** on the test set.
- **Confusion Matrix:** Visualized to assess class-wise performance.
- **Classification Report:** Provided precision, recall, and F1-score for each class.

---

## 📁 Project Structure
plant-disease-detection/
│
├── 📁 data/                            # Raw and preprocessed image data
│   ├── train/
│   ├── test/
│   └── validation/
│
├── 📁 notebooks/                       # Jupyter notebooks for EDA and modeling
│   └── plant_disease_detection.ipynb
│
├── 📁 models/                          # Saved model files (e.g., pickle, joblib)
│   └── logistic_regression_model.pkl
│
├── 📁 utils/                           # Helper scripts
│   ├── preprocessing.py               # Image loading, resizing, flattening
│   └── metrics.py                     # Custom metric calculations (optional)
│
├── 📁 results/                         # Plots, confusion matrix, reports
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── 📁 app/                             # Optional: Web app code for deployment
│   ├── app.py                         # Flask or Streamlit app
│   └── templates/                     # HTML files (for Flask)
│
├── requirements.txt                   # Python dependencies
├── logistic_regression.py             # Core model training script (optional)
├── README.md                          # Project documentation
└── .gitignore                         # Ignore datasets, models, etc.


---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/plant-disease-detection.git
   cd plant-disease-detection
   cd plant-disease-detection
   pip install -r requirements.txt
   python predict.py
