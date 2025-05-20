#  Fake News Detection using NLP & Machine Learning

This project builds a machine learning pipeline to classify short spoken stories as **True** or **Deceptive** using acoustic features extracted from audio recordings.  
It simulates a real-world fake news or deception detection system using speech-based patterns.

The notebook includes data preprocessing, audio segmentation, feature extraction, model training, tuning, and evaluation.

---

##  Project Overview

- Preprocessed audio recordings and segmented them into 30-second clips  
- Extracted acoustic features: MFCCs, chroma, pitch, spectral contrast, voiced fraction  
- Converted features into numerical format using `StandardScaler`  
- Trained and evaluated four classifiers:
  - Logistic Regression (final selected model)
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
- Tuned hyperparameters using GridSearchCV
- Evaluated models using Accuracy, F1-Score, and Confusion Matrix
- Analyzed feature importance for interpretability

---

##  Final Model

**Logistic Regression** achieved the best balance between performance and generalization, with a validation accuracy of **61%**, outperforming other models after tuning.

---

##  Tools & Libraries

- Python, Pandas, NumPy  
- scikit-learn  
- NLTK  
- TF-IDF Vectorizer  
- Matplotlib, Seaborn

---

## ⚠️ Data Access

> The dataset used in this project consists of privately recorded and labeled audio stories.  
> Due to privacy and usage restrictions, the dataset is **not publicly available** and is excluded from this repository.  
> However, the full code and pipeline are available for review and adaptation using similar data.

---

##  How to Use

This notebook is for reference and educational use.  
To run it on your own data:
1. Prepare a labeled dataset of audio files
2. Extract relevant features (e.g., MFCCs, chroma)
3. Follow the pipeline to scale features, train models, and evaluate

---

## 🧑‍💻 Author

**Sarbesh Bhatta**    


