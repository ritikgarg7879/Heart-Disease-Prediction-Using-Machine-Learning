# 🫀 Heart Disease Prediction using Machine Learning

Heart disease continues to be one of the leading causes of death worldwide. With timely intervention and proper diagnosis, many of these cases can be prevented. This project aims to build a robust machine learning-based predictive model that can help identify individuals at risk of developing heart disease, thus aiding in early diagnosis and preventive care.

## 📌 Objective

The goal of this project is to develop and evaluate multiple machine learning models to predict the presence of heart disease based on various medical attributes of a patient. This is a **binary classification problem**, where the output variable indicates the presence (1) or absence (0) of heart disease.

## 📊 Dataset

- **Source**: UCI Heart Disease Dataset
- **Features include**:
  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Serum Cholesterol
  - Fasting Blood Sugar
  - Resting ECG Results
  - Maximum Heart Rate Achieved
  - Exercise Induced Angina
  - ST Depression
  - Slope of Peak Exercise ST Segment
  - Number of Major Vessels Colored by Fluoroscopy
  - Thalassemia
- **Target Variable**: `target` (1 = heart disease, 0 = no heart disease)

## 🛠️ Technologies Used

- **Python** (Jupyter Notebook)
- **Libraries**:
  - `Pandas`, `NumPy` – Data manipulation
  - `Matplotlib`, `Seaborn` – Data visualization
  - `Scikit-learn` – ML model implementation
  - `Keras` (TensorFlow backend) – ANN model
  - `XGBoost` – Gradient Boosting Classifier

## 📈 ML Models Implemented

I trained and evaluated several supervised machine learning algorithms:

- ✅ Logistic Regression
- ✅ Naive Bayes
- ✅ Support Vector Machine (Linear)
- ✅ K-Nearest Neighbours (KNN)
- ✅ Decision Tree Classifier
- ✅ Random Forest Classifier
- ✅ XGBoost Classifier
- ✅ Artificial Neural Network (ANN) using Keras with 1 hidden layer

Each model was evaluated based on metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

## ⚙️ Workflow

1. **Data Preprocessing**:
   - Checked for missing values
   - Standardized features using `StandardScaler`
   - Label encoding for categorical values

2. **Exploratory Data Analysis (EDA)**:
   - Visualized correlations using heatmaps
   - Compared distributions of features across target classes

3. **Model Training**:
   - Trained models using 80/20 train-test split
   - Applied cross-validation to reduce variance
   - Performed hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`

4. **Neural Network Architecture**:
   - Input Layer
   - Hidden Layer (ReLU activation)
   - Output Layer (Sigmoid activation)
   - Optimized using Adam optimizer and binary cross-entropy loss

## 🏆 Best Model

- **Random Forest Classifier** performed the best with an **accuracy of 95%**, followed by ANN and XGBoost with slightly lower accuracy but strong overall metrics.
- The model can generalize well and performs efficiently on unseen data.

