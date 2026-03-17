# Customer Churn Prediction System

## Overview

This project predicts customer churn using a **complete end-to-end data pipeline**, combining data preprocessing, feature engineering, machine learning, and customer segmentation to generate actionable business insights.

---

## Pipeline Overview

```
Raw Data → Ingestion → Cleaning → Feature Engineering → Segmentation → Modeling
```

---

## Key Components

### 📥 Data Ingestion (`ingest.py`)
- Loads raw dataset and prepares it for processing  


### 🧹 Data Cleaning (`clean.py`)
- Removes duplicates and handles missing values (median/mode)  


### ⚙️ Feature Engineering (`features.py`)
- Creates key features:
  - Engagement Index  
  - Risk Score  
  - Value Score  
 

### 🧠 Segmentation (`segmentation.py`)
- Uses **Gaussian Mixture Model (GMM)**  
- Segments customers into:
  - Loyal  
  - At Risk  
  - Needs Attention  
  

### 🤖 Modeling (`model_pipeline.py`)
- Trains **Random Forest Classifier**  
- Evaluates using:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC  
- Generates **churn probability scores**  

### 🔍 Validation (`check.py`)
- Performs data quality checks  

---

## Notebooks

- `01_EDA.ipynb` — Data exploration  
- `02_analysis.ipynb` — Churn analysis  
- `03_modeling.ipynb` — Model building  
- `04_insights.ipynb` — Business insights  

---

## Key Insights

- Low engagement → higher churn  
- High-value customers → higher revenue risk  
- Behavioral features strongly influence churn  

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  

---

## Run the Project

```
git clone https://github.com/shashwatkul/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

python ingest.py
python clean.py
python features.py
python segmentation.py
python model_pipeline.py
```

---

## Author

**Shashwat Kulshrestha**  
Aspiring Data Analyst / ML Engineer
