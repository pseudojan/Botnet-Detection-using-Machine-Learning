# Botnet-Detection-using-Machine-Learning


This project detects botnet attacks using a supervised machine learning approach with preprocessing, SMOTE for handling class imbalance, and Random Forest for classification.

---

## 📁 Dataset

We use the **CICIDS2017** dataset, specifically the file:

```
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Download it from [Kaggle Dataset]([https://www.unb.ca/cic/datasets/ids-2017.html](https://www.kaggle.com/datasets/ishasingh03/friday-workinghours-afternoon-ddos)).

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```


##  Model Pipeline

### Step 1: Data Cleaning

* Drop unnecessary or irrelevant columns.
* Handle missing/infinite values.
* Encode categorical columns.

### Step 2: Feature Scaling

* Standardize features using `StandardScaler`.

### Step 3: Train-Test Split

* Stratified 70-30 split to preserve label distribution.

### Step 4: Imbalance Handling

* Use **SMOTE** to balance the classes in training data.

### Step 5: Model Training

* Train a `RandomForestClassifier` (with `class_weight='balanced'`).

### Step 6: Evaluation Metrics

* Confusion Matrix
* Classification Report (Precision, Recall, F1)
* Balanced Accuracy Score
* ROC-AUC Curve
* Cross-validation scores

---

## 📊 Results
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     22599
           1       1.00      1.00      1.00     38409

    accuracy                           1.00     61008
   macro avg       1.00      1.00      1.00     61008
weighted avg       1.00      1.00      1.00     61008


Balanced Accuracy: 0.9997917154833502


### ✅ Confusion Matrix

Confusion Matrix:
 [[22599     0]
 [   16 38393]]

### 🔍 ROC Curve

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/60928e90-13c1-4a77-ac7b-3ea89ab4af65" />


### 📌 Feature Importance

Top features identified from the model.
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/d6ee49f6-7123-42b2-8c46-2a1f6a00fc38" />


---


## 🧪 Cross-Validation

The script uses Stratified K-Fold (5-fold) to assess model performance across splits.

---

## 🚀 How to Run

```bash
python botnet_detector.py
```

Make sure to place your dataset in the same directory or update the path accordingly.

---

## 📂 Directory Structure

```
project_root/
├── botnet_detector.py
├── requirements.txt
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── README.md
```


## ✍️ Author

pseudojan



## 📎 References

* CICIDS2017 Dataset: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
* SMOTE: [https://imbalanced-learn.org/](https://imbalanced-learn.org/)
* Scikit-learn Docs: [https://scikit-learn.org/](https://scikit-learn.org/)
