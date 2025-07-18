# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


# 2. Load Dataset
df = pd.read_csv("/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

df.columns


# 3. Preprocessing
# Drop unnecessary columns

df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces in column names
print(f"Columns: {df.columns}")

# Drop irrelevant or unnecessary columns
drop_cols = [
    'Flow ID', 'Source IP', 'Destination IP', 'Protocol', 'Timestamp',  # Identifiers and timestamp
    'Flow Byts/s', 'Flow Pkts/s', 'Flow Packets/s',  # Flow statistics columns (not needed for analysis)
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',  # Flags columns
    'Fwd Packets/s', 'Bwd Packets/s', 'Fwd Header Length.1',  # Redundant/irrelevant columns
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',  # Traffic-related, not needed
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',  # Traffic-related, not needed
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',  # Subflow-related
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward'  # Miscellaneous
]

# Drop the unnecessary columns
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Check for missing values and fill accordingly
# Fill missing values in numeric columns with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing values in non-numeric columns with the mode (most frequent value)
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
for col in non_numeric_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify that there are no missing values left
print(f"Missing values after filling: {df.isnull().sum().sum()}")

# Encode categorical columns if any
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# Define features and target
X = df.drop(columns=['Label'])  # 'Label' is the target column
y = df['Label']


import numpy as np

# Check for infinite values
print(f"Any NaN values in X: {X.isnull().sum().sum()}")
print(f"Any infinite values in X: {np.isinf(X).sum().sum()}")

# Replace infinite values with NaN
X.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

# Fill NaN values with the column mean
X.fillna(X.mean(), inplace=True)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify that the transformation worked
print(f"First 5 rows of the scaled data:\n{X_scaled[:5]}")


# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)


# Print the class distribution in the training set
print(f"Class distribution in training set:\n{y_train.value_counts()}")


# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Print the class distribution in the training set after SMOTE
print(f"Class distribution in training set after SMOTE:\n{y_train_res.value_counts()}")


model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train_res, y_train_res)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Balanced Accuracy Score
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc}")

# Optional: Feature importance
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance.values[:15], y=feat_importance.index[:15])
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

# Use Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
print(f"Cross-validated accuracy scores: {cross_val_scores}")
print(f"Mean cross-validated accuracy: {cross_val_scores.mean()}")

# Train model with full training data
model.fit(X_train_res, y_train_res)

# Predictions and ROC-AUC
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


