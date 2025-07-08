import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Load Data
df = pd.read_csv("Disease_Data.csv")

# Convert categorical values to numerical (Yes = 1, No = 0)
binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
df.replace({'Yes': 1, 'No': 0}, inplace=True)

# Label Encoding for categorical features
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Blood Pressure'] = encoder.fit_transform(df['Blood Pressure'])
df['Cholesterol Level'] = encoder.fit_transform(df['Cholesterol Level'])
df['Outcome Variable'] = encoder.fit_transform(df['Outcome Variable'])

# Encode Disease labels and store mapping
disease_encoder = LabelEncoder()
df['Disease'] = disease_encoder.fit_transform(df['Disease'])
disease_mapping = dict(zip(df['Disease'], disease_encoder.classes_))

# Normalize Age
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# Define Features (X) and Target (y)
X = df.drop(columns=['Disease'])
y = df['Disease']

# Remove Rare Diseases (Less than 2 samples)
disease_counts = Counter(y)
rare_diseases = [key for key, count in disease_counts.items() if count < 2]
df_filtered = df[~df['Disease'].isin(rare_diseases)]
X_filtered = df_filtered.drop(columns=['Disease'])
y_filtered = df_filtered['Disease']

# Handle Imbalanced Data using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_filtered, y_filtered)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Example: Backtrack a predicted disease
predicted_disease = disease_mapping.get(y_pred[0], "Unknown Disease")
print("Predicted Disease for First Test Case:", predicted_disease)
