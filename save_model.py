# save_model.py
# Run this ONCE to train your model and save it to files.
# After this, your notebook is no longer needed for the app.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("Step 1: Loading data...")
df = pd.read_csv("Sample - Superstore.csv", encoding="latin-1")

print("Step 2: Feature engineering...")
df['Order Date']    = pd.to_datetime(df['Order Date'])
df['Ship Date']     = pd.to_datetime(df['Ship Date'])
df['Delivery Days'] = (df['Ship Date'] - df['Order Date']).dt.days

print("Step 3: Removing outliers (IQR method, same as your notebook)...")
Q1 = df['Profit'].quantile(0.25)
Q3 = df['Profit'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[
    (df['Profit'] >= Q1 - 1.5 * IQR) &
    (df['Profit'] <= Q3 + 1.5 * IQR)
]
print(f"   Rows before: {len(df)} | After cleaning: {len(df_clean)}")

print("Step 4: Preparing features (same as your notebook cell 68)...")
FEATURES = ['Sales', 'Discount', 'Delivery Days',
            'Sub-Category', 'Region', 'Segment', 'Ship Mode']

X = df_clean[FEATURES].copy()
y = df_clean['Profit'].copy()

# Encode categories to numbers — same as pd.get_dummies in your notebook
X_encoded = pd.get_dummies(X, drop_first=True).astype(int)

# IMPORTANT: Save the column names in exact order
# This is needed so predictions always match what the model expects
feature_columns = list(X_encoded.columns)
print(f"   Total features after encoding: {len(feature_columns)}")

print("Step 5: Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

print("Step 6: Evaluating...")
r2  = r2_score(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"   R2 Score : {r2:.4f}")
print(f"   MAE      : ${mae:.2f}")

print("Step 7: Saving model files...")
# Save the trained model
joblib.dump(model, 'retail_model.pkl')

# Save column names — Flask needs this to align input data
joblib.dump(feature_columns, 'feature_columns.pkl')

# Save dropdown options — Streamlit needs this to build the UI
meta = {
    'sub_categories': sorted(df_clean['Sub-Category'].unique().tolist()),
    'regions':        sorted(df_clean['Region'].unique().tolist()),
    'segments':       sorted(df_clean['Segment'].unique().tolist()),
    'ship_modes':     sorted(df_clean['Ship Mode'].unique().tolist()),
}
joblib.dump(meta, 'model_meta.pkl')

print("\n✅ Done! These 3 files were created:")
print("   retail_model.pkl    — your trained model")
print("   feature_columns.pkl — column order for predictions")
print("   model_meta.pkl      — dropdown options for the UI")
print("\nSub-Categories found:", meta['sub_categories'])
print("Regions found:       ", meta['regions'])
print("Segments found:      ", meta['segments'])
print("Ship Modes found:    ", meta['ship_modes'])