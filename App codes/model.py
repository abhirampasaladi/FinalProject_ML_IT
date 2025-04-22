import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load and preprocess dataset
file_path = "dataset_cars.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)
df = df[(df['price'] >= 1000) & (df['price'] <= 100000)]
df = df[df['odometer'] < 3000000]
df['car_age'] = 2025 - df['year']
df.drop(columns=['year'], inplace=True)

categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'transmission', 'drive', 'type', 'paint_color']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler, label_encoders), f)