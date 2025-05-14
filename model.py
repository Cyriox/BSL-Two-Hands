import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("dataset/two_hand_dataset.csv")

# Extract features and labels
X = df.iloc[:, :-1].values  # all columns except last
y = df.iloc[:, -1].values   # last column

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save the encoder
joblib.dump(encoder, "label_encoder.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Save model
model.save("twohand_model.h5")
