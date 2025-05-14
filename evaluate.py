import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("dataset/two_hand_dataset.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels
encoder = joblib.load("label_encoder.pkl")
y_encoded = encoder.transform(y)

# Split again (same ratio as before)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Load model
model = tf.keras.models.load_model("twohand_model.h5")

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluation
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_classes, target_names=encoder.classes_.astype(str)))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
