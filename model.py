import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Load Dataset
dataset = pd.read_csv("labeled_eeg_dataset_with_ictal.csv")
X = dataset.drop(columns=['label'])
y = dataset['label']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_categorical = to_categorical(y, num_classes=3)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y)

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# Train SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, np.argmax(y_train, axis=1))
joblib.dump(svm_model, "svm_model.pkl")

# Train GRU Model
X_train_gru = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_gru = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

gru_model = Sequential([
    GRU(128, return_sequences=True, input_shape=(X_train_gru.shape[1], X_train_gru.shape[2])),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
gru_model.fit(X_train_gru, y_train, epochs=10, batch_size=32, validation_data=(X_test_gru, y_test), verbose=0)
gru_model.save("gru_model.keras")