import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.preprocess import preprocess_data

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'sleep_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'sleep_model.pkl')

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)
df = preprocess_data(df)

# ---------- FEATURE SELECTION ----------
FEATURES = [
    'Sleep Duration',
    'Physical Activity Level',
    'Stress Level',
    'Daily Steps'
]

X = df[FEATURES]
y = df['Quality of Sleep']

# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------- MODEL + GRID SEARCH ----------
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ---------- BEST MODEL ----------
model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# ---------- EVALUATION ----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy * 100:.2f}%")

# ---------- SAVE MODEL ----------
joblib.dump(model, MODEL_PATH)
print("Model saved successfully at:", MODEL_PATH)
