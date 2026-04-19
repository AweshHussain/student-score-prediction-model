import pandas as pd

# -------------------------------
# 1. LOAD DATA
# -------------------------------
data = pd.read_csv("student-scores.csv")

# -------------------------------
# 2. CLEAN DATA
# -------------------------------
# Remove useless columns
data = data.drop(["id", "first_name", "last_name", "email"], axis=1, errors='ignore')

print("Step 1: Data after cleaning")
print(data.head())

# -------------------------------
# 3. ENCODE CATEGORICAL DATA
# -------------------------------
data = pd.get_dummies(data, drop_first=True)

print("\nStep 2: Data after encoding")
print(data.head())

# -------------------------------
# 4. FEATURE SELECTION
# -------------------------------
# Select meaningful features
feature_cols = [
    "physics_score",
    "chemistry_score",
    "biology_score",
    "english_score",
    "history_score",
    "geography_score",
    "weekly_self_study_hours",
    "absence_days",
    "part_time_job",
    "extracurricular_activities"
]

# Keep only existing columns (safe check)
feature_cols = [col for col in feature_cols if col in data.columns]

# Base features
X = data[feature_cols]

# Add encoded categorical features
career_cols = [col for col in data.columns if col.startswith("career_aspiration_")]
gender_cols = [col for col in data.columns if col.startswith("gender_")]

X = pd.concat([X, data[career_cols], data[gender_cols]], axis=1)

# Target variable
y = data["math_score"]

print("\nStep 3: Final Features")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. MODEL TRAINING
# -------------------------------
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully")

# -------------------------------
# 7. PREDICTION
# -------------------------------
pred = model.predict(X_test)

print("\nPredictions:", pred[:5])

# -------------------------------
# 8. EVALUATION
# -------------------------------
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, pred)
print("\nMean Absolute Error:", error)

# -------------------------------
# 9. FEATURE IMPORTANCE
# -------------------------------
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))