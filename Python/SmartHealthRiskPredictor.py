import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Synthetic Data Generation (you can replace with real dataset)
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'Age': np.random.randint(25, 80, n),
    'Gender': np.random.randint(0, 2, n),
    'BMI': np.random.uniform(18, 40, n),
    'Cholesterol': np.random.randint(150, 300, n),
    'RestingBP': np.random.randint(80, 180, n),
    'FastingSugar': np.random.randint(0, 2, n),
    'PhysicalActivity': np.random.randint(0, 10, n),
    'Smoking': np.random.randint(0, 2, n),
    'AlcoholIntake': np.random.randint(0, 10, n),
    'FamilyHistory': np.random.randint(0, 2, n)
})

# Labels (Simulated based on conditions)
data['Heart_Disease'] = ((data['Cholesterol'] > 240) | (data['BMI'] > 30) | (data['Smoking'] == 1)).astype(int)
data['Diabetes'] = ((data['FastingSugar'] == 1) | (data['BMI'] > 28)).astype(int)
data['Hypertension'] = ((data['RestingBP'] > 130) | (data['Age'] > 55)).astype(int)

# 2️⃣ Split features and targets
X = data.drop(['Heart_Disease', 'Diabetes', 'Hypertension'], axis=1)
y = data[['Heart_Disease', 'Diabetes', 'Hypertension']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Train Multi-output Model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
multi_model = MultiOutputClassifier(rf)
multi_model.fit(X_train, y_train)

# 4️⃣ Predictions
y_pred = multi_model.predict(X_test)

# 5️⃣ Evaluate
print("🔍 Model Accuracy for each disease:")
for i, col in enumerate(y.columns):
    print(f"{col}: {accuracy_score(y_test[col], y_pred[:, i]):.2f}")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=y.columns))

# 6️⃣ Predict for a new user
new_user = np.array([[45, 1, 27.5, 220, 140, 0, 3, 1, 2, 1]])  # Example input
pred = multi_model.predict(new_user)
print("\n🧑‍⚕️ Predicted Health Risks (Heart, Diabetes, Hypertension):", pred[0])
