Here's a **perfectly combined and structured Scikit-learn Notes** from all your uploaded HTML cheat sheets (`scikit-learn.html`, `sklearn1.html`, `sklearn2.html`) and the code examples from the notebook. The notes are clean, professional, beginner-friendly, and include code with comments for understanding.

---

# 🧠 Scikit-learn Cheatsheet – Complete Notes with Code

> Author: **Vedant Kawade**
> Language: **Python 3.x**
> Library: **Scikit-learn (sklearn)**
> Use-case: **Preprocessing, Training, Tuning, Saving & Evaluating ML models**

---

## 📦 1. Installation

```bash
# Install scikit-learn
pip install scikit-learn

# For updating to the latest version
pip install --upgrade scikit-learn
```

---

## 📚 2. Loading Datasets

```python
from sklearn.datasets import load_iris

# Load built-in dataset
iris = load_iris()
X, y = iris.data, iris.target
```

---

## ⚙️ 3. Data Preprocessing

### ➤ Impute Missing Values

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent' also possible
X = imputer.fit_transform(X)
```

### ➤ Feature Scaling (Standardization)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### ➤ Encoding Categorical Variables

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(np.array([['Male'], ['Female'], ['Male']]))
```

---

## ✂️ 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 🏗️ 5. Model Training

### ➤ Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### ➤ Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
```

### ➤ Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
```

---

## 🧪 6. Model Evaluation

```python
from sklearn.metrics import accuracy_score, mean_squared_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
```

For classification:

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

---

## 🧠 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid)
grid.fit(X, y)

print("Best Params:", grid.best_params_)
```

---

## 💾 8. Saving & Loading Models

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')
```

---

## 🧩 9. Commonly Used Modules & Their Purpose

| Module                    | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `sklearn.preprocessing`   | Feature scaling, encoding, imputing           |
| `sklearn.linear_model`    | Regression, classification models             |
| `sklearn.ensemble`        | Random Forest, Boosting methods               |
| `sklearn.tree`            | Decision Trees                                |
| `sklearn.cluster`         | KMeans and other clustering algorithms        |
| `sklearn.model_selection` | Splitting, cross-validation, tuning           |
| `sklearn.metrics`         | Evaluation: accuracy, F1, MSE, R² etc.        |
| `sklearn.datasets`        | Access to toy datasets like iris, digits etc. |

---

## 🧪 10. Popular Algorithms in Sklearn

| Algorithm           | Class Name                                |
| ------------------- | ----------------------------------------- |
| Linear Regression   | `sklearn.linear_model.LinearRegression`   |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` |
| Random Forest       | `sklearn.ensemble.RandomForestClassifier` |
| Decision Tree       | `sklearn.tree.DecisionTreeClassifier`     |
| K-Means Clustering  | `sklearn.cluster.KMeans`                  |

---

## ✅ End-to-End Mini Example

```python
# Load dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

Let me know if you want this in **Markdown**, **Notion format**, or as a downloadable `.ipynb` or `.pdf`.
