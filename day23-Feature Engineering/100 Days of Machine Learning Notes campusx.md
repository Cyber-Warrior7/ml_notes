## 🧠 Bhramastra Notes — Day 1 to Day 10

---

### ✅ **Day 1: What is Machine Learning?**

**📌 Definition**:
Machine Learning is a subset of AI that enables systems to learn from data without being explicitly programmed.

**🧠 Core Idea**:

* Program + Data → Model → Predictions
* ML ≠ Traditional programming

**⚙️ Components**:

* **Data** (features + labels)
* **Model** (algorithm that learns from data)
* **Learning** (process of optimizing model)

---

### ✅ **Day 2: AI vs ML vs DL**

| Category | AI               | ML                     | DL           |
| -------- | ---------------- | ---------------------- | ------------ |
| Scope    | Broad            | Subset of AI           | Subset of ML |
| Input    | Human-like logic | Data                   | Large data   |
| Examples | Expert systems   | Linear Regression, SVM | CNNs, RNNs   |

**🧠 Key Point**:
DL uses neural networks (multi-layered), while ML uses both statistical and probabilistic models.

---

### ✅ **Day 3: Types of Machine Learning**

1. **Supervised Learning**

   * Labeled data (input → output)
   * Examples: Linear Regression, Classification

2. **Unsupervised Learning**

   * No labels (find structure)
   * Examples: Clustering, Dimensionality Reduction

3. **Reinforcement Learning**

   * Agent learns by interacting with environment
   * Examples: AlphaGo, Self-driving cars

---

### ✅ **Day 4: Batch vs Online Learning**

**📦 Batch (Offline) Learning**:

* Trains on entire dataset
* Slower updates, good for stable systems

**🌐 Online Learning**:

* Learns incrementally (one sample at a time)
* Best for real-time data or streaming

| Type   | Memory | Speed | Use Case          |
| ------ | ------ | ----- | ----------------- |
| Batch  | High   | Slow  | Static systems    |
| Online | Low    | Fast  | Real-time systems |

---

### ✅ **Day 5: Online Learning Continued**

* **Concept Drift**: When data distribution changes over time (e.g., spam filters)
* **Out-of-core learning**: Learning from data that doesn’t fit in memory
* **Learning Rate**: Controls step size in learning

**⚠️ Risk**: Online models can be unstable if learning rate is too high.

---

### ✅ **Day 6: Instance-Based vs Model-Based Learning**

**📌 Instance-Based**:

* Memorizes training examples (lazy learning)
* Example: K-Nearest Neighbors

**📌 Model-Based**:

* Learns a mathematical model (generalizes)
* Example: Linear Regression, Decision Trees

| Aspect          | Instance-Based | Model-Based |
| --------------- | -------------- | ----------- |
| Learning Time   | Low            | High        |
| Prediction Time | High           | Fast        |
| Accuracy        | Local          | Global      |

---

### ✅ **Day 7: Challenges in ML**

1. **Insufficient Data**
2. **Poor Quality Data**
3. **Non-representative Data**
4. **Irrelevant Features**
5. **Overfitting / Underfitting**
6. **Data Leakage**
7. **Software Integration**
8. **Deployment Complexity**

---

### ✅ **Day 8: Applications of ML**

| Domain         | Example                        |
| -------------- | ------------------------------ |
| Retail         | Amazon recommendations         |
| Banking        | Fraud detection                |
| Healthcare     | Disease prediction             |
| Transportation | Demand prediction (OLA/Uber)   |
| Manufacturing  | Predictive maintenance (Tesla) |

---

### ✅ **Day 9: ML Development Life Cycle (MLDLC)**

**Stages**:

1. Frame the Problem
2. Gather Data
3. Data Preprocessing
4. Exploratory Data Analysis (EDA)
5. Feature Engineering & Selection
6. Model Training & Evaluation
7. Deployment
8. Monitoring & Testing

**🔁 It's an iterative process.**

---

### ✅ **Day 10: Data Engineer vs Analyst vs Scientist vs ML Engineer**

| Role               | Focus                     | Key Skills                                 |
| ------------------ | ------------------------- | ------------------------------------------ |
| **Data Engineer**  | Data pipelines, databases | SQL, Big Data tools (Hadoop, Spark), AWS   |
| **Data Analyst**   | Reports, dashboards       | Excel, SQL, Python (pandas), visualization |
| **Data Scientist** | Predictive modeling       | Python, ML, statistics, storytelling       |
| **ML Engineer**    | Production ML systems     | ML + Software Engineering + Deployment     |

**🧠 Key Insight**:

* *ML Engineers* deploy and maintain models.
* *Data Scientists* experiment and analyze.
* *Engineers* ensure models run reliably.

---

### 📌 Bonus Tip:

> Don’t memorize — *understand the flow.* These early days are about setting the foundation.

---

## 🧠 Bhramastra Notes — Day 11 to Day 15
---

### ✅ **Day 11: What Are Tensors?**

**📌 Definition**:
A **tensor** is a multi-dimensional array — the core data structure in ML & DL.

| Tensor    | Shape                      | Example               |
| --------- | -------------------------- | --------------------- |
| Scalar    | 0D                         | `x = 5`               |
| Vector    | 1D                         | `[1, 2, 3]`           |
| Matrix    | 2D                         | `[[1, 2], [3, 4]]`    |
| 3D Tensor | (2, 2, 2)                  | Data over time/images |
| 4D        | Images in DL models        | CNNs                  |
| 5D        | videos in DL models        | CNNs                  |

**🔎 Concepts**:

* **Rank** = Number of dimensions (axes)
* **Shape** = Tuple of dimensions (e.g., `(3, 2)`)

---

### ✅ **Day 12: Installing Tools — Anaconda, Jupyter, Colab**

| Tool                 | Use                                        |
| -------------------- | ------------------------------------------ |
| **Anaconda**         | Local ML environment with Python + Jupyter |
| **Jupyter Notebook** | Browser-based code+text notebook           |
| **Google Colab**     | Online Jupyter + free GPU                  |

**🛠️ Setup Tips**:

* Use `conda create -n ml_env python=3.9` to create a new env
* Use **Google Colab** if no GPU or RAM locally

---

### ✅ **Day 13: End-to-End ML Toy Project(Placement Prediction)**
---

### 🎯 **Objective**:

To understand the complete machine learning pipeline:

* From raw data to deployment-ready model
* Using classification (Logistic Regression) on a placement dataset

---

**📦 Step 0: Imports and Data Loading**:

```python
import numpy as np
import pandas as pd
df = pd.read_csv('/content/placement.csv')
```

* Load the dataset using pandas.
* Use `.head()`, `.info()`, `.shape` to explore the dataset.

```python
df = df.iloc[:, 1:]  # Remove first unnamed index column
```

---

**📊 Step 1: Data Visualization and EDA**:

```python
import matplotlib.pyplot as plt
plt.scatter(df['cgpa'], df['iq'], c=df['placement'])
```

* Visualize the relationship between CGPA and IQ.
* Color indicates placement status (`0` or `1`).
* Helps see if a linear boundary might work (good for logistic regression).

---

**🔍 Step 2: Feature and Label Extraction**:

```python
X = df.iloc[:, 0:2]  # Features: CGPA, IQ
y = df.iloc[:, -1]   # Label: Placement (0 or 1)
```

* `X` → Input features
* `y` → Target output

---

**📤 Step 3: Train-Test Split**:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
```

* Split the data into training and testing (10% test data).
* `random_state` not set, so split varies each run.

---

**📏 Step 4: Feature Scaling**:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

* Standardizes features to mean = 0 and std = 1.
* Improves training speed and performance for gradient-based models.

---

**🤖 Step 5: Model Training (Logistic Regression)**:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
```

* Logistic Regression is used as a binary classifier.
* Model learns a decision boundary.

---

**🧪 Step 6: Model Evaluation**:

```python
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

* Compares predicted labels with actual labels.
* Returns accuracy as a performance metric.

---

**📈 Step 7: Decision Boundary Visualization**:

```python
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train, y_train.values, clf=clf, legend=2)
```

* Visualize how the model separates classes.
* Shows decision regions based on the learned boundary.

---

**💾 Step 8: Save the Model**:

```python
import pickle
pickle.dump(clf, open('model.pkl', 'wb'))
```

* Saves the trained model in binary format.
* Can be loaded later for prediction or deployment.

---

**📌 Workflow Overview**:

```
1. Data Loading
2. Preprocessing (EDA + Cleaning)
3. Feature Extraction
4. Train-Test Split
5. Feature Scaling
6. Model Training
7. Evaluation
8. Deployment (Saving Model)
```

> This toy project shows how **each ML step connects** — use this structure in your future projects too!

---

### ✅ **Day 14: Framing a Machine Learning Problem**

**Step-by-Step Breakdown**:

1. Business Problem → ML Problem
2. Choose **type of ML**: Regression, Classification, Clustering
3. Check existing solutions (if any)
4. Define **metrics** to measure success (e.g., accuracy, RMSE)
5. Choose **Online vs Batch Learning**
6. Assumption checks (feature distributions, nulls)

> 💡 *Framing the problem is 80% of the project.* Don’t rush this part.

---

### ✅ **Day 15: Working with CSV Files**

**📌 Note** From Day 15 also refer notes from `D:\Desktop\python practice\ML journey\100-days-of-machine-learning`
**Python Example**:

```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Show top 5 rows
print(df.head())

# Save to file
df.to_csv('output.csv', index=False)
```

**🛠️ Useful Functions**:

* `df.info()` – Structure of dataset
* `df.describe()` – Statistical summary
* `df.columns`, `df.shape`, `df.dtypes`

**💡 Tip**: Always inspect the data *before* training your model!

---

### 🔚 Summary for Days 10–15

| Day | Key Theme                   |
| --- | --------------------------- |
| 10  | Job roles in DS/ML          |
| 11  | Tensors (core DL concept)   |
| 12  | ML tools setup              |
| 13  | End-to-end toy project      |
| 14  | Framing the ML problem      |
| 15  | CSV data handling in Pandas |

---

## 🧠 Bhramastra Notes — Day 16 to Day 20

---

### ✅ **Day 16: Working with JSON & SQL Data**

**📦 JSON (JavaScript Object Notation)**

* Format for semi-structured data (API responses, configs, logs)

```python
import pandas as pd
import json

# Load JSON from file
df = pd.read_json('data.json')

# Nested JSON example
with open('nested.json') as f:
    data = json.load(f)
df = pd.json_normalize(data)
```

---

**🗄️ SQL Data Access with Python (using SQLite/PostgreSQL/MySQL)**

```python
import sqlite3
conn = sqlite3.connect('mydata.db')

# Run SQL query
df = pd.read_sql_query("SELECT * FROM customers", conn)
```

**🔑 Key Skills**:

* Understand table structure (primary key, foreign key)
* Use `JOIN`, `GROUP BY`, `WHERE` for querying
* SQL + Pandas = 🔥 combo in real-world projects

---

### ✅ **Day 17: Fetching Data from APIs**

**📌 APIs = Application Programming Interfaces**

* Used to get data from services like Twitter, YouTube, etc.

**🛠️ Example: Fetching JSON via API**

```python
import requests

response = requests.get("https://api.example.com/data")
data = response.json()
df = pd.json_normalize(data)
```

**Key Terms**:

* `GET`, `POST` methods
* `Headers`, `Parameters`, `Token` (for authentication)

**🔒 Tip**: Always handle failures with `try-except` and check `response.status_code`.

---

### ✅ **Day 18: Web Scraping**

**📌 Used when no API is available**

**Tools**: `requests`, `BeautifulSoup`, `Selenium`

```python
from bs4 import BeautifulSoup
import requests

url = 'https://example.com'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')

# Extract text
title = soup.find('h1').text
```

**⚠️ Follow Website's `robots.txt` to avoid scraping forbidden content**

---

### ✅ **Day 19: Understanding Your Data (Descriptive Statistics)**

| Method              | Purpose                         |
| ------------------- | ------------------------------- |
| `df.head()`         | View top rows                   |
| `df.describe()`     | Get mean, std, min, max         |
| `df.info()`         | Structure of dataset            |
| `df.value_counts()` | Frequency of categorical values |
| `df.isnull().sum()` | Count missing values            |

**🔍 Stats to Look At**:

* **Mean vs Median** (detect skew)
* **Standard Deviation** (spread)
* **Kurtosis / Skewness** (distribution shape)

---

### ✅ **Day 20: Univariate EDA (Exploratory Data Analysis)**

**📊 Focus on 1 variable at a time**

| Data Type   | Visualization      | Function                             |
| ----------- | ------------------ | ------------------------------------ |
| Numerical   | Histogram, Boxplot | `sns.histplot`, `plt.boxplot`        |
| Categorical | Bar Chart          | `sns.countplot`, `df.value_counts()` |

**Python Examples**:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['age'])
sns.boxplot(x=df['salary'])
```

**🧠 Goal**: Understand data range, distribution, outliers, skewness.

---

### 🧠 Summary (Days 16–20)

| Day | Focus                                     |
| --- | ----------------------------------------- |
| 16  | JSON & SQL file handling                  |
| 17  | Fetching API data                         |
| 18  | Web Scraping                              |
| 19  | Descriptive stats                         |
| 20  | Univariate EDA (Visualization + Analysis) |

---

## 🧠 Bhramastra Notes — Day 21 to Day 25 (Detailed)

---

### ✅ **Day 21: Bivariate & Multivariate EDA**

#### 🔍 Bivariate Analysis:

* Study relationship between **2 variables**
* Useful for identifying **correlations or group-wise trends**

| Variable Type | Visualization                     |
| ------------- | --------------------------------- |
| Num + Num     | Scatter plot, Correlation heatmap |
| Cat + Num     | Boxplot, Violin plot              |
| Cat + Cat     | Grouped bar chart, Stacked bar    |

**📌 Code Examples:**

```python
# Numerical vs Numerical
sns.scatterplot(x='age', y='salary', data=df)

# Categorical vs Numerical
sns.boxplot(x='gender', y='salary', data=df)

# Correlation
df.corr()  # Numerical only
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

#### 🔍 Multivariate Analysis:

* Explore 3 or more variables together
* Use **pairplots**, **hue**, and **facet plots**

```python
# All numeric pairs
sns.pairplot(df)

# Add color grouping
sns.pairplot(df, hue='target')

# Facet plot
sns.catplot(x='gender', y='salary', col='education', data=df, kind='bar')
```

> 🔑 Tip: Always begin with bivariate → then explore multivariate if patterns are unclear.

---

### ✅ **Day 22: Pandas Profiling (EDA Automation Tool)**

#### ⚙️ What it is:

* Automatically generates a complete EDA report from a Pandas DataFrame.

#### 🔧 How to Use:

```python
import pandas_profiling
profile = df.profile_report(title="EDA Report")
profile.to_file("output.html")
```

#### 📊 It includes:

* Variable summary
* Missing values
* Correlation matrix
* Skewness & distributions
* Duplicate detection
* Interaction graphs

#### ⚠️ When to use:

* **Early** in the pipeline for fast insights.
* Not suitable for huge datasets without sampling.

> ⚠️ Use only when you're starting to explore — for production, manual EDA is better and more controllable.

---

### ✅ **Day 23: What is Feature Engineering?**

#### 🎯 Purpose:

Improving model performance by **transforming raw data** into useful features.

#### 🔨 Categories:

1. **Missing Value Imputation**
2. **Handling Categorical Features**
3. **Outlier Detection**
4. **Feature Scaling**
5. **Feature Construction** (e.g., Age → AgeGroup)
6. **Feature Extraction** (e.g., PCA)

#### 💬 Quote:

> “Better features beat better models.”

#### 🔍 Example:

```python
# Construct new feature from date
df['age'] = 2025 - df['birth_year']
```

#### 🧠 Real-world Tip:

Domain knowledge + creativity = best feature engineering!

---

### ✅ **Day 24: Feature Scaling — Standardization**

#### ⚖️ Why Needed:

* ML algorithms like **KNN**, **SVM**, **Logistic Regression**, and **Gradient Descent** are sensitive to feature scale.

#### 📐 Standardization:

Formula:

$$
z = \frac{x - \mu}{\sigma}
$$

* Mean = 0
* Std Dev = 1

#### 📦 Sklearn Code:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 🧪 When to Use:

* Features are **normally distributed**
* Algorithms using **distance** or **gradient descent**

---

### ✅ **Day 25: Feature Scaling — Normalization**

#### 📐 Normalization:

Transforms features into a **bounded range** — typically \[0, 1]

#### 🔢 Min-Max Scaling:

$$
x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

#### 🔸 Other Types:

| Type                   | When to Use                                          |
| ---------------------- | ---------------------------------------------------- |
| **MaxAbsScaler**       | Sparse data (keeps zero entries)                     |
| **RobustScaler**       | Data with outliers                                   |
| **Mean Normalization** | Not commonly used in sklearn                         |
| **MinMaxScaler**       | When bounded output is expected (e.g., pixel values) |

#### 💡 When to Use Normalization:

* Algorithms like **KNN**, **Neural Networks**
* Models where **magnitude matters**

---

### 🔚 Summary (Days 21–25)

| Day | Focus                                            |
| --- | ------------------------------------------------ |
| 21  | Bivariate & Multivariate EDA                     |
| 22  | Pandas Profiling (Auto EDA)                      |
| 23  | Feature Engineering Overview                     |
| 24  | Feature Scaling – Standardization                |
| 25  | Feature Scaling – Normalization & Robust Methods |

---

## 🧠 Bhramastra Notes — Day 26 to Day 28 (Detailed)

---

### ✅ **Day 26: Encoding Categorical Variables – Ordinal & Label Encoding**

#### 🎯 Why Encoding Is Needed:

ML models only work with **numerical inputs**, so we must convert categorical variables (like 'Male', 'Female') into numbers.

---

### 🔷 **1. Ordinal Encoding**

* Assigns a **rank** or **order** to each category
* Use when categories have a **logical sequence**

Example:

```python
education = ['High School', 'Bachelor', 'Master', 'PhD']
Encoded as → [1, 2, 3, 4]
```

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
X[['education_encoded']] = encoder.fit_transform(X[['education']])
```

> ⚠️ Don't use this on nominal (unordered) data — the model may assume mathematical relationship that doesn’t exist.

---

### 🔷 **2. Label Encoding**

* Converts categories into **integer values**
* Used for **target/label** column in classification (not features)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
```

Example:

```
['dog', 'cat', 'dog', 'fish'] → [1, 0, 1, 2]
```

> ⚠️ For input features, LabelEncoding is dangerous — use OneHotEncoding instead for unordered variables.

---

### ✅ **Day 27: One Hot Encoding (OHE)**

#### 🔳 What it does:

Creates **binary columns** for each category.

Example:

```
Color → ['Red', 'Blue', 'Green']
OHE → [1,0,0], [0,1,0], [0,0,1]
```

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, drop='first')
X_encoded = ohe.fit_transform(X[['color']])
```

---

#### 🧠 Dummy Variable Trap:

* Happens when OHE creates **multicollinearity**
* Fix: use `drop='first'` to remove 1 column

---

#### ✅ When to use OHE vs Ordinal:

| Data Type                          | Encoding         |
| ---------------------------------- | ---------------- |
| Ordered (e.g. Low < Medium < High) | Ordinal          |
| Unordered (e.g. City, Gender)      | One Hot Encoding |

---

### ✅ **Day 28: ColumnTransformer in Sklearn**

#### 🎯 What is it?

Efficient way to apply **different preprocessing steps to different columns**.

---

### 🔧 Example Use Case:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'salary']),
        ('cat', OneHotEncoder(drop='first'), ['gender', 'city'])
    ]
)

X_processed = ct.fit_transform(X)
```

* `'num'` applies `StandardScaler` to numerical features
* `'cat'` applies `OneHotEncoder` to categorical features

---

### 🔗 Why it’s useful:

* **Keeps preprocessing organized**
* Works smoothly inside **pipelines**
* Handles **mixed-type** datasets easily

---

### ✅ Bonus: Common Pitfalls to Avoid

| Mistake                        | Fix                                                 |
| ------------------------------ | --------------------------------------------------- |
| Using LabelEncoder on features | Use OHE or Ordinal depending on context             |
| Not scaling after encoding     | Apply `StandardScaler` on numeric columns only      |
| Manual encoding on large data  | Use `ColumnTransformer` to automate and chain steps |

---

### 🔚 Summary (Days 26–28)

| Day | Focus                                                                     |
| --- | ------------------------------------------------------------------------- |
| 26  | Ordinal & Label Encoding                                                  |
| 27  | One Hot Encoding & Dummy Variable Trap                                    |
| 28  | ColumnTransformer — applying different preprocessing pipelines to columns |

---

### 🎁 What’s Next?

You're now ready to move into **Day 29: Pipelines**, which builds directly on ColumnTransformer.

---
