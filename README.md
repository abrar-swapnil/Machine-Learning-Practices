# Machine Learning Practices 🤖

Welcome to my Machine Learning repository! This repository serves as a practical workspace where I document my journey into data science, data preprocessing, model selection, and hyperparameter tuning using Python and Scikit-Learn.

---

## 🧭 Project Overview: Diabetes Progression Predictor
The primary objective of this project is to build an end-to-end Machine Learning Regression pipeline to predict a patient's diabetes progression score based on real-world physiological features.

### 📊 Dataset Details
The project utilizes the real-world **Diabetes Dataset** from Scikit-Learn, featuring 442 patient records with attributes such as:
* **Features ($X$):** Age, Sex, Body Mass Index (BMI), Blood Pressure (BP), and various blood serum measurements.
* **Target ($y$):** A continuous numeric score representing diabetes progression.

---

## 🛠️ Data Preprocessing & Engineering
Real-world data is rarely pristine, requiring essential pipeline structural changes:
1. **Feature Scaling:** The numeric inputs are pre-standardized (scaled between approximately $-0.1$ and $+0.1$) to ensure all attributes hold equal weight during mathematical optimization.
2. **Categorical Encoding:** Handled categorical non-numeric variations (e.g., a structural `City` column) by implementing **One-Hot Encoding** (`pd.get_dummies()`). 
3. **Optimized Feature Dropping:** Applied `drop_first=True` to eliminate the dummy variable trap and avoid redundant multicollinearity loops in the data pipeline.

---

## 📈 Model Performance & Evaluation
I experimented with multiple regression models, tracking their accuracy via the **$R^2$ Score (Coefficient of Determination)**:

| Model | $R^2$ Score | Status / Insights |
| :--- | :---: | :--- |
| **Linear Regression (Baseline)** | **0.4546 (45.46%)** | Best baseline; captures major linear tendencies despite real-world noise. |
| **Decision Tree Regressor (Default)** | **0.0100 (1.00%)** | ❌ Failed due to severe **Overfitting** (Memorization Trap). |
| **Decision Tree Regressor (Tuned)** | **0.3294 (32.94%)** |  Optimized using Hyperparameter Tuning. |

---

## 🧠 Overfitting & Hyperparameter Tuning Solution

### The Overfitting Trap
When training the default **Decision Tree Regressor**, the model achieved near-perfect accuracy on training data but collapsed to a mere **1.00% score** on unseen testing data. This occurred because the unconstrained algorithm memorized individual rows and noise from the training set instead of generalizing broad patterns.

### The Sweet Spot Solution
To break the memorization loop, I implemented **Hyperparameter Tuning** by enforcing strict growth limits on the model structure:
```python
# Restricting the tree depth to ignore micro-level noise
tuned_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
