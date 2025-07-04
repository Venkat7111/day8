# 🔍 SVM Classification with Scikit-learn (Linear & Non-Linear)

This project demonstrates how to use **Support Vector Machines (SVM)** for both **linear** and **non-linear** binary classification using Scikit-learn in Google Colab.

---"""
# 🌸 Iris Dataset Analysis with Seaborn

This script demonstrates how to load and visualize the Iris dataset using Seaborn.

## 📊 Dataset
- 150 iris flower samples
- 4 features: sepal_length, sepal_width, petal_length, petal_width
- 3 species: setosa, versicolor, virginica

## 🖼️ Visualizations
- Pairplot (colored by species)
- Heatmap (correlation)
- Boxplots for each feature

## 📚 Requirements
Install dependencies using:
pip install seaborn pandas matplotlib
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Iris dataset from seaborn
df = sns.load_dataset("iris")
print("📥 First 5 rows of the Iris dataset:")
print(df.head())

# Plot: Pairplot to visualize all feature relationships
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.tight_layout()
plt.show()

# Plot: Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Plot: Boxplot of petal lengths across species
plt.figure(figsize=(6, 4))
sns.boxplot(x="species", y="petal_length", data=df)
plt.title("Petal Length Distribution by Species")
plt.tight_layout()
plt.show()

print("✅ Iris data analysis complete!")


## 📌 Objective

- Use **SVM** with both **Linear** and **RBF kernels**
- Visualize decision boundaries in 2D
- Evaluate model performance

---

## 🛠️ Tools Used

- Python
- Scikit-learn
- NumPy
- Matplotlib
- Google Colab

---

## 📚 Steps Followed

1. **Load 2D dataset** using `make_moons` from sklearn
2. **Split** data into training and testing sets
3. **Scale** features using `StandardScaler`
4. **Train** SVM with:
   - Linear Kernel
   - RBF (Radial Basis Function) Kernel
5. **Visualize** decision boundaries using `matplotlib`
6. **Evaluate** model accuracy on test data

---

## 📈 Results

- Compared performance between linear and non-linear SVMs
- Visualized how decision boundaries differ
- Showed RBF is better for non-linear patterns

---

## 🚀 How to Run

1. Open the project in **Google Colab**
2. Run each cell step by step
3. Observe decision boundaries and accuracy

> ✅ No installation required if running in Google Colab.

---

## 📎 Example Output

- SVM with Linear Kernel – straight decision boundary
- SVM with RBF Kernel – curved boundary that better fits complex data

---

## 🧠 Concepts Covered

- Binary Classification
- SVM Kernels (Linear vs. RBF)
- Data Scaling
- Decision Boundaries
- Model Accuracy

---

## 🔗 Related Topics

- Logistic Regression
- KNN Classifier
- Decision Trees
- Hyperparameter Tuning with GridSearchCV

---

## 📧 Contact

**Author**: Venkata Sai  
**GitHub**: [github.com/ven](https://github.com/ven)  
**Topic**: AI | ML | Web Dev | Automation  
