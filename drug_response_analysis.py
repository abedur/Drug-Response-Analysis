"""
===============================================================
COURSE PROJECT: DRUG RESPONSE CLASSIFICATION
===============================================================

Project Title:
Why Do Some People React to the Same Drugs Differently Than Others?

Author:
Md Abedur Rahman

---------------------------------------------------------------
PROJECT DESCRIPTION
---------------------------------------------------------------
This Python program analyzes the Drug200 dataset to investigate
whether patient characteristics can be used to predict drug
classification.

The dataset contains the following variables:

1. Age          - Patient age
2. Sex          - Patient sex
3. BP           - Blood pressure category
4. Cholesterol  - Cholesterol category
5. Na_to_K      - Sodium-to-potassium ratio
6. Drug         - Drug class (target variable)

The analysis compares three machine-learning classification
models:

1. Decision Tree
2. Random Forest
3. Logistic Regression

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


---------------------------------------------------------------
HOW TO RUN THE PROGRAM
---------------------------------------------------------------

STEP 1:
Install Python 3 on your computer.

STEP 2:
Install the required Python libraries by running:

pip install pandas matplotlib scikit-learn

STEP 3:
Place these two files in the same folder:

    drug_response_analysis.py
    drug200.csv

STEP 4:
Open the folder in VS Code or another Python environment.

STEP 5:
Run the program using:

python drug_response_analysis.py

If "python" does not work on Windows, try:

py drug_response_analysis.py


---------------------------------------------------------------
PROGRAM SECTIONS
---------------------------------------------------------------

SECTION 1 - LOAD DATA
Loads drug200.csv into a Pandas DataFrame and displays basic
information about the dataset.

SECTION 2 - DATA EXPLORATION
Calculates descriptive statistics, checks missing values, and
examines the distribution of the five drug categories.

SECTION 3 - DATA VISUALIZATION
Creates charts showing the distribution of drug classes and
differences in the Na_to_K ratio among drug groups.

SECTION 4 - DATA PREPROCESSING
Separates the predictor variables from the target variable.
Categorical variables (Sex, BP, and Cholesterol) are converted
using one-hot encoding. Numerical variables (Age and Na_to_K)
are standardized.

SECTION 5 - TRAIN/TEST SPLIT
Divides the dataset into 80% training data and 20% testing data.
A random_state of 42 is used to make the analysis reproducible.
Stratification is used to preserve the distribution of drug
classes.

SECTION 6 - MACHINE LEARNING
Trains and evaluates three classification models:
Decision Tree, Random Forest, and Logistic Regression.

SECTION 7 - MODEL EVALUATION
Compares the models using accuracy, weighted precision,
weighted recall, weighted F1 score, and confusion matrices.

SECTION 8 - OUTPUT
Displays model results and saves visualization files that can
be used in the written report and PowerPoint presentation.


---------------------------------------------------------------
IMPORTANT NOTE
---------------------------------------------------------------
This project is an educational machine-learning analysis.
The Drug200 dataset represents drug classification based on
selected patient characteristics. It does not directly measure
clinical drug effectiveness, adverse reactions, or genetic
differences. Therefore, the results should not be interpreted
as medical or prescribing recommendations.
===============================================================
"""
# Importing required libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load data
df = pd.read_csv("drug200.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())

# 2. Descriptive analysis
print("\nNumeric summary:\n", df[["Age", "Na_to_K"]].describe())
print("\nDrug counts:\n", df["Drug"].value_counts())
print("\nDrug by blood pressure:\n", pd.crosstab(df["BP"], df["Drug"]))
print("\nDrug by cholesterol:\n", pd.crosstab(df["Cholesterol"], df["Drug"]))
print("\nDrug by sex:\n", pd.crosstab(df["Sex"], df["Drug"]))
print("\nGroup statistics:\n",
      df.groupby("Drug")[["Age", "Na_to_K"]].agg(["mean","std","min","max"]))

# 3. Visualizations
drug_order = ["drugA", "drugB", "drugC", "drugX", "drugY"]

plt.figure(figsize=(7, 4.5))
df["Drug"].value_counts().reindex(drug_order).plot(kind="bar")
plt.title("Distribution of Drug Classes")
plt.xlabel("Drug Class")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("figure1_drug_distribution.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
groups = [df.loc[df["Drug"] == d, "Na_to_K"] for d in drug_order]
plt.boxplot(groups, tick_labels=drug_order)
plt.title("Sodium-to-Potassium Ratio by Drug Class")
plt.xlabel("Drug Class")
plt.ylabel("Na-to-K Ratio")
plt.tight_layout()
plt.savefig("figure2_natok_boxplot.png", dpi=200)
plt.close()

# 4. Prepare predictors and outcome
X = df.drop(columns="Drug")
y = df["Drug"]

categorical_features = ["Sex", "BP", "Cholesterol"]
numeric_features = ["Age", "Na_to_K"]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("numeric", StandardScaler(), numeric_features)
])

# Stratification keeps class proportions similar in training/testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Fit and evaluate classification models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000)
}

results = []
confusion_matrices = {}

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    })

    confusion_matrices[name] = confusion_matrix(
        y_test, predictions, labels=drug_order
    )

    print(f"\n{name}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(confusion_matrices[name])

results_df = pd.DataFrame(results)
print("\nModel comparison:\n", results_df.round(3))
results_df.to_csv("model_metrics.csv", index=False)

# Plot model performance
results_df.set_index("Model")[
    ["Accuracy", "Weighted Precision",
     "Weighted Recall", "Weighted F1"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance on the Test Set")
plt.xlabel("")
plt.ylabel("Score")
plt.ylim(0.85, 1.01)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "figure3_model_performance.png",
    dpi=300,
    bbox_inches="tight"
)

#plt.show()

# 6. Plot Random Forest confusion matrix
cm = confusion_matrices["Random Forest"]
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(len(drug_order)), drug_order, rotation=45)
plt.yticks(range(len(drug_order)), drug_order)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("figure4_confusion_matrix.png", dpi=200)
plt.close()

print("\nAnalysis complete.")
