# Course Project: Drug Response Classification

## Project Title
**Why Do Some People React to the Same Drugs Differently Than Others?**

**Author:** Md Abedur Rahman

---

## Project Description

This Python program analyzes the **Drug200 dataset** to investigate whether patient characteristics can be used to predict drug classification.

The dataset contains the following variables:

1. **Age** - Patient age
2. **Sex** - Patient sex
3. **BP** - Blood pressure category
4. **Cholesterol** - Cholesterol category
5. **Na_to_K** - Sodium-to-potassium ratio
6. **Drug** - Drug class (target variable)

The analysis compares three machine-learning classification models:

- Decision Tree
- Random Forest
- Logistic Regression

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## How to Run the Program

### Step 1: Install Python

Install **Python 3** on your computer.

You can verify the installation by running:

```bash
python --version
```

On Windows, you can also try:

```bash
py --version
```

### Step 2: Install Required Libraries

Open a terminal or the VS Code terminal and run:

```bash
python -m pip install pandas matplotlib scikit-learn
```

If `python` does not work on Windows, use:

```bash
py -m pip install pandas matplotlib scikit-learn
```

### Step 3: Prepare the Project Files

Place these two files in the same folder:

```text
Drug_Response_Project/
├── drug_response_analysis.py
└── drug200.csv
```

### Step 4: Open the Project

Open the project folder in **Visual Studio Code (VS Code)** or another Python development environment.

### Step 5: Run the Program

In the terminal, run:

```bash
python drug_response_analysis.py
```

If `python` does not work on Windows, try:

```bash
py drug_response_analysis.py
```

---

## Program Sections

### Section 1 - Load Data
Loads `drug200.csv` into a Pandas DataFrame and displays basic information about the dataset.

### Section 2 - Data Exploration
Calculates descriptive statistics, checks missing values, and examines the distribution of the five drug categories.

### Section 3 - Data Visualization
Creates charts showing the distribution of drug classes and differences in the `Na_to_K` ratio among drug groups.

### Section 4 - Data Preprocessing
Separates the predictor variables from the target variable. Categorical variables (`Sex`, `BP`, and `Cholesterol`) are converted using one-hot encoding. Numerical variables (`Age` and `Na_to_K`) are standardized.

### Section 5 - Train/Test Split
Divides the dataset into **80% training data and 20% testing data**. A `random_state` of 42 is used to make the analysis reproducible. Stratification is used to preserve the distribution of drug classes.

### Section 6 - Machine Learning
Trains and evaluates three classification models:

- Decision Tree
- Random Forest
- Logistic Regression

### Section 7 - Model Evaluation
Compares the models using accuracy, weighted precision, weighted recall, weighted F1 score, and confusion matrices.

### Section 8 - Output
Displays model results and saves visualization files that can be used in the written report and PowerPoint presentation.

The program generates:

```text
figure1_drug_distribution.png
figure2_natok_boxplot.png
figure3_confusion_matrix.png
model_metrics.csv
```

---

## Important Note

This project is an educational machine-learning analysis. The Drug200 dataset represents drug classification based on selected patient characteristics. It does not directly measure clinical drug effectiveness, adverse reactions, or genetic differences. Therefore, the results should not be interpreted as medical or prescribing recommendations.
