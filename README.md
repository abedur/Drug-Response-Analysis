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
