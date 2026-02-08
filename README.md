# Loan Allocation Classification (SVM & Decision Tree)

This repository contains a machine learning classification project developed as part of a graduate-level Machine Learning course.  
The goal is to predict loan approval risk using supervised learning, with a detailed comparison between **Decision Tree** and **Support Vector Machine (SVM)** classifiers.

The project covers the full machine learning pipeline, including data preprocessing, feature selection, class imbalance handling, model training, hyperparameter tuning, and detailed performance analysis.

---

## Dataset

The dataset consists of demographic, financial, and employment-related features for loan applicants.  
The target variable is `Risk_Flag`, indicating whether an applicant is considered risky.

**Preprocessing steps include:**
- Removal of irrelevant high-cardinality features (`CITY`, `STATE`)  
- Removal of identifier column (`Id`)  
- Encoding of categorical variables using `LabelEncoder`  
- Feature scaling using `StandardScaler` (with comparison to `MinMaxScaler`)  
- Stratified train-test split to preserve label distribution  

---

## Feature Engineering & Analysis

- Feature selection based on:
  - Number of unique values
  - Correlation analysis (after encoding categorical variables)
- Correlation matrix visualization to assess redundancy
- Explicit handling of categorical variables before correlation computation

---

## Class Imbalance Handling

The dataset is highly imbalanced. To address this:
- **RandomUnderSampler** is applied to the training set
- Oversampling was evaluated but avoided due to increased SVM training time
- Label distributions are visualized before and after resampling

---

## Models Implemented

### Decision Tree
- Trained using Gini impurity
- Evaluated on both training and test sets
- Tree structure exported using `export_graphviz`
- Hyperparameter tuning using `GridSearchCV`:
  - `criterion`
  - `max_depth`

### Support Vector Machine (SVM)
- Linear kernel
- Feature scaling applied prior to training
- Hyperparameter tuning using `GridSearchCV`:
  - Regularization parameter `C`

---

## Evaluation Metrics

For both models, the following metrics are reported:
- Accuracy (train and test)
- Confusion matrix
- Precision, recall, and F1-score (classification report)

Additionally, a **comparative error analysis** is performed by:
- Inspecting samples correctly classified by one model and misclassified by the other
- Analyzing cases where both models succeed or fail

---

## Project Structure

- `Loan_Classification.ipynb`  
  Complete Jupyter notebook containing preprocessing, training, evaluation, and analysis



---

## Key Takeaways

- Proper feature scaling is critical for SVM performance
- Undersampling provides a good balance between performance and computational efficiency
- Decision Trees offer interpretability, while SVMs show competitive performance after tuning
- Model comparison at the sample level provides deeper insight beyond aggregate metrics

---

## Requirements

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- imbalanced-learn  
- matplotlib  
- seaborn  

---

## Author

**Mahdieh Nouri**  
📧 [mahdiyenouri99@gmail.com](mailto:mahdiyenouri99@gmail.com)

---

## License

This project is intended for educational and academic use.
