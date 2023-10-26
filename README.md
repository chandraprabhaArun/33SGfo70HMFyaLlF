

---

# Machine Learning Model Evaluation

This repository contains a collection of machine learning models and their respective performance evaluations on a specific dataset. Each model is trained and tested on the dataset, and their accuracy scores are reported for reference.

## Models and Their Performance

### 1. Stochastic Gradient Descent (SGD)
- Accuracy Score: 57.69%

The Stochastic Gradient Descent model achieved an accuracy score of 57.69%. This score reflects the model's ability to make correct predictions on the given dataset.

### 2. Linear Support Vector Classifier (SVC)
- Accuracy Score: 50.0%

The Linear Support Vector Classifier yielded an accuracy score of 50.0%. This score measures the model's classification accuracy on the dataset.

### 3. SVC Parameter Tuning with GridSearch
- Accuracy: 53.85%

By applying hyperparameter tuning with GridSearch, the Support Vector Classifier improved its accuracy to 53.85%. This demonstrates the impact of hyperparameter optimization on model performance.

### 4. K-Nearest Neighbors (KNN)
- Accuracy: 76.9%

The K-Nearest Neighbors model achieved a notable accuracy score of 76.9%. This high accuracy suggests that KNN is well-suited for the given dataset.

### 5. Decision Tree
- Accuracy: 53.85%

The Decision Tree model achieved an accuracy score of 53.85%. This score reflects the model's performance in making decisions based on the dataset's features.

### 6. Logistic Regression
- Accuracy: 62.0%

Logistic Regression achieved an accuracy score of 62.0%. This logistic regression model is suitable for binary classification tasks and performed well on this dataset.

### 7. Random Forest
- Accuracy Score: 73.6%

The Random Forest model achieved an accuracy score of 73.6%. Random Forest is an ensemble model, and its high accuracy reflects its ability to combine multiple decision trees effectively.

### Hyperparameter Tuning and Model Details

In addition to presenting model accuracy scores, this repository includes information about hyperparameter tuning and model details:

- **Hyperparameter tuning using RandomizedSearch CV:** The Random Forest model was fine-tuned using RandomizedSearch Cross-Validation to optimize its hyperparameters.

- **Hyperparameter Tuning - GridSearchCV:** The Support Vector Classifier was enhanced through GridSearchCV, allowing us to find the best combination of hyperparameters for improved performance.

- **Random Forest Classifier model with default parameters:** The initial performance of the Random Forest model using default parameters is provided for reference.

- **Feature Scaling:** Note that feature scaling techniques were applied to ensure that the input features were properly scaled before model training.

---
