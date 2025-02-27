from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from lightgbm import LGBMClassifier
import pandas as pd

# Define the models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': LGBMClassifier()
}

# Define the parameter grids for each model
param_grids = {
    'Random Forest': {'n_estimators': [100], 'max_depth': [3, 4, 5, 6]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf']},
    'KNN': {'n_neighbors': [3, 5, 7, 9, 11, 15]},
    'Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    'XGBoost': {'n_estimators': [100], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5, 6]},
    'LightGBM': {'n_estimators': [100], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5, 6]}
}

def train_and_validate(models, train_data, train_labels, param_grids, skf):
    """
    Trains machine learning models using a training dataset, performs hyperparameter tuning 
    using GridSearchCV with cross-validation, and evaluates performance on a validation set.
    
    Parameters:
        models (dict): Dictionary of model names and their corresponding initialized classifiers.
        train_data (array-like): Feature matrix for training.
        train_labels (array-like): Labels corresponding to the training data.
        param_grids (dict): Dictionary of hyperparameter grids for each model.

    Saves:
        - 'crossval_results.csv': Cross-validation results (averaged across folds).
        - 'internal_results.csv': Validation results on the hold-out validation set.
    """
    crossval_results, internal_results = [], []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )

        # Perform hyperparameter tuning using GridSearchCV
        grid = GridSearchCV(model, param_grid=param_grids.get(model_name, {}), cv=skf,
                            scoring=scoring, refit='accuracy', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        # Get the best model from GridSearchCV
        best_model = grid.best_estimator_
        best_results = grid.cv_results_
        best_index = best_results['mean_test_accuracy'].argmax()

        # Store cross-validation results
        crossval_results.append([
            model_name,
            best_results['mean_test_accuracy'][best_index],
            best_results['mean_test_roc_auc'][best_index],
            best_results['mean_test_aucpr'][best_index],
            best_results['mean_test_sensitivity'][best_index],
            best_results['mean_test_specificity'][best_index],
            best_results['mean_test_balanced_accuracy'][best_index],
            best_results['mean_test_f1_macro'][best_index],
            best_results['mean_test_mcc'][best_index],
        ])

        # Evaluate on the validation set
        y_pred = best_model.predict(X_val)
        y_pred_prob = best_model.predict_proba(X_val)[:, 1]
        internal_results.append([model_name] + list(evaluate(y_val, y_pred_prob).values()))

    # Save results
    pd.DataFrame(crossval_results, columns=[
        "Model", "Accuracy", "ROC AUC", "AUC-PR", "Sensitivity",
        "Specificity", "Balanced Accuracy", "F1 Score", "MCC"
    ]).to_csv("crossval_results.csv", index=False)

    pd.DataFrame(internal_results, columns=[
        "Model", "Accuracy", "Balanced Accuracy", "ROC AUC", "AUC-PR",
        "Sensitivity", "Specificity", "F1 Score", "MCC"
    ]).to_csv("internal_results.csv", index=False)

    print("\nValidation results saved successfully!")


def train_and_test(models, train_data, train_labels, test_data, test_labels, param_grids, skf):
    """
    Trains machine learning models on the full training dataset, performs hyperparameter tuning 
    using GridSearchCV, and evaluates performance on a separate test dataset.

    Parameters:
        models (dict): Dictionary of model names and their corresponding initialized classifiers.
        train_data (array-like): Feature matrix for training.
        train_labels (array-like): Labels corresponding to the training data.
        test_data (array-like): Feature matrix for testing.
        test_labels (array-like): Labels corresponding to the test data.
        param_grids (dict): Dictionary of hyperparameter grids for each model.

    Saves:
        - 'cv_results_ML_DNABERT.csv': Cross-validation results.
        - 'test_results_ML_DNABERT.csv': Test set evaluation results.
    """
    crossval_results, external_results = [], []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Perform hyperparameter tuning using GridSearchCV
        grid = GridSearchCV(model, param_grid=param_grids.get(model_name, {}), cv=skf,
                            scoring=scoring, refit='accuracy', n_jobs=-1, verbose=1)
        grid.fit(train_data, train_labels)

        # Get the best model from GridSearchCV
        best_model = grid.best_estimator_
        best_results = grid.cv_results_
        best_index = best_results['mean_test_accuracy'].argmax()

        # Store cross-validation results
        crossval_results.append([
            model_name,
            best_results['mean_test_accuracy'][best_index],
            best_results['mean_test_roc_auc'][best_index],
            best_results['mean_test_aucpr'][best_index],
            best_results['mean_test_sensitivity'][best_index],
            best_results['mean_test_specificity'][best_index],
            best_results['mean_test_balanced_accuracy'][best_index],
            best_results['mean_test_f1_macro'][best_index],
            best_results['mean_test_mcc'][best_index],
        ])

        # Evaluate on the test set
        y_pred = best_model.predict(test_data)
        y_pred_prob = best_model.predict_proba(test_data)[:, 1]
        external_results.append([model_name] + list(evaluate(test_labels, y_pred_prob).values()))

    # Save results
    pd.DataFrame(crossval_results, columns=[
        "Model", "Accuracy", "ROC AUC", "AUC-PR", "Sensitivity",
        "Specificity", "Balanced Accuracy", "F1 Score", "MCC"
    ]).to_csv("cv_results_ML_DNABERT.csv", index=False)

    pd.DataFrame(external_results, columns=[
        "Model", "Accuracy", "ROC AUC", "AUC-PR", "Sensitivity",
        "Specificity", "Balanced Accuracy", "F1 Score", "MCC"
    ]).to_csv("test_results_ML_DNABERT.csv", index=False)

    print("\nTest results saved successfully!")