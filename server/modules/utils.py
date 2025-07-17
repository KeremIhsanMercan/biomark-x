import os
import sys

import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import dill
import pickle

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import classification_report


from modules.exception import CustomException
from modules.logger import logging

# Save a Python object to a file using pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Saved Model object at '{file_path}'")
    except Exception as e:
        raise CustomException(e, sys)

    
# Load a Python object from a file using pickle

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

        logging.info(f"Loaded object from '{file_path}'")

    except Exception as e:
        raise CustomException(e, sys)
    

# Load JSON data as a dictionary

def load_json(json_path):
    """
    Load json data as a dictionary
    """

    try:
        with open(json_path) as f:
            file = f.read()
        json_data = json.loads(file)
        logging.info(f"Loaded JSON data from '{json_path}'")
    except Exception as e:
        raise CustomException(e, sys)

    return json_data

# Save a dictionary or object as a JSON file

def save_json(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        json_object = json.dumps(obj, indent=2)

        with open(file_path, "w") as file_obj:
            file_obj.write(json_object)
        logging.info(f"Saved JSON data at '{file_path}'")

    except Exception as e:
        raise CustomException(e, sys)

# Train and evaluate multiple models using cross-validation and test sets

def evaluate_models(X_train, 
                    y_train,
                    X_test,
                    y_test,
                    models,
                    param, 
                    param_finetune = True, 
                    n_folds = 10, 
                    finetune_fraction = 1.0, 
                    verbose:bool = True,
                    outdir = None
                   ):
    """
    Train and evaluate multiple models using cross-validation and test sets.

    Args:
    X_train, y_train: Training features and labels.
    X_test, y_test: Test features and labels.
    models (dict): Dictionary of model names and model objects.
    param (dict): Dictionary of hyperparameters for models.
    param_finetune (bool): If True, perform hyperparameter tuning using GridSearchCV.
    n_folds (int): Number of folds for cross-validation.
    finetune_fraction (float): Fraction of training data used for hyperparameter tuning.
    verbose (bool): If True, print training and evaluation details.
    outdir (str): Output directory to save model evaluation results as tables.

    Returns:
    dict: Report containing cross-validation, training, and test performance for each model.

    Raises:
    CustomException: If any error occurs during model training or evaluation.
    """
    try:
        report = {}

        kfold = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle = True)
        scoring = "f1"

        if verbose:
            length = 110
            print("=" * length)
            print(" Starting Model Training and Evaluation")
            print("=" * length)

        logging.info("Interating over Models")
        for i in range(len(list(models))):
            model = list(models.items())[i][1]
            model_name = list(models.items())[i][0]

            logging.info(f"Training {model_name} model")
            if verbose:
                print("=" * length)
                print(f" Starting {model_name} Model Training and Evaluation")
                print("=" * length)
            if param_finetune:
                logging.info(f"Fine tunning {model_name} model")
                n_samples = int(finetune_fraction * len(y_train))
                # Combine X_train and y_train into a single DataFrame
                train_data = pd.concat([pd.DataFrame(X_train.reset_index(drop= True)), pd.DataFrame(y_train)], axis=1)
                
                # Sample the combined data
                sampled_data = train_data.sample(n=n_samples, random_state=32)
                
                # Split back into X_train_cv and y_train_cv
                X_train_cv = sampled_data.iloc[:, :-1].values  # All columns except the last one
                y_train_cv = sampled_data.iloc[:, -1].values   # The last column

                para=param[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=kfold, scoring = scoring)
                gs.fit(X_train_cv,y_train_cv)
                model_best_params = gs.best_params_
                model.set_params(**model_best_params)
            model.fit(X_train,y_train)

            #get cross validation and test report
            logging.info(f"Cross Validating {model_name} model")
            cross_val_report = get_cross_validation_scores(model, X_train,y_train, cv= kfold)
            
            # make predictions
            
            y_train_pred = model.predict(X_train)  # prediction on train set
            y_test_pred = model.predict(X_test)   # prediction on test set

            logging.info(f"Testing {model_name} model")

            train_report = get_test_report(y_train, y_train_pred)
            test_report = get_test_report(y_test, y_test_pred)
            report[list(models.keys())[i]] = {"test_report":test_report, 
                                              "train_report":train_report, 
                                              "cross_val_report":cross_val_report}

            # 
            cv1 = cross_val_report["accuracy"]["mean"]
            cv2 = cross_val_report["precision"]["mean"]
            cv3 = cross_val_report["recall"]["mean"]
            cv4 = cross_val_report["f1"]["mean"]
            cv5 = cross_val_report["roc_auc"]["mean"]
            cv6 = int(len(y_train)/n_folds)
    
            # 
            tr1 = train_report["accuracy"]
            tr2 = train_report["precision"]
            tr3 = train_report["recall"]
            tr4 = train_report["f1"]
            tr5 = train_report["roc_auc"]
            tr6 = len(y_train)
    
            # 
            te1 = test_report["accuracy"]
            te2 = test_report["precision"]
            te3 = test_report["recall"]
            te4 = test_report["f1"]
            te5 = test_report["roc_auc"]
            te6 = len(y_test)
            
            {'f1': 1.0, 'accuracy': 1.0, 'roc_auc': 1.0, 'precision': 1.0, 'recall': 1.0}
            s = f"""              
                                    accuracy    precision    recall    f1-score    roc_auc    support\n   
                cross validation    {cv1:.2f}        {cv2:.2f}         {cv3:.2f}      {cv4:.2f}        {cv5:.2f}       {cv6}
                train set           {tr1:.2f}        {tr2:.2f}         {tr3:.2f}      {tr4:.2f}        {tr5:.2f}       {tr6}
                test set            {te1:.2f}        {te2:.2f}         {te3:.2f}      {te4:.2f}        {te5:.2f}       {te6}\n
                """
            print(s)
                
            # Save model results as a table (if outdir is specified)
            if outdir:
                import matplotlib.pyplot as plt
                import matplotlib
                from matplotlib.backends.backend_pdf import PdfPages
                
                # Create model output directory
                model_outdir = os.path.join(outdir, 'models', model_name)
                os.makedirs(os.path.join(model_outdir, 'png'), exist_ok=True)
                os.makedirs(os.path.join(model_outdir, 'pdf'), exist_ok=True)
                
                # Create table data
                data = [
                    ['Cross Val', f"{cv1:.2f}", f"{cv2:.2f}", f"{cv3:.2f}", f"{cv4:.2f}", f"{cv5:.2f}", f"{cv6}"],
                    ['Train Set', f"{tr1:.2f}", f"{tr2:.2f}", f"{tr3:.2f}", f"{tr4:.2f}", f"{tr5:.2f}", f"{tr6}"],
                    ['Test Set', f"{te1:.2f}", f"{te2:.2f}", f"{te3:.2f}", f"{te4:.2f}", f"{te5:.2f}", f"{te6}"]
                ]
                
                # Create table
                fig, ax = plt.figure(figsize=(12, 1)), plt.subplot(111)
                ax.axis('off')
                ax.axis('tight')
                table = ax.table(cellText=data,
                                colLabels=['', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Support'],
                                loc='center',
                                cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(16)
                table.scale(1, 2.2)

                # Add title
                feature_status = "After Feature Selection" if "AfterFeatureSelection" in outdir else "Without Feature Selection"
                fig.suptitle(f'Results for Model: {model_name} ({feature_status})', fontsize=18, y=2)
                
                # Save as PNG
                png_path = os.path.join(model_outdir, 'png', f'{model_name}_results.png')
                plt.savefig(png_path, bbox_inches='tight', dpi=300)
                
                # Save as PDF
                with PdfPages(os.path.join(model_outdir, 'pdf', f'{model_name}_results.pdf')) as pdf:
                    pdf.savefig(fig, bbox_inches='tight')
                
                plt.close()
                logging.info(f"Model results saved to {model_outdir} directory.")
                
                # Print file path to stdout (to be captured by Node.js)
                relative_path = png_path.split('server/')[-1] if 'server/' in png_path else png_path
                print(relative_path)
                
        if verbose:
            print("=" * length)
            print(f" Model Training and Evaluation Completed")
            print("=" * length)
                
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
# Get cross validation scores for classification

def get_cross_validation_scores(model, X, y, cv):
    """
    Get cross validation scores:
        ('f1', 'precision', 'recall', 'roc_auc', "accuracy") for classification
    """
    try:
        scoring = ('f1', 'precision', 'recall', 'roc_auc', "accuracy")
        scores = cross_validate(model, X, y, cv=cv,scoring=scoring,return_train_score=False)
        score_report = {"_".join(score_name.split("_")[1:]):{"mean":scores[score_name].mean(), 
                                                         "std":scores[score_name].std(),
                                                         "all":list(scores[score_name])} for score_name in scores}
    except Exception as e:
        raise CustomException(e, sys)
    
    return score_report

# Get test set evaluation metrics

def get_test_report(true, predicted):

    """
    Run Various Evaluation Metrics on data
    """
    try:
        score_report = {"f1":f1_score(true, predicted),
                        "accuracy": accuracy_score(true, predicted),
                        "roc_auc": roc_auc_score(true, predicted),
                        "precision":precision_score(true, predicted),
                        "recall":recall_score(true, predicted)
                    }
        return score_report

    except Exception as e:
        raise CustomException(e, sys)

# Get tunable hyperparameters for various machine learning models

def getparams():

    """
    Returns a dictionary of tunable hyperparameters for various machine learning models.
    
    The dictionary includes common models like Decision Tree, Random Forest, Gradient Boosting,
    Logistic Regression, XGBClassifier, CatBoosting Classifier, AdaBoost Classifier, MLPClassifier, and SVC.
    
    Returns:
        dict: A dictionary where each key is a model name and the value is a dictionary of hyperparameters.
    """
    
    params = {
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"]
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Gradient Boosting": {
            "learning_rate": [0.001, 0.01, 0.1, 0.2],
            "n_estimators": [100, 200, 300],
            "subsample": [0.5, 0.7, 1.0],
            "criterion": ["friedman_mse", "squared_error"],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"]
        },
        "Logistic Regression": {
            "penalty": ["l2", "elasticnet", None],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "liblinear", "sag", "saga"],
            "max_iter": [100, 200, 300],
        },
        "XGBClassifier": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.001, 0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.5, 0.7, 1.0],
            "gamma": [0, 0.1, 0.2],
        },
        "CatBoosting Classifier": {
            "iterations": [100, 200, 500],
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "depth": [3, 5, 7, 10],
            "l2_leaf_reg": [1, 3, 5, 7],
            "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"]
        },
        "AdaBoost Classifier": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.1, 1.0],
            "algorithm": ["SAMME", "SAMME.R"]
        },
        "MLPClassifier": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "max_iter": [200, 300, 400]
        },
        "SVC": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [3, 4],
            "gamma": ["scale", "auto"]
        }
    }
    return params


