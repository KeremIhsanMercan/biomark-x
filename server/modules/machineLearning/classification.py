import os, json
import sys
import numpy as np 
import pandas as pd
from sklearn import set_config
set_config(transform_output = "pandas")

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


from modules.exception import CustomException
from modules.logger import logging
from modules.utils import save_object, evaluate_models, load_json, save_json, getparams

class Classification:
    """
    A class for performing machine learning classification tasks, including data preprocessing,
    model training, and model evaluation with cross-validation.

    Parameters
    ----------
    data : pd.DataFrame, optional
        The input dataset. Default is None.
    labels_column : str, optional
        The column name of the target variable (labels). Default is "Diagnosis".
    n_folds : int, optional
        Number of cross-validation folds. Default is 10.
    test_size : float, optional
        Proportion of the data to use for the test set. Default is 0.2.
    outdir : str, optional
        Directory to save outputs such as models, transformers, and reports. Default is "outputs".
    param_finetune : bool, optional
        Whether to perform hyperparameter tuning. Default is False.
    finetune_fraction : float, optional
        Fraction of the dataset to use for hyperparameter tuning. Default is 1.0.
    save_best_model : bool, optional
        Whether to save the best performing model. Default is False.
    standard_scaling : bool, optional
        Whether to apply standard scaling to numerical features. Default is False.
    save_data_transformer : bool, optional
        Whether to save the data preprocessing transformer object. Default is False.
    save_label_encoder : bool, optional
        Whether to save the label encoder object. Default is False.
    model_list : list, optional
        List of models to use for classification. Default includes "Logistic Regression" and "Decision Tree".
    verbose : bool, optional
        Whether to print detailed logs during execution. Default is True.

    Attributes
    ----------
    data : pd.DataFrame
        The input dataset with modified column names.
    labels_column : str
        The column name of the target variable (labels).
    n_folds : int
        Number of cross-validation folds.
    test_size : float
        Proportion of the data used for testing.
    param_finetune : bool
        Whether to perform hyperparameter tuning.
    finetune_fraction : float
        Fraction of data used for hyperparameter tuning.
    model_list : list
        List of models to use for classification.
    verbose : bool
        Whether to print detailed logs.
    """
    
    def __init__(self,
                 data = None,
                 labels_column:str = "Diagnosis",
                 n_folds:int = 10,
                 test_size:float = 0.2,
                 outdir:str="outputs",
                 param_finetune:bool = False,
                 finetune_fraction:float = 1.0,
                 save_best_model:bool = False,
                 standard_scaling:bool = False,
                 save_data_transformer:bool = False,
                 save_label_encoder:bool = False,
                 model_list:list = ["Logistic Regression", "Decision Tree"],
                 verbose:bool = True
                 ):
        """
        Initialize the Classification class with dataset and configuration options.
        """

        self.data = data
        self.data.columns = [f"Feature_{i}" if feature != labels_column else feature for i, feature in enumerate(data.columns)]  
        self.outdir = outdir
        self.labels_column = labels_column
        self.n_folds = n_folds
        self.test_size = test_size
        self.param_finetune = param_finetune
        self.finetune_fraction = finetune_fraction
        self.save_best_model = save_best_model
        self.standard_scaling = standard_scaling
        self.save_data_transformer = save_data_transformer
        self.save_label_encoder = save_label_encoder
        self.model_list = model_list
        self.verbose = verbose
        

    def build_transformer(self):
        
        """
        Create a data preprocessing pipeline for numerical and categorical features.

        Returns
        -------
        ColumnTransformer
            A transformer object for numerical and categorical feature processing.
        
        Raises
        ------
        CustomException
            If an error occurs during transformer construction.
        """
        try:
            # set standard scaling configs
            if self.standard_scaling:
                num_standard_scaler = StandardScaler()
                cat_standard_scaler = StandardScaler(with_mean=False)
            else:
                num_standard_scaler = None
                cat_standard_scaler = None
            
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",num_standard_scaler),
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("categorical_encoder",OneHotEncoder(sparse_output=False)),
                ("scaler",cat_standard_scaler)
                ]
            )
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,self.numerical_columns),
                ("cat_pipelines",cat_pipeline,self.categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    
    def data_transfrom(self):
        """
        Split the data into training and test sets, encode labels, and apply preprocessing.

        Returns
        -------
        None

        Raises
        ------
        CustomException
            If an error occurs during data transformation.
        """

        logging.info("Transforming Data: Encoding Labels, Creating a Train/Test Split, Transforming Features")
        try:
            # create train test split
            self.X = self.data.drop(self.labels_column, axis = 1)
            self.labels = self.data[self.labels_column]
    
            # encode labels
            label_encoder = LabelEncoder()
            self.y = label_encoder.fit_transform(self.labels)
    
            # save encoder
            if self.save_label_encoder:
                save_object(f"{self.outdir}/artifacts/label_encoder.pkl", label_encoder)
            
            X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=32)
    
            # initialize transformer
            self.numerical_columns = [feature for feature in self.X.columns if self.X[feature].dtype != 'O']
            self.categorical_columns = [feature for feature in self.X.columns if self.X[feature].dtype == 'O']
            preprocessor = self.build_transformer()
            
            # fit transformer and transform training data
            self.X_train = preprocessor.fit_transform(X_train)
            # transform test data
            self.X_test = preprocessor.transform(X_test)
    
            # save transformer
            if self.save_data_transformer:
                save_object(f"{self.outdir}/artifacts/preprocessor.pkl", preprocessor)

        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_model_trainer(self):
        """
        Train models and evaluate their performance using cross-validation.

        Returns
        -------
        tuple
            A tuple containing the best model name and its score.

        Raises
        ------
        CustomException
            If an error occurs during model training or evaluation.
        """
        try:

            models_base = {
                "Logistic Regression": LogisticRegression(random_state = 32),
                "Random Forest": RandomForestClassifier(random_state = 32),
                "XGBClassifier": XGBClassifier(random_state = 32),
                "Decision Tree": DecisionTreeClassifier(random_state = 32), 
                "Gradient Boosting": GradientBoostingClassifier(random_state = 32),
                "CatBoosting Classifier": CatBoostClassifier(random_state = 32,verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(random_state = 32),
                "MLPClassifier": MLPClassifier(random_state = 32, verbose=False),
                "SVC": SVC(kernel="linear",random_state = 32, probability=True)
                }
            models = {model_name:models_base[model_name] for model_name in self.model_list}
            
            params = getparams()

            logging.info("TRAINING AND EVALUATING MODELS")
            model_report:dict=evaluate_models(X_train=self.X_train,
                                              y_train=self.y_train,
                                              X_test=self.X_test,
                                              y_test=self.y_test,
                                              models=models,
                                              param=params,
                                              n_folds = self.n_folds,
                                              param_finetune = self.param_finetune,
                                              finetune_fraction = self.finetune_fraction,
                                              verbose = self.verbose,
                                              outdir = self.outdir
                                              )
            
            # Update the JSON file with new model results without resetting existing content
            json_path = f"{self.outdir}/model_reports.json"            
            
            # Read existing data if available
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    try:
                        existing_data = json.load(f)  # Read JSON file
                    except json.JSONDecodeError:
                        existing_data = {}  # If JSON is corrupted, start with empty dict
            else:
                existing_data = {}  # If file does not exist, start empty

            # Update existing data with new model results
            existing_data.update(model_report)  # Add new results to existing data

            # Save updated JSON
            with open(json_path, "w") as f:
                json.dump(existing_data, f, indent=4)
                
            ## To get best model score from dict
            best_model_score = 0
            best_model_name = ""

            for model_name in model_report:
                score = model_report[model_name]["cross_val_report"]["f1"]["mean"]

                if score >= best_model_score:
                    best_model_score = score
                    best_model_name = model_name
            
            best_model = models[best_model_name]

            if best_model_score < 0.1:
                raise CustomException("No best model found", sys)

            # save best model
            if self.save_best_model:
                save_object(f"{self.outdir}/artifacts/best_model_{best_model_name}.pkl",best_model)
            
            # Update the JSON file with new model results without resetting existing content
            json_path = f"{self.outdir}/model_reports.json"            
            
            # Read existing data if available
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    try:
                        existing_data = json.load(f)  # Read JSON file
                    except json.JSONDecodeError:
                        existing_data = {}  # If JSON is corrupted, start with empty dict
            else:
                existing_data = {}  # If file does not exist, start empty

            # Update existing data with new model results
            existing_data.update(model_report)  # Add new results to existing data

            # Save updated JSON
            with open(json_path, "w") as f:
                json.dump(existing_data, f, indent=4)
                        
            logging.info("MODEL TRAINING AND EVALUATION COMPLETE")
            print(best_model_name)
            print(f"Best model: {best_model_name}\nBest model cross validation score: {best_model_score}")
            
            
        except Exception as e:
            raise CustomException(e,sys)