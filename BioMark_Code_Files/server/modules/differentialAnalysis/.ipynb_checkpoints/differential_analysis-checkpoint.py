# Load general packages
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm

# Load sklearn packages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn import set_config

# Load visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Load statistical packages
from scipy.stats import rankdata, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load specialized packages
import shap
import lime
import lime.lime_tabular

# Load Custom Modules
from modules.utils import save_json, load_json
from modules.differentialAnalysis.shap_analysis import SHAP_Analysis
from modules.differentialAnalysis.lime_analysis import LIME_Analysis
from modules.differentialAnalysis.feature_importance_analysis import FeatureImportance_Analysis
from modules.logger import logging


# Set sklearn configuration
set_config(transform_output="pandas")


class DifferentiatingFactorAnalysis:
    """
    The DifferentiatingFactorAnalysis class is designed to perform various statistical and machine learning analyses
    to identify the features that differentiate between different classes or conditions in a dataset. 

    This class includes methods for the following analyses:
    
    - SHAP Analysis: Uses SHAP (SHapley Additive exPlanations) to interpret the output of machine learning models. In this class, a shap summary plot, cluster plot and feature importances are generated.
    - Lime Analysis: Utilizes LIME (Local Interpretable Model-agnostic Explanations) to provide local explanations for model predictions. A lime summary plot and feature importances are generated.
    - ANOVA: Conducts Analysis of Variance (ANOVA) to identify features that have significant differences between groups.
    - t-test: Performs t-tests to compare the means of two groups and identify significant differences.
    - Feature Importance Analysis: Evaluates the importance of features in a predictive model to determine which features have the most influence on the outcome.

    Methods
    -------
    perform_shap_analysis():
        Performs SHAP analysis on the provided model and dataset.
    
    perform_lime_analysis(y):
        Performs LIME analysis on the provided model and dataset.
    
    perform_anova():
        Conducts ANOVA to find features with significant differences between groups.
    
    perform_t_test():
        Performs t-tests to identify significant differences between two groups.
    
    perform_randomforest_feature_importance_analysis():
        Computes Random Forest Feature Importances.

    perform_xgb_feature_importance_analysis():
        Computes XGBoost Feature Importances.

    perform_permutation_feature_importance_analysis():
        Computes Permutation Feature Importances Using Random Forest Model.

    run_all_analyses():
        Run All Methods Listed from the analyses parameter.


    Attributes
    ----------
    
    X : DataFrame or array-like
        The feature matrix.
    
    y : Series or array-like
        The target variable.
    """

    
    def __init__(self, 
                 data=None, 
                 analyses=None, 
                 labels_column="Diagnosis", 
                 reference_class:str = "Control",
                 sample_id_column="Sample ID", 
                 outdir: str = "output", 
                 mode: str = "classification",
                 feature_type: str = "microRNA",
                 test_size: float = 0.2,
                 lime_global_explanation_sample_num:int =10,
                 shap_model_finetune:bool = False,
                 lime_model_finetune:bool = False,
                 n_folds:int = 5,
                 scoring:str = "f1",
                 top_features_to_plot:int = 20,
                 feature_importance_finetune:bool = True
                ):
        """
        Initializes the DifferentiatingFactorAnalysis class with the specified parameters.

        Parameters:
                data (pd.DataFrame, optional): The input data as a pandas DataFrame.
                analyses (list of str, optional): List of analyses to perform. If None, a predefined list is used.
                labels_column (str, optional): Column name for labels. Default is "Diagnosis".
                reference_class (str, optional): The reference class for comparison. Default is "Control".
                sample_id_column (str, optional): Column name for sample IDs. Default is "Sample ID".
                outdir (str, optional): Output directory for saving results. Default is "output".
                mode (str, optional): Mode of operation, e.g., "classification". Default is "classification".
                feature_type (str, optional): Type of feature, e.g., "microRNA". Default is "microRNA".
                test_size (float, optional): Proportion of data to be used as the test set. Default is 0.2.
                lime_global_explanation_sample_num (int, optional): Number of samples to use for global LIME explanations. Default is 10.
                shap_model_finetune (bool, optional): Flag to indicate if model fine-tuning is applied for SHAP analysis. Default is False.
                lime_model_finetune (bool, optional): Flag to indicate if model fine-tuning is applied for LIME analysis. Default is False.
                n_folds (int, optional): Number of folds for cross-validation. Default is 5.
                scoring (str, optional): Scoring method to evaluate model performance. Default is "f1".
                top_features_to_plot (int, optional): Number of top features to display in feature importance plots. Default is 20.
                feature_importance_finetune (bool, optional): Flag to indicate if fine-tuning is applied for feature importance analysis. Default is True.


        Attributes:
        X (pd.DataFrame): Features extracted from the data.
        y (np.ndarray): Encoded labels.
        labels (pd.Series): Original labels.
        label_encodings (dict): Dictionary mapping encoded labels to original labels.
        class_names (list): List of unique class names.
        outdir (str): Output directory.
        mode (str): Mode of operation.
        feature_type (str): Type of feature.
        test_size (float): Test set size.
        SHAP_Analyzer: Placeholder for SHAP analysis object.
        LIME_Analyzer: Placeholder for LIME analysis object.
        ANOVA_Analyzer: Placeholder for ANOVA analysis object.
        feature_IMPORT_RF_Analyzer: Placeholder for Random Forest feature importance analysis object.
        feature_IMPORT_XGB_Analyzer: Placeholder for XGBoost feature importance analysis object.
        feature_IMPORT_PERMUTATION_Analyzer: Placeholder for permutation feature importance analysis object.
        t_TEST_Analyzer: Placeholder for t-test analysis object.
        analyses (list of str): List of analyses to perform.
        top_features (dict): Dictionary for storing top features for each analysis.
        """

        # store variables
        sns.set_theme(style="darkgrid")
        self.data = data
        self.X = data.drop([labels_column, sample_id_column], axis=1)
        self.y = LabelEncoder().fit_transform(data[labels_column])
        self.feature_map = {feature: f"Feature_{i}" for i, feature in enumerate(self.X.columns)}
        self.feature_map_reverse = {value:key for key,value in self.feature_map.items()}
        self.X.columns = list(map(lambda x: self.feature_map[x], self.X.columns)) # we map features as certaim may contain unsupported chars {/,\,%, etc}
        self.labels = data[labels_column]
        self.label_encodings = dict(zip(self.y, self.labels))
        self.class_names = list(self.labels.unique())
        self.outdir = outdir
        self.mode = mode
        self.reference_class = reference_class
        self.feature_type = feature_type
        self.test_size = test_size
        self.SHAP_Analyzer = None
        self.LIME_Analyzer = None
        self.ANOVA_Analyzer = None
        self.feature_IMPORT_RF_Analyzer = None
        self.feature_IMPORT_XGB_Analyzer = None
        self.feature_IMPORT_PERMUTATION_Analyzer = None
        self.t_TEST_Analyzer = None
        self.analyses = ["shap", "lime", "anova", "feature_importance", "t_test"] if analyses is None else analyses
        self.n_folds = n_folds
        self.top_features_to_plot = top_features_to_plot
        
        # select random samples from each class to compare
        np.random.seed(42) # Ensure reproducibility by setting a random seed
        
        # Filter indices for AD and Control samples
        class0_indices = data[data[labels_column] == self.class_names[0]].index
        class1_indices = data[data[labels_column] == self.class_names[1]].index
        
        # Randomly select one index from each group
        random_class0_index = np.random.choice(class0_indices)
        random_class1_index = np.random.choice(class1_indices)
        self.random_samples = {self.class_names[0]:random_class0_index , 
                               self.class_names[1]:random_class1_index }

        # Initialize Analyses
        if "shap" in self.analyses:
            self.SHAP_Analyzer = SHAP_Analysis(X=self.X, y=self.y, outdir = f"{self.outdir}/shap", random_samples = self.random_samples,feature_type = self.feature_type, feature_map_reverse = self.feature_map_reverse, 
                                               model_finetune=shap_model_finetune, fine_tune_cv_nfolds = n_folds, scoring = scoring, top_features_to_plot = top_features_to_plot)
        if "lime" in self.analyses:
            self.LIME_Analyzer = LIME_Analysis(X=self.X, y=self.y, outdir = f"{self.outdir}/lime",
                                               class_names=self.class_names, random_samples = self.random_samples,
                                               feature_type = self.feature_type,
                                               global_explanation_sample_num = lime_global_explanation_sample_num,
                                               feature_map_reverse = self.feature_map_reverse,
                                               model_finetune = lime_model_finetune,
                                               fine_tune_cv_nfolds = n_folds,
                                               scoring = scoring,
                                               top_features_to_plot = top_features_to_plot
                                              )
        if ("xgb_feature_importance" in self.analyses) or ("randomforest_feature_importance" in self.analyses) or ("permutation_feature_importance" in self.analyses):
            self.FeatureImportance_Analyzer = FeatureImportance_Analysis(X=self.X,y=self.y, test_size = test_size, feature_type = feature_type, top_features_to_plot = top_features_to_plot, feature_map_reverse = self.feature_map_reverse,
                                                                         model_finetune = feature_importance_finetune, fine_tune_cv_nfolds = n_folds, scoring = scoring, outdir = outdir)
        
        # Initialize top features json object for each analysis
        self.top_features = {}

        # Prepare output directories
        directories = [term for term in analyses if 'feature_importance' not in term]
        directories.append("feature_importance")
        for analysis in directories:
            for subdir in ["png", "pdf"]:
                os.makedirs(os.path.join(self.outdir, analysis, subdir), exist_ok=True)


    def perform_shap_analysis(self):
        """
        Executes all SHAP analysis functions from the custom SHAP_Analysis module.
        
        This method sequentially runs the following SHAP analysis functions:
        1. shapWaterFall: Creates waterfall plots for random samples from different classes.
        2. shapForce: Generates force plots for random samples from different classes.
        3. shapSummary: Produces a summary plot of SHAP values.
        4. shapHeatmap: Creates a heatmap of the top differentiating features.
        5. shapFeatureImportance: Computes and returns the top differentiating features based on SHAP values.
        
        The results are saved as PNG and PDF files in the specified output directory.
        """
        logging.info("Performing SHAP Analysis")
        length = 110
        print("="*length)
        print(" Starting SHAP Waterfall Analysis ")
        print("="*length)
        self.SHAP_Analyzer.shapWaterFall()
        
        print("="*length)
        print(" Starting SHAP Force Plot Analysis ")
        print("="*length)
        self.SHAP_Analyzer.shapForce()
        
        print("="*length)
        print(" Starting SHAP Summary Plot Analysis ")
        print("="*length)
        self.SHAP_Analyzer.shapSummary()
        
        print("="*length)
        print(" Starting SHAP Heatmap Analysis ")
        print("="*length)
        self.SHAP_Analyzer.shapHeatmap()

        print("="*length)
        print(" Starting Mean SHAP Plot Analysis ")
        print("="*length)
        self.SHAP_Analyzer.meanSHAP()
        
        print("="*length)
        print(" Starting SHAP Feature Importance Analysis ")
        print("="*length)
        self.top_features["shap"] = self.SHAP_Analyzer.shapFeatureImportance()
        print(sorted(self.top_features["shap"], key=self.top_features["shap"].get, reverse=True)[:self.top_features_to_plot])
        
        print("="*length)
        print(" SHAP Analysis Completed ")
        print("="*length)


    def perform_lime_analysis(self):
        """
        Executes all LIME analysis functions from the custom LIME_Analysis module.
        """
        logging.info("Performing LIME Analysis")
        length = 110
        print("="*length)
        print(" Starting LIME Per Sample Explanations")
        print("="*length)
        self.LIME_Analyzer.explain_samples()
        
        print("="*length)
        print(" Starting LIME Feature Importance Analysis ")
        print("="*length)
        self.top_features["lime"] = self.LIME_Analyzer.limeFeatureImportance()
        print(sorted(self.top_features["lime"], key=self.top_features["lime"].get, reverse=True)[:self.top_features_to_plot])

        print("="*length)
        print(" LIME Analysis Completed ")
        print("="*length)

    
    def perform_anova(self):
        """
        Perform ANOVA to identify significantly different features.
        """
        logging.info("Performing ANOVA Analysis")
        length = 110
        print("=" * length)
        print(" Starting ANOVA")
        print("=" * length)
        
        # Perform ANOVA
        f_statistic, p_values = f_classif(self.X, self.y)
        significant_features = pd.DataFrame({
            "Features": list(map(lambda x: self.feature_map_reverse[x], self.X.columns)), 
            "F-value": f_statistic, 
            "p-value": p_values
        })
        
        # Sort significant features by F-value
        logging.info("Computing ANOVA Features")
        significant_features = significant_features.sort_values(by="F-value", ascending=False).reset_index(drop=True)
        
        # Set colors for the bar plot (all bars blue)
        logging.info("Plotting ANOVA Features")
        top_anova_features = significant_features[significant_features["p-value"] < 0.05].head(20)
        colors = top_anova_features['F-value'].apply(lambda x: "blue")
        
        # Plot the F-values of the top differentiating features
        plt.figure(figsize=(4, 8))
        plt.barh(top_anova_features['Features'], top_anova_features['F-value'], color=colors)
        plt.xlabel('F-value', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f'ANOVA Feature Importance of Top Differentiating {self.feature_type}s', fontsize=15, loc='center', pad=15)
        plt.gca().invert_yaxis()
        
        # Save the plot
        logging.info("Saving Plots")
        plt.savefig(f'{self.outdir}/anova/png/anova_features_plot.png', bbox_inches='tight')
        plt.savefig(f'{self.outdir}/anova/pdf/anova_features_plot.pdf', bbox_inches='tight')
        plt.show()
    
        # Save top features
        self.top_features["anova"] = {significant_features.Features[i]: significant_features["F-value"][i] 
                    for i in range(len(significant_features))}
        
        # Print the top 25 features
        print(sorted(self.top_features["anova"], key=self.top_features["anova"].get, reverse=True)[:self.top_features_to_plot])
        print("=" * length)
        print(" ANOVA Analysis Completed ")
        print("=" * length)


    def perform_t_test(self):
        """
        Perform t-Test to identify significantly different features.
        """
        logging.info("Performing T-TEST")
        length = 110
        print("=" * length)
        print(" Starting t-Test")
        print("=" * length)
        
        # Sample dataset
        np.random.seed(0)
        
        # Separate the data by class
        class_0 = self.X.iloc[self.labels[self.labels == self.class_names[0]].dropna().index,:]
        class_1 = self.X.iloc[self.labels[self.labels == self.class_names[1]].dropna().index,:]
        
        # Perform t-tests and store p-values
        t_test_values = {"Features": [], "statistic": [], "pvalue": [], "df": []}
        for column in self.X.columns:
            output = ttest_ind(class_0[column], class_1[column])
            t_test_values["Features"].append(column)
            t_test_values["pvalue"].append(output.pvalue)
            t_test_values["statistic"].append(output.statistic)
            t_test_values["df"].append(output.df)
        
        # Convert p-values to a DataFrame for easy sorting and viewing
        logging.info("Computing Feature Importance by Statistical Significance")
        p_values_df = pd.DataFrame(t_test_values)
        p_values_df["Abs(statistic)"] = p_values_df["statistic"].abs()
        p_values_df = p_values_df.sort_values(by = "Abs(statistic)", ascending =False)

        # reverse map feature names
        p_values_df["Features"] = p_values_df["Features"].apply(lambda x: self.feature_map_reverse[x])

        # Set colors for the bar plot (all bars blue)
        logging.info("Plotting Top n Features")
        top_t_test_features = p_values_df[p_values_df.pvalue < 0.05].head(self.top_features_to_plot)
        colors = top_t_test_features['Abs(statistic)'].apply(lambda x: "blue")
        
        # Plot the F-values of the top differentiating features
        plt.figure(figsize=(4, 8))
        plt.barh(top_t_test_features['Features'], top_t_test_features['Abs(statistic)'], color=colors)
        plt.xlabel('Abs(statistic)', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f't-Test Feature Importance of Top Differentiating {self.feature_type}s', 
                  fontsize=15, loc='center', pad=15)
        plt.gca().invert_yaxis()
        
        # Save the plot
        logging.info("Saving Plots")
        plt.savefig(f'{self.outdir}/t_test/png/t_test_features_plot.png', bbox_inches='tight')
        plt.savefig(f'{self.outdir}/t_test/pdf/t_test_features_plot.pdf', bbox_inches='tight')
        plt.show()
        
        # Save top features
        self.top_features["t_test"] = {p_values_df.Features[i]: p_values_df["Abs(statistic)"][i] 
                    for i in range(len(p_values_df))}
        
        # Print the top 25 features
        print(sorted(self.top_features["t_test"], key=self.top_features["t_test"].get, reverse=True)[:self.top_features_to_plot])
        print("=" * length)
        print(" t-Test Analysis Completed ")
        print("=" * length)

    
    def perform_randomforest_feature_importance_analysis(self):
        """
        Executes feature importance analysis using a Random Forest classifier and prints the top N features based on their importance scores. It logs the process and displays a formatted message indicating the start and completion of the analysis.
        """
        logging.info("Performing Random Forest Feature Importance Analysis")
        
        length = 110
        print("=" * length)
        print(" Starting Feature Importance Analysis with Random Forest")
        print("=" * length)

        # 
        self.top_features["randomforest_feature_importance"] = self.FeatureImportance_Analyzer.RandomForestFeatureImportance()
        
        # Print the top n features
        print(sorted(self.top_features["randomforest_feature_importance"], key=self.top_features["randomforest_feature_importance"].get, reverse=True)[:self.top_features_to_plot])
        print("=" * length)
        print(" Random Forest Feature Importance Analysis Completed ")
        print("=" * length)


    def perform_xgb_feature_importance_analysis(self):
        """
        Executes feature importance analysis using an XGBoost classifier and prints the top N features based on their importance scores. It logs the process and displays a formatted message indicating the start and completion of the analysis.
        """
        logging.info("Performing XGBoost Feature Importance Analysis")
        length = 110
        print("=" * length)
        print(" Starting Feature Importance Analysis with XGBoost")
        print("=" * length)

        # 
        self.top_features["xgb_feature_importance"] = self.FeatureImportance_Analyzer.XGBoostFeatureImportance()
        
        # Print the top 25 features
        print(sorted(self.top_features["xgb_feature_importance"], key=self.top_features["xgb_feature_importance"].get, reverse=True)[:self.top_features_to_plot])
        print("=" * length)
        print(" XGBoost Feature Importance Analysis Completed ")
        print("=" * length)


    def perform_permutation_feature_importance_analysis(self):
        """
        Executes feature importance analysis using a Random Forest classifier, performs permutation feature importance analysis and prints the top N features based on their importance scores. It logs the process and displays a formatted message indicating the start and completion of the analysis.
        """
        logging.info("Performing Permutation Feature Importance Analysis")
        length = 110
        print("=" * length)
        print(" Starting Feature Importance Analysis with Permutation Method")
        print("=" * length)

        # 
        self.top_features["permutation_feature_importance"] = self.FeatureImportance_Analyzer.PermutationFeatureImportance()
        
        # Print the top 25 features
        print(sorted(self.top_features["permutation_feature_importance"], key=self.top_features["permutation_feature_importance"].get, reverse=True)[:self.top_features_to_plot])
        print("=" * length)
        print(" Permutation Feature Importance Analysis Completed ")
        print("=" * length)


    def run_all_analyses(self):
        """
        Executes all analyses specified in the 'analyses' attribute of the class.
        This method sequentially runs SHAP, LIME, ANOVA, t-test, and Feature Importance analyses.
        
        The results from each analysis are saved in the designated output directories, 
        and the top differentiating features are stored in the 'top_features' attribute.
        """

        logging.info("RUNNING ALL ANALYSES")
        length = 110
        print("="*length)
        print(" Starting All Analyses ")
        print("="*length)
        
        if "shap" in self.analyses:
            self.perform_shap_analysis()
        
        if "lime" in self.analyses:
            self.perform_lime_analysis()
        
        if "anova" in self.analyses:
            self.perform_anova()
        
        if "t_test" in self.analyses:
            self.perform_t_test()
        
        if "randomforest_feature_importance" in self.analyses:
            self.perform_randomforest_feature_importance_analysis()

        if "xgb_feature_importance" in self.analyses:
            self.perform_xgb_feature_importance_analysis()

        if "permutation_feature_importance" in self.analyses:
            self.perform_permutation_feature_importance_analysis()

        # save self.top_features convert all values to floats
        for a in self.top_features.keys():
            for feature in self.top_features[a].keys():
                self.top_features[a][feature] = float(self.top_features[a][feature])

        logging.info("Saving Feature Importances")
        save_json(f"{self.outdir}/feature_importances.json", self.top_features)
        print("="*length)
        print(" All Analyses Completed ")
        print("="*length)

        logging.info("ALL ANALYSES COMPLETED!!!")
