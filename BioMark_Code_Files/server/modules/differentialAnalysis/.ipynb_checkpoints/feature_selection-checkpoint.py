import pandas as pd
from modules.logger import logging

def feature_rank(top_features: dict = None, num_top_features: int = 20, feature_type: str = None, outdir: str = "outputs"):
    """
    Ranks features based on their importance scores from different methods (e.g., SHAP, ANOVA), aggregates 
    the scores, and selects the top N features. The ranked features are saved to a CSV file.

    Args:
        top_features (dict): A dictionary containing feature importance scores from different methods.
        num_top_features (int): The number of top features to select and return.
        feature_type (str): The type of features being ranked (e.g., 'microRNA').
        outdir (str): The output directory where the ranked features CSV will be saved.

    Returns:
        list: A list of the top N ranked features.
    """
    
    # Prepare ranking data 
    def rank_dict(d):
        # Create a list of keys sorted by their values in descending order
        sorted_keys = sorted(d, key=d.get, reverse=True)
        # Create a dictionary of ranks
        ranked_dict = {key: rank + 1 for rank, key in enumerate(sorted_keys)}
        return ranked_dict

    logging.info("Performing Feature Selection by Feature Ranking")
    # Apply ranking to each sub-dictionary
    ranked_data = {outer_key: rank_dict(outer_dict) for outer_key, outer_dict in top_features.items()}
    ranked_data_df = pd.DataFrame(ranked_data)
    
    # ANOVA features with many NaN values were filtered off due to high p-values, so we remove them
    ranked_data_df = ranked_data_df.dropna().reset_index().rename(columns={"index": feature_type})
    
    # Aggregate scores and sort by minimum
    ranked_data_df["overall score"] = ranked_data_df.iloc[:, 1:].sum(axis=1)
    ranked_data_df.sort_values(by="overall score", ascending=True, inplace=True)

    ranked_data_df.to_csv(f"{outdir}/ranked_features_df.csv", index=False)
    
    # Get top features
    top_n_features = ranked_data_df.head(num_top_features)[feature_type].to_list()

    return top_n_features
