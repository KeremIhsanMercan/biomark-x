import pandas as pd
import os
from modules.logger import logging

def feature_rank(top_features: dict = None, num_top_features: int = 20, feature_type: str = None, outdir: str = "outputs"):
    """
    Ranks features based on their importance scores from different methods (e.g., SHAP, ANOVA), aggregates 
    the scores, and selects the top N features. The ranked features are saved to a CSV file.

    Args:
        top_features (dict): A dictionary containing feature importance scores from class pairs and different methods.
        num_top_features (int): The number of top features to select and return.
        feature_type (str): The type of features being ranked (e.g., 'microRNA').
        outdir (str): The output directory where the ranked features CSV will be saved.

    Returns:
        list: A list of the top N ranked features.
    """
    
    # Prepare ranking data 
    def rank_dict(d):
        # Check if d is a dictionary
        if not isinstance(d, dict):
            logging.error(f"Expected a dictionary but got {type(d).__name__} instead. Value: {d}")
            # Return empty dict to avoid breaking the process, but log the error
            return {}
            
        # Create a list of keys sorted by their values in descending order
        sorted_keys = sorted(d, key=d.get, reverse=True)
        # Create a dictionary of ranks
        ranked_dict = {key: rank + 1 for rank, key in enumerate(sorted_keys)}
        return ranked_dict

    logging.info("Performing Feature Selection by Feature Ranking")
    
    # First, create a main list for all class pairs
    all_top_features = {}
    
    # Process each class pair
    for class_pair, analysis_data in top_features.items():
        # Data type check
        if not isinstance(analysis_data, dict):
            logging.error(f"Class pair '{class_pair}': Expected dictionary but got {type(analysis_data).__name__}")
            continue  # Skip this class pair
            
        # Apply ranking to each sub-dictionary
        ranked_data = {}
        for outer_key, outer_dict in analysis_data.items():
            # Data type check
            if not isinstance(outer_dict, dict):
                logging.error(f"Class pair '{class_pair}', analysis '{outer_key}': Expected dictionary but got {type(outer_dict).__name__}")
                continue  # Skip this analysis
                
            ranked_data[outer_key] = rank_dict(outer_dict)
            
        # Skip this class pair if no valid analysis
        if not ranked_data:
            logging.warning(f"No valid analysis data found for class pair '{class_pair}'")
            continue
            
        ranked_data_df = pd.DataFrame(ranked_data)
        
        # ANOVA features with many NaN values were filtered off due to high p-values, so we remove them
        ranked_data_df = ranked_data_df.dropna().reset_index().rename(columns={"index": feature_type})
        
        # Aggregate scores and sort by minimum
        ranked_data_df["overall score"] = ranked_data_df.iloc[:, 1:].sum(axis=1)
        ranked_data_df.sort_values(by="overall score", ascending=True, inplace=True)
        
        # Create a folder for each class pair
        pair_dir = os.path.join(outdir, "feature_ranking", class_pair)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Save a separate CSV file for each class pair
        ranked_data_df.to_csv(f"{pair_dir}/ranked_features_df.csv", index=False)
        
        # Additionally, copy a classic ranked_features_df.csv file to the main directory
        # (For the first or last processed class pair, for backward compatibility)
        ranked_data_df.to_csv(f"{outdir}/ranked_features_df.csv", index=False)
        
        # Get top features for this class pair
        top_n_features = ranked_data_df.head(num_top_features)[feature_type].to_list()
        all_top_features[class_pair] = top_n_features
    
    # For backward compatibility, return only the first or single class pair (to not break old code)
    if len(all_top_features) > 0:
        first_class_pair = list(all_top_features.keys())[0]
        return all_top_features[first_class_pair]
    
    return []
