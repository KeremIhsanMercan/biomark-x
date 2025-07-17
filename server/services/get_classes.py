import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Logging settings
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'get_classes.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load data and print unique values of the specified column
def load_data(data_path, column_name, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(df[column_name].unique().tolist())
    return df

# Visualize the distribution of a categorical column in the data
def visualize_diagnosis_distribution(df, column_name, outdir):
    # Plot the distribution of the Diagnosis Group
    plt.figure(figsize=(10,6))  # Set the figure size
    ax = sns.countplot(x=column_name, data=df, hue=column_name, saturation=0.95, legend=False)
    
    # Add bar labels for all bars
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', color='black', size=10)
        
    image_path = os.path.join(outdir, f'{column_name}_distribution.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(image_path)

# Main script execution
if __name__ == "__main__":
    
    # Get parameters
    data_path = sys.argv[1]
    column_name = sys.argv[2]

    # Output directory
    base_name = os.path.basename(data_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    outdir = os.path.join("results", file_name_without_ext)
    
    # Load data
    df = load_data(data_path, column_name, outdir)
    
    # Visualize data
    visualize_diagnosis_distribution(df, column_name, outdir)
