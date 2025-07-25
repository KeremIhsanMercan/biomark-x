import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Get parameters from command line
data_path = sys.argv[1]  # File path
feature_count = int(sys.argv[2]) if len(sys.argv) > 2 else 20  # Number of miRNAs to display

# Optional class pair and csv path parameters
class_pair = sys.argv[3] if len(sys.argv) > 3 else None  # Class pair (optional)
csv_path = sys.argv[4] if len(sys.argv) > 4 else None  # CSV file path (optional)

# Extract analysis name from data_path (remove 'uploads/' and .csv extension)
# Example: "uploads/GSE120584_serum_norm.csv" -> "GSE120584_serum_norm"
file_name = os.path.basename(data_path).split('.')[0]

# Use custom csv_path if specified, otherwise use default path
if csv_path:
    # Use custom CSV file
    ranked_features_path = csv_path
else:
    # Default path
    ranked_features_path = os.path.join("results", file_name, "ranked_features_df.csv")

# Read the CSV file
df = pd.read_csv(ranked_features_path)

df.drop(columns=["overall score"], inplace=True)

# Calculate mean rank
df["Mean Rank"] = df.iloc[:, 1:].mean(axis=1)

# Round mean rank values to integer
df["Mean Rank"] = df["Mean Rank"].round().astype(int)

# Select and sort the top N (feature_count) biomarkers with the smallest (most effective) mean rank
df_top = df.nsmallest(feature_count, "Mean Rank")

# Visualization settings - adjust for larger size
column_count = df.shape[1] - 1
# Set minimum width to 12 inches and scale by number of columns
min_width = max(12, column_count * 1.5)  
# Increase height, especially for more rows
height = min(15, feature_count/3 + 5)  
plt.figure(figsize=(min_width, height))

# Custom annotation function for heatmap
# Annotates each cell with integer value and increases font size

def annotate_heatmap(data, annot, fmt=".2f", **textkw):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j == data.shape[1] - 1:  # Last column (Mean Rank)
                text = format(int(data[i, j]), "d")  # Format as integer
            else:
                text = format(int(data[i, j]), "d")
            # Increase font size for cell text
            annot[i * data.shape[1] + j].set_text(text)
            annot[i * data.shape[1] + j].set_fontsize(20)  # Increased cell font size

# Find the appropriate feature column name (feature type)
feature_column = df_top.columns[0]  # First column is feature type (microRNA, gene, etc)

# Use a more readable and high-contrast color palette ("magma")
ax = sns.heatmap(
    df_top.set_index(feature_column),  # Do not include last column
    annot=True,  # Annotate cells with numbers
    cmap="magma_r",  # Reversed magma colormap for dark background, light text
    fmt="",  # Custom format
    linewidths=0.7,  # Add more visible lines between cells
    linecolor="gray",  # Gray lines between cells
    square=False,  # Use rectangular cells
    annot_kws={"size": 14}  # Increase annotation font size
)

# Output directory
if class_pair:
    # If class pair is specified, create directory accordingly
    outdir = os.path.join("results", file_name, "summaryStatisticalMethods", class_pair)
else:
    # Default directory structure
    outdir = os.path.join("results", file_name, "summaryStatisticalMethods")

# Create folders if they do not exist
os.makedirs(os.path.join(outdir, "png"), exist_ok=True)
os.makedirs(os.path.join(outdir, "pdf"), exist_ok=True)

# Apply custom annotation to heatmap
annotate_heatmap(ax.collections[0].get_array(), ax.texts)

# Adjust title font size based on length
# Use a more generic heading and format class pair with 'vs' for readability
if class_pair:
    # Replace underscores with ' vs ' for display purposes
    class_pair_display = class_pair.replace('_', ' vs ')
    title_text = (
        f"Top {feature_count} Biomarkers and Their Rankings by Statistical Methods\n"
        f"for Class Pair: {class_pair_display}"
    )
else:
    title_text = f"Top {feature_count} Biomarkers and Their Rankings by Statistical Methods"

# Set font size
fontsize = 20 if column_count >= 5 else 18

# Set title and labels with larger font size
plt.title(title_text, fontsize=fontsize, fontweight="bold", pad=20)
plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.xlabel("Statistical Methods", fontsize=24)
plt.ylabel(feature_column, fontsize=24)

# Expand and adjust plot area
plt.subplots_adjust(top=0.92, bottom=0.15, left=0.20, right=0.95)

# Apply tight_layout for proper content placement
plt.tight_layout()

# Save files with higher resolution
png_output_path = os.path.join(outdir, "png", "summary_of_statistical_methods_plot.png")
plt.savefig(png_output_path, dpi=400, bbox_inches='tight')

# Print relative path (to be used by server.js)
print(png_output_path)

# Save as PDF
pdf_output_path = os.path.join(outdir, "pdf", "summary_of_statistical_methods_plot.pdf")
plt.savefig(pdf_output_path, dpi=400, bbox_inches='tight')
