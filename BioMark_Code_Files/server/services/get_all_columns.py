import pandas as pd
# Add modules path and import helper
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.utils import load_table

# Function to print all column names from a CSV file
def get_all_columns(file_path):
    try:
        # Read the CSV file
        df = load_table(file_path)
        
        # Get all column names
        columns = df.columns.tolist()
        
        # Print each column name on a new line
        for column in columns:
            print(column)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

# Main script execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_all_columns.py <file_path>", file=sys.stderr)
        sys.exit(1)
        
    file_path = sys.argv[1]
    get_all_columns(file_path) 