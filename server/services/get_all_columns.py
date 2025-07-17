import sys
import pandas as pd

# Function to print all column names from a CSV file
def get_all_columns(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
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