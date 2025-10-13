import sys, os, pandas as pd, json

def merge_files(file_paths, key_column='Sample ID'):
    dfs = []
    
    for i, path in enumerate(file_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        df = pd.read_csv(path)
        
        # If key_column not found, create one with row index
        if key_column not in df.columns:
            df.insert(0, key_column, range(1, len(df) + 1))
        
        # Rename overlapping columns (except key)
        if i > 0:
            df = df.rename(columns={col: f"{col}_file{i+1}" for col in df.columns if col != key_column})
        
        dfs.append(df)
    
    # Merge iteratively using outer join on key
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=key_column, how='outer')

    # 🔹 Clean up column types
    # Convert all non-key columns that look numeric into float
    for col in merged_df.columns:
        if col != key_column:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
    
    # 🔹 Replace any infinities or weird symbols with NaN
    merged_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

    # 🔹 Optional: Fill NaNs (to prevent later ML crashes)
    merged_df.fillna(0, inplace=True)

    # Save merged CSV
    out_dir = os.path.join('results', 'merged_files')
    os.makedirs(out_dir, exist_ok=True)
    merged_file_path = os.path.join(out_dir, 'merged.csv')
    merged_df.to_csv(merged_file_path, index=False)
    
    return {
        'mergedFilePath': merged_file_path,
        'columns': merged_df.columns.tolist()
    }

if __name__ == "__main__":
    file_paths = sys.argv[1:]
    key_column = 'Sample ID'  # default
    try:
        result = merge_files(file_paths, key_column)
        print(json.dumps(result))
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)
