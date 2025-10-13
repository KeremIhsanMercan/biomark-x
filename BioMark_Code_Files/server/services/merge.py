import sys, os, pandas as pd, json

def merge_files(file_paths, key_column='Sample ID'):
    dfs = []
    
    for i, path in enumerate(file_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        df = pd.read_csv(path, low_memory=False)
        
        # Ensure key column exists
        if key_column not in df.columns:
            df.insert(0, key_column, range(1, len(df) + 1))
        
        # Rename overlapping columns (except key)
        if i > 0:
            df = df.rename(columns={col: f"{col}_file{i+1}" for col in df.columns if col != key_column})
        
        dfs.append(df)
    
    # Merge all files using outer join on the key
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=key_column, how='outer')
    
    # Combine columns with same base name (e.g., "Age", "Age_file2", ...)
    base_cols = set()
    for col in merged_df.columns:
        if '_file' in col:
            base_cols.add(col.split('_file')[0])
    
    for base in base_cols:
        related = [c for c in merged_df.columns if c == base or c.startswith(f"{base}_file")]
        if len(related) > 1:
            merged_df[base] = merged_df[related].bfill(axis=1).iloc[:, 0]
            for c in related:
                if c != base:
                    merged_df.drop(columns=c, inplace=True)

    # Separate numeric and categorical columns
    numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    object_cols = merged_df.select_dtypes(include=['object']).columns.tolist()

    # Convert numeric-like strings to numbers safely
    for col in merged_df.columns:
        if col != key_column:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='ignore')

    # Re-detect numeric and categorical columns after conversion
    numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [c for c in merged_df.columns if c not in numeric_cols + [key_column]]

    # Fill numeric NaNs with median values
    for col in numeric_cols:
        if merged_df[col].isna().any():
            merged_df[col].fillna(merged_df[col].median(), inplace=True)

    # Fill categorical NaNs with "Unknown" (optional)
    for col in categorical_cols:
        merged_df[col].fillna("Unknown", inplace=True)

    # Replace infinite values
    merged_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

    # Save merged file
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
