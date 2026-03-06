import os
import json
import pandas as pd
import requests

def load_config(config_path='config.json'):
    """Read API and Path configurations"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def call_qwen_api(prompt, config):
    """Call Qwen 2.5 API for Table-level English description"""
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config['model_name'],
        "messages": [
            {
                "role": "system", 
                "content": "You are a professional data cataloger. Describe the database table strictly in English based on its schema and sample values."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    try:
        response = requests.post(config['api_url'], json=payload, headers=headers, timeout=60)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"LLM Error: {e}"

def get_table_metadata(file_path):
    """Extract columns, types, and 3 non-empty samples per column"""
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        column_items = []

        for col in df.columns:
            # Drop NaN and get top 3 samples
            non_empty_samples = df[col].dropna().head(3).tolist()
            # Convert to string for JSON safety
            samples = [str(s) for s in non_empty_samples]
            
            column_items.append({
                "column_name": col,
                "data_type": str(df[col].dtype),
                "samples": samples
            })
        return column_items
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    config = load_config()
    data_root = config['data_path']

    # Iterate through each source folder
    for i in range(1, 19):
        source_id = f"source{i}"
        source_dir = os.path.join(data_root, source_id)
        
        if not os.path.exists(source_dir):
            continue

        print(f"--- Processing Table-level Enhancement for {source_id} ---")
        
        # Identify all CSV tables in the current source directory
        csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        
        for csv_name in csv_files:
            file_path = os.path.join(source_dir, csv_name)
            base_name = os.path.splitext(csv_name)[0]
            
            # 1. Extract Column Metadata (Names, Types, Samples)
            col_metadata = get_table_metadata(file_path)
            if col_metadata is None: continue

            # 2. Generate Table Description via LLM (English Only)
            print(f"  Generating description for table: {csv_name}")
            prompt = (
                f"Task: Write a technical English description (max 80 words) for this table.\n"
                f"Table Filename: {csv_name}\n"
                f"Schema & Sample Data: {json.dumps(col_metadata, indent=2)}\n"
                f"Output: Only the description text."
            )
            table_desc = call_qwen_api(prompt, config)

            # 3. Construct Table-level Semantic View
            table_view = {
                "table_name": csv_name,
                "description": table_desc,
                "columns": col_metadata
            }

            # 4. Save to [TableName]_semantic_view.json in the same directory
            output_name = f"{base_name}_semantic_view.json"
            output_path = os.path.join(source_dir, output_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(table_view, f, indent=4, ensure_ascii=False)
            
            print(f"  Successfully saved: {output_name}")

if __name__ == "__main__":
    main()