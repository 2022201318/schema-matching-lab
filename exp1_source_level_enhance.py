import os
import json
import pandas as pd
import requests

def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def call_llm_english(prompt, config):
    """Interact with LLM using English for both prompt and response"""
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config['model_name'],
        "messages": [
            {
                "role": "system", 
                "content": "You are a professional data scientist. Provide descriptions strictly in English."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    try:
        response = requests.post(config['api_url'], json=payload, headers=headers, timeout=60)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"LLM Error: {e}"

def get_full_structure(source_path):
    """Collect all tables and their columns from the source directory"""
    tables_info = []
    # Filter only CSV files
    csv_files = [f for f in os.listdir(source_path) if f.endswith('.csv')]
    for csv in csv_files:
        try:
            # nrows=0 is efficient for getting column names only
            df = pd.read_csv(os.path.join(source_path, csv), nrows=0)
            tables_info.append({
                "table_name": csv,
                "columns": df.columns.tolist()
            })
        except:
            continue
    return tables_info

def main():
    config = load_config()
    data_root = config['data_path']

    # Process each source folder from 1 to 18
    for i in range(1, 19):
        source_id = f"source{i}"
        source_dir = os.path.join(data_root, source_id)
        
        if not os.path.exists(source_dir):
            continue

        print(f"Processing {source_id}...")

        # 1. Load existing s_metadata.json for context/description
        s_meta_path = os.path.join(source_dir, "s_metadata.json")
        existing_meta = {}
        if os.path.exists(s_meta_path):
            with open(s_meta_path, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)

        description = existing_meta.get("description", "")

        # 2. Extract full schema inventory (Part 2 of requirements)
        structure = get_full_structure(source_dir)

        # 3. Handle Description (Part 1 of requirements)
        # If original description is missing or invalid, invoke LLM
        if not description or len(str(description).strip()) < 10:
            print(f"  -> Generating English semantic description via LLM...")
            
            prompt = (
                f"Task: Based on the provided metadata and table structures, generate a concise professional description of this data source in English (under 100 words).\n"
                f"Source Identity: {source_id}\n"
                f"Reference Metadata: {json.dumps(existing_meta)}\n"
                f"Table Inventory: {json.dumps(structure)}\n"
                f"Constraint: Return ONLY the English text description."
            )
            description = call_llm_english(prompt, config)
        else:
            print(f"  -> Using validated description from s_metadata.")

        # 4. Construct the Semantic View object
        semantic_view_result = {
            "description": description,
            "schema_inventory": structure
        }

        # 5. Save with the new naming convention: sourceX_semantic_view.json
        output_filename = f"{source_id}_semantic_view.json"
        view_path = os.path.join(source_dir, output_filename)
        
        with open(view_path, 'w', encoding='utf-8') as f:
            json.dump(semantic_view_result, f, indent=4, ensure_ascii=False)
            
        print(f"  -> Successfully saved: {output_filename}")

if __name__ == "__main__":
    main()