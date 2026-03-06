import os
import json
import pandas as pd
import requests

def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def call_llm(prompt, model_name, config):
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a professional data analyst. Respond strictly in English in JSON format."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"} # ???? JSON ????
    }
    try:
        response = requests.post(config['api_url'], json=payload, headers=headers, timeout=90)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return None

def process_batch(batch_data, table_name, model_name, config, is_strong=False):
    """
    batch_data: list of dicts {'column_name': ..., 'samples': ...}
    """
    if not is_strong:
        # --- 14B (SLM) ??????????????? ---
        instruction = (
            "You are a professional data analyst. Your task is to infer the semantic meaning of database columns.\n"
            "CONTEXT RULES:\n"
            "1. The provided samples are ONLY a small non-empty subset (3 rows) of the data, not the complete dataset.\n"
            "2. Column names may contain cryptic abbreviations, technical jargon, or even spelling errors.\n"
            "3. If the column name and samples are too ambiguous to define accurately, output 'UNCERTAIN'."
        )
    else:
        # --- 72B (LLM) ???????????? UNCERTAIN ---
        instruction = (
            "You are a Senior Data Scientist and Domain Expert. Your task is to perform deep semantic reasoning.\n"
            "REASONING RULES:\n"
            "1. These columns were flagged as difficult. Analyze them by decomposing potential abbreviations (e.g., 'pt' for patient, 'tst' for test) and correcting possible typos.\n"
            "2. The 'samples' provided are just a 3-row glimpse; use them to validate your hypothesis about the column name.\n"
            "3. STRICT CONSTRAINT: Do NOT output 'UNCERTAIN'. You must provide your most professional and likely inference for every column based on the table context."
        )
    
    prompt = (
        f"{instruction}\n\n"
        f"Table Name: {table_name}\n"
        f"Target Columns (Name & Samples): {json.dumps(batch_data)}\n\n"
        f"OUTPUT FORMAT:\n"
        f"Return a JSON object where keys are column names and values are concise English descriptions (max 25 words)."
    )
    
    response_text = call_llm(prompt, model_name, config)
    try:
        return json.loads(response_text) if response_text else {}
    except:
        # ???????????????????????????????????
        print(f"Error: Failed to parse JSON from {model_name}")
        return {}

def main():
    config = load_config()
    MODEL_14B = "qwen2.5-14b-instruct"
    MODEL_72B = "qwen2.5-72b-instruct"
    BATCH_SIZE = 10

    for i in range(1, 19):
        source_id = f"source{i}"
        source_dir = os.path.join(config['data_path'], source_id)
        if not os.path.exists(source_dir): continue

        print(f"=== [Batched Column Level] Enhancing {source_id} ===")
        csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

        for csv_name in csv_files:
            file_path = os.path.join(source_dir, csv_name)
            base_name = os.path.splitext(csv_name)[0]
            
            try:
                df = pd.read_csv(file_path)
            except: continue

            final_columns_view = {}
            all_cols = df.columns.tolist()
            
            # --- ???? (Batching) ---
            for j in range(0, len(all_cols), BATCH_SIZE):
                current_batch_cols = all_cols[j:j + BATCH_SIZE]
                batch_input = []
                for col in current_batch_cols:
                    samples = df[col].dropna().head(3).tolist()
                    batch_input.append({"column_name": col, "samples": [str(s) for s in samples]})

                print(f"  Table: {csv_name} | Processing batch {j//BATCH_SIZE + 1} with 14B...")
                
                # 1. 14B ??
                batch_results = process_batch(batch_input, csv_name, MODEL_14B, config)
                
                # 2. ????? UNCERTAIN?????? 72B
                uncertain_batch = []
                for col_name, desc in batch_results.items():
                    if "UNCERTAIN" in desc.upper() and len(desc) < 15:
                        # ????????
                        orig_data = next(item for item in batch_input if item["column_name"] == col_name)
                        uncertain_batch.append(orig_data)
                    else:
                        final_columns_view[col_name] = desc

                if uncertain_batch:
                    print(f"    -> Found {len(uncertain_batch)} uncertain columns. Escalating to 72B...")
                    strong_results = process_batch(uncertain_batch, csv_name, MODEL_72B, config, is_strong=True)
                    final_columns_view.update(strong_results)

            # ??
            output_name = f"{base_name}_columns_semantic_view.json"
            with open(os.path.join(source_dir, output_name), 'w', encoding='utf-8') as f:
                json.dump(final_columns_view, f, indent=4, ensure_ascii=False)
            
            print(f"  Successfully saved: {output_name}")

if __name__ == "__main__":
    main()