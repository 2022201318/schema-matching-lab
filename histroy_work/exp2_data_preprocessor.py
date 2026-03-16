import os
import json
import pandas as pd

def prepare_rich_data(base_path):
    all_data_for_cognee = []

    for i in range(1, 19):
        source_id = f"source{i}"
        source_dir = os.path.join(base_path, source_id)
        if not os.path.exists(source_dir): continue

        print(f"Processing {source_id}...")

        # 1. ?????? (Source Level)
        source_view_path = os.path.join(source_dir, f"{source_id}_semantic_view.json")
        if os.path.exists(source_view_path):
            with open(source_view_path, 'r', encoding='utf-8') as f:
                source_info = json.load(f)
                all_data_for_cognee.append({
                    "layer": "source",
                    "id": source_id,
                    "content": f"Data Source {source_id}: {source_info.get('description', '')}",
                    "metadata": {"type": "source", "source_id": source_id}
                })

        # 2. ? CSV ??????????
        files = os.listdir(source_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        for csv_name in csv_files:
            table_name = os.path.splitext(csv_name)[0]
            csv_path = os.path.join(source_dir, csv_name)
            
            # --- ??????????? Type ? Samples ---
            try:
                df = pd.read_csv(csv_path, nrows=5) # ?????????
                col_types = df.dtypes.apply(lambda x: x.name).to_dict()
            except:
                col_types = {}

            # --- A. ?????? (Table Level) ---
            # ???????_semantic_view.json
            table_view_path = os.path.join(source_dir, f"{table_name}_semantic_view.json")
            table_desc = "No description available."
            if os.path.exists(table_view_path):
                with open(table_view_path, 'r', encoding='utf-8') as f:
                    table_desc = json.load(f).get("description", "")
                    all_data_for_cognee.append({
                        "layer": "table",
                        "id": f"{source_id}_{table_name}",
                        "content": f"Table '{table_name}' in {source_id}. Description: {table_desc}",
                        "metadata": {"type": "table", "source_id": source_id, "table_name": table_name}
                    })

            # --- B. ???????? (Column Level) ---
            # ???????_columns_semantic_view.json
            col_view_path = os.path.join(source_dir, f"{table_name}_columns_semantic_view.json")
            if os.path.exists(col_view_path):
                with open(col_view_path, 'r', encoding='utf-8') as f:
                    cols_semantic = json.load(f)
                    
                    for col_name, col_desc in cols_semantic.items():
                        # ??????? (????? df ??)
                        samples = []
                        if col_name in df.columns:
                            samples = df[col_name].dropna().head(3).tolist()
                        
                        data_type = col_types.get(col_name, "unknown")

                        # ??????????????+??+??+??+??????
                        all_data_for_cognee.append({
                            "layer": "column",
                            "id": f"{source_id}_{table_name}_{col_name}",
                            "content": (
                                f"Column '{col_name}' (Type: {data_type}) belongs to table '{table_name}' in {source_id}. "
                                f"Semantic Meaning: {col_desc}. "
                                f"Sample Values: {samples}. "
                                f"Table Context: This table is about {table_desc[:200]}..." # ????????
                            ),
                            "metadata": {
                                "type": "column",
                                "source_id": source_id,
                                "table_name": table_name,
                                "column_name": col_name
                            }
                        })
    
    # ????
    with open("ready_for_cognee.json", "w", encoding='utf-8') as f:
        json.dump(all_data_for_cognee, f, indent=4, ensure_ascii=False)
    print(f"\n[Success] Pre-processing complete. Saved {len(all_data_for_cognee)} entries to 'ready_for_cognee.json'.")

if __name__ == "__main__":
    # ??? data ???????
    prepare_rich_data("data")