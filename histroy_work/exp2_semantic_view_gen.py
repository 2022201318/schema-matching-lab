import os
import json

def generate_perfect_stratified_kb(base_path):
    source_entries = []
    table_entries = []
    column_entries = []
    
    for i in range(1, 19):
        source_id = f"source{i}"
        source_dir = os.path.join(base_path, source_id)
        if not os.path.exists(source_dir): continue

        # --- Phase 1: Source Level ---
        sv_path = os.path.join(source_dir, f"{source_id}_semantic_view.json")
        source_desc = ""
        if os.path.exists(sv_path):
            with open(sv_path, 'r', encoding='utf-8') as f:
                sv = json.load(f)
                source_desc = sv.get("description", "")
                inventory_text = ""
                for item in sv.get("schema_inventory", []):
                    inventory_text += f"- Table: {item.get('table_name')} (Columns: {', '.join(item.get('columns', []))})\n"
                
                source_entries.append(
                    f"DATA SOURCE: {source_id}\n"
                    f"BUSINESS SCOPE: {source_desc}\n"
                    f"ASSET MANIFEST:\n{inventory_text.strip()}"
                )

        # --- Phase 2 & 3: Table and Column Level ---
        # ???????????????????? _columns_semantic_view.json ?????
        all_files = os.listdir(source_dir)
        table_files = [f for f in all_files if f.endswith("_semantic_view.json") 
                       and not f.startswith(f"{source_id}_") 
                       and not f.endswith("_columns_semantic_view.json")]

        for filename in table_files:
            table_name = filename.replace("_semantic_view.json", "")
            tv_path = os.path.join(source_dir, filename)
            cv_path = os.path.join(source_dir, f"{table_name}_columns_semantic_view.json")
            
            # Load Table metadata (contains samples and types)
            with open(tv_path, 'r', encoding='utf-8') as f:
                tv_data = json.load(f)
            
            # Load Column semantics (the dictionary with descriptions)
            column_semantics = {}
            if os.path.exists(cv_path):
                with open(cv_path, 'r', encoding='utf-8') as f:
                    column_semantics = json.load(f)

            # --- Level 2: Table Narrative (Including Sample Data) ---
            t_desc = tv_data.get("description", "")
            col_detail_list = []
            col_names_only = []
            
            for col in tv_data.get("columns", []):
                c_name = col.get("column_name")
                c_samples = col.get("samples", [])
                col_names_only.append(c_name)
                # ????????????????
                col_detail_list.append(f"{c_name} (Samples: {c_samples})")

            table_entries.append(
                f"TABLE: {table_name}\n"
                f"SOURCE: {source_id}\n"
                f"DESCRIPTION: {t_desc}\n"
                f"COLUMN_MANIFEST_WITH_SAMPLES:\n- " + "\n- ".join(col_detail_list)
            )

            # --- Level 3: Column Narrative (Fine-grained) ---
            for col in tv_data.get("columns", []):
                c_name = col.get("column_name")
                c_type = col.get("data_type", "unknown")
                c_samples = col.get("samples", [])
                precise_desc = column_semantics.get(c_name, "Refer to table description.")
                
                column_entries.append(
                    f"FIELD: '{c_name}' in Table '{table_name}' ({source_id})\n"
                    f"SEMANTIC MEANING: {precise_desc}\n"
                    f"TECHNICAL SPEC: Type {c_type}, Samples {c_samples}\n"
                    f"LINEAGE: {source_id} > {table_name} > {c_name}"
                )

    # Save outputs
    with open("KB_LEVEL_1_SOURCES.txt", "w", encoding='utf-8') as f: f.write("\n\n===\n\n".join(source_entries))
    with open("KB_LEVEL_2_TABLES.txt", "w", encoding='utf-8') as f: f.write("\n\n===\n\n".join(table_entries))
    with open("KB_LEVEL_3_COLUMNS.txt", "w", encoding='utf-8') as f: f.write("\n\n===\n\n".join(column_entries))

if __name__ == "__main__":
    generate_perfect_stratified_kb("data")