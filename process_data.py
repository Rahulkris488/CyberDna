import pandas as pd
import numpy as np
import os
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
FOLDER_PATH = 'C:/Users/navit/OneDrive/Desktop/Datasets/archive' # UPDATE THIS
OUTPUT_FILE = 'final_labeled_dataset_chunked.csv' # Changed output name slightly
# --------------------

def calculate_entropy(series):
    """Calculates the Shannon Entropy."""
    counts = series.value_counts()
    probabilities = counts / len(series)
    log_probs = np.log2(probabilities[probabilities > 0])
    entropy = -np.sum(probabilities[probabilities > 0] * log_probs)
    return entropy if not np.isnan(entropy) else 0

print("--- CyberDNA Dataset Factory Started (Rich Features) ---")

try:
    all_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]
    if not all_files:
        print(f"❌ ERROR: Found 0 CSV files in '{FOLDER_PATH}'.")
        exit()
except FileNotFoundError:
     print(f"❌ ERROR: The folder '{FOLDER_PATH}' was not found.")
     exit()

# Flag to check if it's the first file (to write headers)
first_file = True

print(f"\n[*] Processing {len(all_files)} files one by one...")

# === LOOP THROUGH EACH FILE ===
for file_path in all_files:
    file_name = os.path.basename(file_path)
    print(f"\n--- Processing file: {file_name} ---")

    try:
        # === STEP 1 (Inside Loop): READ ONE FILE ===
        print(f"  [1/4] Reading file...")
        current_df = pd.read_csv(file_path, low_memory=False)

        # === STEP 2 (Inside Loop): CLEAN DATA ===
        print(f"  [2/4] Cleaning data...")
        current_df.columns = current_df.columns.str.strip()

        # ** NEW, RICHER SET OF REQUIRED COLUMNS **
        required_columns = [
            'Src IP', 'Dst IP', 'Dst Port', 'Timestamp', 'Flow Duration',
            'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd Pkts/s', 'Bwd Pkts/s', 'FIN Flag Cnt', 'SYN Flag Cnt', 'Label'
        ]

        # Check if essential columns exist in THIS file
        if not all(col in current_df.columns for col in ['Src IP', 'Dst IP', 'Label']):
            print(f"        ⚠️ WARNING: Skipping file {file_name} due to missing essential IP/Label columns.")
            continue # Skip to the next file

        # Select only required columns present in this file
        available_req_cols = [col for col in required_columns if col in current_df.columns]
        current_df = current_df[available_req_cols].copy()

        # Convert Timestamp and numeric columns, coercing errors
        current_df['Timestamp'] = pd.to_datetime(current_df['Timestamp'], errors='coerce')
        numeric_cols = [
            'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd Pkts/s', 'Bwd Pkts/s', 'FIN Flag Cnt', 'SYN Flag Cnt'
        ]
        
        # Ensure we only try to convert columns that actually exist in this file
        available_numeric_cols = [col for col in numeric_cols if col in current_df.columns]
        for col in available_numeric_cols:
            current_df[col] = pd.to_numeric(current_df[col], errors='coerce')

        # Replace infinites and drop rows with NaN in essential columns
        current_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        current_df.dropna(subset=['Src IP', 'Timestamp'] + available_numeric_cols + ['Label'], inplace=True)
        print("        Data cleaned.")

        # === STEP 3 (Inside Loop): ENGINEER FEATURES ===
        print(f"  [3/4] Engineering Genome features...")
        if len(current_df) == 0:
             print("        No valid data after cleaning, skipping feature engineering.")
             continue

        # ** NEW, RICHER AGGREGATIONS **
        aggregations = {
            'Mean_Flow_Duration': ('Flow Duration', 'mean'),
            'Mean_Tot_Fwd_Pkts': ('Tot Fwd Pkts', 'mean'),
            'Mean_Tot_Bwd_Pkts': ('Tot Bwd Pkts', 'mean'),
            'Mean_Fwd_Pkt_Len': ('Fwd Pkt Len Mean', 'mean'),
            'Std_Fwd_Pkt_Len': ('Fwd Pkt Len Std', 'mean'), # Using mean of stds as approximation
            'Mean_Bwd_Pkt_Len': ('Bwd Pkt Len Mean', 'mean'),
            'Std_Bwd_Pkt_Len': ('Bwd Pkt Len Std', 'mean'), # Using mean of stds as approximation
            'Mean_Flow_IAT': ('Flow IAT Mean', 'mean'),
            'Std_Flow_IAT': ('Flow IAT Std', 'mean'), # Using mean of stds as approximation
            'Mean_Fwd_Pkts_s': ('Fwd Pkts/s', 'mean'),
            'Mean_Bwd_Pkts_s': ('Bwd Pkts/s', 'mean'),
            'Sum_FIN_Flag_Cnt': ('FIN Flag Cnt', 'sum'),
            'Sum_SYN_Flag_Cnt': ('SYN Flag Cnt', 'sum'),
            'Port_Diversity': ('Dst Port', 'nunique'),
            'IP_Entropy': ('Dst IP', calculate_entropy),
            'Label': ('Label', lambda x: 1 if (x != 'Benign').any() else 0)
        }
        
        # Only include aggregations if the source column exists
        final_aggregations = {k: v for k, v in aggregations.items() if v[0] in current_df.columns}

        if not final_aggregations:
            print("        Not enough columns to perform aggregation, skipping.")
            continue

        genomes_df_chunk = current_df.groupby(['Src IP', pd.Grouper(key='Timestamp', freq='5min')]).agg(
            **final_aggregations
        ).reset_index()
        print(f"        Created {len(genomes_df_chunk)} Genome vectors for this file.")

        # === STEP 4 (Inside Loop): FINALIZE & APPEND ===
        print(f"  [4/4] Finalizing and appending...")
        # Add missing columns (RSSI, JA3)
        genomes_df_chunk['Mean_RSSI'] = 0
        genomes_df_chunk['RSSI_Variance'] = 0
        genomes_df_chunk['JA3_Diversity'] = 0
        genomes_df_chunk.fillna(0, inplace=True)

        # Append to the output file
        if first_file:
            genomes_df_chunk.to_csv(OUTPUT_FILE, index=False, mode='w', header=True)
            first_file = False
        else:
            genomes_df_chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
        print("        Appended results to output file.")

    except Exception as e:
        print(f"❌ ERROR processing file {file_name}: {e}")

print(f"\n🎉 CONGRATULATIONS! 🎉")
print(f"All files processed. Your new, richer dataset is ready at: {OUTPUT_FILE}")