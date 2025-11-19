import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# 1. The large dataset from your CSV files (processed from CSE-CIC-IDS-2018)
CSV_DATASET_FILE = 'final_labeled_dataset_chunked.csv'

# 2. The new dataset you just created from your RSSI .txt files
RSSI_DATASET_FILE = 'rssi_genomes.csv'

# 3. The name for your final, combined master dataset
OUTPUT_MASTER_FILE = 'master_dataset.csv'
# --------------------

print("--- CyberDNA Master Dataset Combiner Started ---")

# === STEP 1: LOAD BOTH DATASETS ===
print(f"\n[1/3] Loading main dataset: '{CSV_DATASET_FILE}'...")
try:
    df_csv = pd.read_csv(CSV_DATASET_FILE)
    print(f"✅ Success! Loaded {len(df_csv)} rows from main dataset.")
except FileNotFoundError:
    print(f"❌ ERROR: File '{CSV_DATASET_FILE}' not found. Please make sure it's in the same folder.")
    exit()

print(f"\n[2/3] Loading RSSI dataset: '{RSSI_DATASET_FILE}'...")
try:
    df_rssi = pd.read_csv(RSSI_DATASET_FILE)
    print(f"✅ Success! Loaded {len(df_rssi)} rows from RSSI dataset.")
except FileNotFoundError:
    print(f"❌ ERROR: File '{RSSI_DATASET_FILE}' not found. Please make sure it's in the same folder.")
    exit()

# === STEP 3: COMBINE, CLEAN, AND SAVE ===
print("\n[3/3] Combining datasets and saving the master file...")

# Ensure both DataFrames have the exact same columns in the same order
# We will use the columns from the main (CSV) dataset as the standard
master_columns = df_csv.columns.tolist()

# Reorder columns in the RSSI dataset to match the master list.
# This will also add any missing columns (like Packet_Size_Variance) and fill them with NaN
df_rssi_aligned = df_rssi.reindex(columns=master_columns)

# Use pd.concat to stack the two DataFrames vertically
master_df = pd.concat([df_csv, df_rssi_aligned], ignore_index=True)

# Final cleanup: Fill any and all missing values (NaNs) with 0
master_df.fillna(0, inplace=True)

# Save the final, combined dataset
master_df.to_csv(OUTPUT_MASTER_FILE, index=False)

print(f"\n🎉 CONGRATULATIONS! 🎉")
print(f"Your master dataset with {len(master_df)} total rows is saved as: {OUTPUT_MASTER_FILE}")
print("You can now use this file to train your best and final Autoencoder model.")