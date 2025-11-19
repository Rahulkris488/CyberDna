import pandas as pd
import os

# --- CONFIGURATION: SET THE PATH TO YOUR FOLDER ---
folder_path = 'C:/Users/navit/OneDrive/Desktop/Datasets/archive' 
# ----------------------------------------------------

print(f"--- Checking Column Headers in: {folder_path} ---")

try:
    # Find all files in the folder that end with .csv
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not all_files:
        print("❌ ERROR: Found 0 CSV files. Please check the folder path and file extensions.")
        exit()
    # This loop goes through each file name one by one
    for file_name in all_files:
        # Create the full path to the file
        file_path = os.path.join(folder_path, file_name)
        
        # Read only the first row to get the headers quickly
        df_header = pd.read_csv(file_path, nrows=0)
        
        # Clean the column names by removing any leading/trailing spaces
        cleaned_columns = [col.strip() for col in df_header.columns]
        
        print(f"\n-> Headers for file: {file_name}")
        print(cleaned_columns)

except FileNotFoundError:
    print(f"❌ ERROR: The folder '{folder_path}' was not found. Please make sure the path is correct.")

print("\n--- Check Complete ---")