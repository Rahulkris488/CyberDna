import pandas as pd
import numpy as np
import os
import warnings
from collections import defaultdict

# --- CONFIGURATION ---
# <<< CHANGE THIS >>>
# Set this to the TOP-LEVEL folder you want to search in.
# Based on your image, this is the correct path.
FOLDER_PATH = 'RSSI-Dataset/Dataset/' 

# The name for your new, RSSI-specific dataset
OUTPUT_FILE = 'rssi_genomes.csv'

# How many RSSI readings to group into a single "Genome"
GENOME_CHUNK_SIZE = 50
# --------------------

print(f"--- RSSI Genome Factory Started ---")
print(f"Recursively searching for .txt files in: {FOLDER_PATH}")

# This dictionary will hold the list of all RSSI readings for each node
all_readings = defaultdict(list)

# === STEP 1: READ AND PARSE ALL .txt FILES (RECURSIVELY) ===
try:
    txt_files_found = 0
    # os.walk() will go through every folder and subfolder
    for root, dirs, files in os.walk(FOLDER_PATH):
        for file_name in files:
            # Check if the file is a .txt file
            if file_name.endswith('.txt'):
                txt_files_found += 1
                file_path = os.path.join(root, file_name)
                # Print the full path to show it's working
                print(f"    -> Reading file: {file_path}") 
                
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if ':' not in line:
                            continue  # Skip blank lines
                        
                        try:
                            # Split the line, e.g., "Node A: -24"
                            node, rssi_str = line.split(':', 1)
                            node = node.strip()
                            rssi_val = int(rssi_str.strip())
                            
                            # Add the RSSI value to the list for that node
                            all_readings[node].append(rssi_val)
                        except ValueError:
                            print(f"        - Warning: Skipping malformed line: {line}")
    
    if txt_files_found == 0:
        print(f"❌ ERROR: Found 0 .txt files in '{FOLDER_PATH}' and its subdirectories.")
        exit()

    print(f"✅ Success! Read {txt_files_found} files. Loaded all raw RSSI readings for {len(all_readings)} nodes.")

except FileNotFoundError:
    print(f"❌ ERROR: The folder '{FOLDER_PATH}' was not found. Please update the path.")
    exit()
except Exception as e:
    print(f"❌ ERROR: An error occurred: {e}")
    exit()

# === STEP 2: ENGINEER GENOMES FROM RSSI DATA ===
print(f"\n[*] Engineering Genomes by chunking data (Chunk Size = {GENOME_CHUNK_SIZE})...")

# This is the list of all 11 columns for your final dataset
FINAL_HEADERS = [
    'Src IP', 'Timestamp', 'Mean_Packet_Size', 'Packet_Size_Variance', 'Mean_IAT',
    'Port_Diversity', 'IP_Entropy', 'Label', 'Mean_RSSI', 'RSSI_Variance', 'JA3_Diversity'
]

# This list will hold all the new Genome rows we create
genome_list = []
current_time = pd.Timestamp.now() # Start a simulated timestamp

# Loop through each node (Node A, Node B, etc.)
for node, rssi_list in all_readings.items():
    print(f"    -> Processing node: {node} ({len(rssi_list)} total readings)")
    
    # Loop through the list of RSSI readings in chunks of GENOME_CHUNK_SIZE
    for i in range(0, len(rssi_list), GENOME_CHUNK_SIZE):
        chunk = rssi_list[i : i + GENOME_CHUNK_SIZE]
        
        # Make sure we have a full chunk to get accurate stats
        if len(chunk) < GENOME_CHUNK_SIZE:
            continue # Skip the last partial chunk
            
        # Calculate the two features we have
        mean_rssi = np.mean(chunk)
        var_rssi = np.var(chunk)
        
        # Create the new Genome row
        genome_row = {
            'Src IP': node, # Use the Node name as the identifier
            'Timestamp': current_time,
            'Mean_Packet_Size': 0,
            'Packet_Size_Variance': 0,
            'Mean_IAT': 0,
            'Port_Diversity': 0,
            'IP_Entropy': 0,
            'Label': 0, # Assume this is 'Normal' traffic
            'Mean_RSSI': mean_rssi,
            'RSSI_Variance': var_rssi,
            'JA3_Diversity': 0
        }
        genome_list.append(genome_row)
        
        # Increment the simulated time for the next row
        current_time += pd.Timedelta(minutes=5)

print(f"✅ Success! Created {len(genome_list)} new Genome vectors.")

# === STEP 3: SAVE THE NEW DATASET ===
if genome_list:
    # Convert our list of dictionaries into a pandas DataFrame
    rssi_df = pd.DataFrame(genome_list, columns=FINAL_HEADERS)
    
    # Save the new dataset
    rssi_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎉 CONGRATULATIONS! 🎉")
    print(f"Your new dataset with real RSSI values is saved as: {OUTPUT_FILE}")
else:
    print("\n❌ No full chunks of data were found. The output file was not created. Try a smaller GENOME_CHUNK_SIZE.")