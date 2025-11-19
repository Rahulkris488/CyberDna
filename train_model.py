# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import warnings
import os
import pickle

# Ignore common warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow INFO messages

# --- CONFIGURATION ---
INPUT_DATASET_FILE = 'master_dataset.csv' # Input file from the chunked processing
OUTPUT_MODEL_FILE = 'cyberdna_autoencoder.h5'
OUTPUT_SCALER_FILE = 'scaler.pkl'
OUTPUT_FEATURES_FILE = 'feature_names.pkl'
# --------------------

print("--- CyberDNA Model Training Started ---")

# === STEP 1: LOAD YOUR DATASET ===
print(f"\n[1/5] Loading the dataset: '{INPUT_DATASET_FILE}'...")
try:
    df = pd.read_csv(INPUT_DATASET_FILE)
    print(f"✅ Success! Loaded {len(df)} Genome vectors.")
except FileNotFoundError:
    print(f"❌ ERROR: File '{INPUT_DATASET_FILE}' not found. Make sure it's in the same folder.")
    exit()
except Exception as e:
    print(f"❌ ERROR: Could not load dataset. Error: {e}")
    exit()


# === STEP 2: PREPARE DATA FOR TRAINING ===
print("\n[2/5] Preparing data for training...")

# --- Create feature matrix (drop identifiers and label) ---
features_df = df.drop(['Src IP', 'Timestamp', 'Label'], axis=1, errors='ignore').copy()

# --- Drop constant imputed columns ---
# These columns have only 0s and provide no information for the model
constant_cols_to_drop = ['Mean_RSSI', 'RSSI_Variance', 'JA3_Diversity']
print(f"[*] Dropping constant imputed columns: {constant_cols_to_drop}")
features_df.drop(columns=constant_cols_to_drop, inplace=True, errors='ignore')

# --- Log Transform Skewed Features ---
# Apply log(1+x) to features known to be highly skewed (like timing, variance)
# This helps the model handle extreme values better.
skew_cols = ['Mean_IAT', 'Packet_Size_Variance'] # Add others if needed
print(f"[*] Applying log1p transformation to: {skew_cols}")
for col in skew_cols:
    if col in features_df.columns:
        # Fill NaN, ensure non-negative, then transform
        features_df[col] = features_df[col].fillna(0).clip(lower=0)
        features_df[col] = np.log1p(features_df[col])
    else:
        print(f"    - Warning: Column '{col}' not found for log transformation.")

# --- Separate normal (Label=0) data for training ---
# Important: Autoencoder is trained ONLY on normal data
normal_features = features_df[df['Label'] == 0].copy()

if len(normal_features) == 0:
    print("❌ ERROR: No 'Normal' (Label=0) data found after filtering. Cannot train the model.")
    exit()

# Get the final list of feature names used for training
feature_names = normal_features.columns.tolist()
print(f"[*] Training with {len(feature_names)} features: {feature_names}")

# --- Split NORMAL data into training and validation sets ---
X_train, X_val = train_test_split(normal_features, test_size=0.2, random_state=42)

# --- Scale the data ---
# Fit the scaler ONLY on the training data, then transform both train and validation sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- Save the fitted scaler and feature names ---
# These are essential for consistent preprocessing during testing and deployment
print(f"[*] Saving scaler object to '{OUTPUT_SCALER_FILE}'")
with open(OUTPUT_SCALER_FILE, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[*] Saving feature names list to '{OUTPUT_FEATURES_FILE}'")
with open(OUTPUT_FEATURES_FILE, 'wb') as f:
    pickle.dump(feature_names, f)

print("✅ Success! Data is prepared and scaled.")
print(f"    -> Training samples: {len(X_train_scaled)}")
print(f"    -> Validation samples: {len(X_val_scaled)}")

# --- Set Training Epochs ---
# Allow quick runs for testing by setting environment variable QUICK=1
TRAIN_EPOCHS =200 # Keeping the 150 epochs
if os.environ.get('QUICK') == '1':
    print('[DEBUG] QUICK mode enabled: reducing epochs to 5 for fast validation')
    TRAIN_EPOCHS = 5

# === STEP 3: DEFINE THE AUTOENCODER ARCHITECTURE ===
print("\n[3/5] Building the Autoencoder model...")

input_dim = X_train_scaled.shape[1] # Number of features actually used
encoding_dim_1 = max(4, int(input_dim * 0.75)) # Example: 75% of input dim, min 4
encoding_dim_2 = max(2, int(input_dim * 0.50)) # Example: 50% of input dim, min 2 (bottleneck)

print(f"[*] Model Architecture: Input({input_dim}) -> Dense({encoding_dim_1}) -> Dense({encoding_dim_2}) -> Dense({encoding_dim_1}) -> Dense({input_dim})")

# Define the network layers using Keras Functional API
input_layer = Input(shape=(input_dim,), name='Input_Layer')
# --- Encoder ---
encoder = Dense(encoding_dim_1, activation="relu", name='Encoder_Layer_1')(input_layer)
encoder = Dense(encoding_dim_2, activation="relu", name='Bottleneck_Layer')(encoder)
# --- Decoder ---
decoder = Dense(encoding_dim_1, activation="relu", name='Decoder_Layer_1')(encoder)
# Use a linear output activation since the input data is standardized (mean=0)
decoder = Dense(input_dim, activation=None, name='Output_Layer')(decoder) # Linear output

# Create the Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder, name='CyberDNA_Autoencoder')

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print("✅ Success! Model built.")
autoencoder.summary() # Print the model structure

# === STEP 4: TRAIN THE MODEL ===
print("\n[4/5] Training the model... (This will utilize your GPU)")

# Train the model using the prepared data
history = autoencoder.fit(X_train_scaled, X_train_scaled, # Input AND Target are the same
                          epochs=TRAIN_EPOCHS,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(X_val_scaled, X_val_scaled), # Monitor on validation set
                          verbose=1)         # Show progress

print("✅ Success! Model training complete.")

# === STEP 5: SAVE THE TRAINED MODEL ===
print(f"\n[5/5] Saving the trained model to '{OUTPUT_MODEL_FILE}'...")
autoencoder.save(OUTPUT_MODEL_FILE)

print(f"\n🎉 CONGRATULATIONS! 🎉")
print(f"Your trained AI model is saved as: {OUTPUT_MODEL_FILE}")
print(f"Scaler object saved as: {OUTPUT_SCALER_FILE}")
print(f"Feature names saved as: {OUTPUT_FEATURES_FILE}")
print("You are now ready to test this model!")