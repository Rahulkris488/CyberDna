# test_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
# **** NEW IMPORTS for Metrics ****
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
# *********************************
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

# Ignore common warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow INFO messages
# --- CONFIGURATION ---
INPUT_DATASET_FILE = 'master_dataset.csv' # Dataset created by process_data.py
MODEL_FILE = 'cyberdna_autoencoder.h5' # Model saved by train_model.py
SCALER_FILE = 'scaler.pkl'             # Scaler saved by train_model.py
FEATURES_FILE = 'feature_names.pkl'    # Feature names saved by train_model.py
# --------------------

print("--- CyberDNA Model Testing Started ---")

# === STEP 1: LOAD DATASET, MODEL, SCALER, FEATURES ===
print(f"\n[1/6] Loading dataset: '{INPUT_DATASET_FILE}'...") # Step count updated
try:
    df = pd.read_csv(INPUT_DATASET_FILE)
    print(f"✅ Success! Loaded {len(df)} Genome vectors.")
except FileNotFoundError:
    print(f"❌ ERROR: File '{INPUT_DATASET_FILE}' not found.")
    exit()
except Exception as e:
    print(f"❌ ERROR: Could not load dataset. Error: {e}")
    exit()

print(f"\n[2/6] Loading trained model: '{MODEL_FILE}'...") # Step count updated
try:
    autoencoder = tf.keras.models.load_model(MODEL_FILE)
    print("✅ Success! Model loaded.")
except Exception as e:
    print(f"❌ ERROR: Could not load model file. Error: {e}")
    exit()

print(f"\n[3/6] Loading scaler and feature names...") # Step count updated
try:
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Success! Loaded scaler from '{SCALER_FILE}'.")
except FileNotFoundError:
    print(f"❌ ERROR: File '{SCALER_FILE}' not found.")
    exit()
except Exception as e:
     print(f"❌ ERROR: Could not load scaler file. Error: {e}")
     exit()

try:
    with open(FEATURES_FILE, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✅ Success! Loaded feature names from '{FEATURES_FILE}'.")
except FileNotFoundError:
     print(f"❌ ERROR: File '{FEATURES_FILE}' not found.")
     exit()
except Exception as e:
     print(f"❌ ERROR: Could not load feature names file. Error: {e}")
     exit()


# === STEP 3: PREPARE ALL DATA (NORMAL AND ATTACK) FOR TESTING ===
print("\n[4/6] Preparing data for testing...") # Step count updated

if 'Label' not in df.columns:
     print(f"❌ ERROR: 'Label' column not found in '{INPUT_DATASET_FILE}'.")
     exit()
normal_df = df[df['Label'] == 0].copy()
attack_df = df[df['Label'] == 1].copy()

# Drop identifiers first
features_df = df.drop(['Src IP', 'Timestamp', 'Label'], axis=1, errors='ignore').copy()
# Drop constant imputed columns (ensure this list matches training)
constant_cols_to_drop = ['Mean_RSSI', 'RSSI_Variance', 'JA3_Diversity']
features_df.drop(columns=constant_cols_to_drop, inplace=True, errors='ignore')

# Ensure columns match training order and selection
features_df = features_df[feature_names]

# Apply log transformation (ensure this list matches training)
skew_cols = ['Mean_IAT', 'Packet_Size_Variance']
print(f"[*] Applying log1p transformation to: {skew_cols}")
for col in skew_cols:
    if col in features_df.columns:
        features_df[col] = features_df[col].fillna(0).clip(lower=0)
        features_df[col] = np.log1p(features_df[col])
    else:
         print(f"    - Warning: Column '{col}' not found for log transformation.")

# Separate features after transformation
normal_features = features_df.loc[normal_df.index]
attack_features = features_df.loc[attack_df.index]

# Scale using the LOADED scaler
print("[*] Scaling data using the loaded scaler...")
normal_features_scaled = scaler.transform(normal_features)
attack_features_scaled = scaler.transform(attack_features)

print("✅ Success! Data prepared and scaled consistently with training.")

# === STEP 4: CALCULATE RECONSTRUCTION ERROR ===
print("\n[5/6] Calculating reconstruction errors...") # Step count updated
print("[*] Predicting reconstructions for normal data...")
normal_reconstructions = autoencoder.predict(normal_features_scaled)
print("[*] Predicting reconstructions for attack data...")
attack_reconstructions = autoencoder.predict(attack_features_scaled)
# Mean Squared Error per-sample
normal_mse = np.mean(np.power(normal_features_scaled - normal_reconstructions, 2), axis=1)
attack_mse = np.mean(np.power(attack_features_scaled - attack_reconstructions, 2), axis=1)
# Mean Absolute Error per-sample (robust alternative)
normal_mae = np.mean(np.abs(normal_features_scaled - normal_reconstructions), axis=1)
attack_mae = np.mean(np.abs(attack_features_scaled - attack_reconstructions), axis=1)
# Max absolute error across features per-sample (detects single-feature spikes)
normal_maxerr = np.max(np.abs(normal_features_scaled - normal_reconstructions), axis=1)
attack_maxerr = np.max(np.abs(attack_features_scaled - attack_reconstructions), axis=1)
print("✅ Success! Reconstruction errors calculated (MSE, MAE, MaxErr).")

# === STEP 5: CALCULATE METRICS & FIND OPTIMAL THRESHOLD ===
print("\n[6/6] Calculating performance metrics...") # Step count updated

true_labels = np.concatenate([np.zeros(len(normal_mse)), np.ones(len(attack_mse))])

# Evaluate multiple candidate anomaly scores and pick the best by ROC AUC
candidate_metrics = {
    'mse': (normal_mse, attack_mse),
    'mae': (normal_mae, attack_mae),
    'maxerr': (normal_maxerr, attack_maxerr)
}

best_metric = None
best_auc = -1.0
best_scores = None
best_all_errors = None
best_inverted = False

for name, (n_vals, a_vals) in candidate_metrics.items():
    combined = np.concatenate([n_vals, a_vals])
    # prefer higher score => attack; if attacks have lower mean, invert
    if a_vals.mean() < n_vals.mean():
        sc = -combined
        inverted_flag = True
    else:
        sc = combined
        inverted_flag = False
    try:
        auc_val = roc_auc_score(true_labels, sc)
    except Exception:
        auc_val = 0.0
    print(f"Metric '{name}': attack_mean={a_vals.mean():.6e}, normal_mean={n_vals.mean():.6e}, AUC={auc_val:.6f}, inverted={inverted_flag}")
    if auc_val > best_auc:
        best_auc = auc_val
        best_metric = name
        best_scores = sc
        best_all_errors = combined
        best_inverted = inverted_flag

print(f"\nSelected metric for detection: '{best_metric}' with ROC AUC={best_auc:.6f} (inverted={best_inverted})")

# Use the selected scores for downstream PR/thresholding
scores = best_scores
all_errors = best_all_errors
precision, recall, thresholds = precision_recall_curve(true_labels, scores)
f1_scores = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1], out=np.zeros_like(precision[:-1]), where=(precision[:-1] + recall[:-1]) != 0) # Adjust slicing for threshold length

# Find threshold maximizing F1, handle potential empty f1_scores
if len(f1_scores) > 0:
    optimal_idx = np.argmax(f1_scores)
    # Ensure index is within bounds of thresholds array
    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
        best_f1 = f1_scores[optimal_idx]
    else:
        # Fallback if index issue occurs (e.g., use max threshold or default)
        print("Warning: Could not determine optimal threshold reliably, using max error as fallback.")
        optimal_threshold = thresholds[-1] if len(thresholds)>0 else normal_mse.max() + 1e-6
        best_f1 = f1_score(true_labels, (all_errors > optimal_threshold).astype(int))

else:
     print("Warning: No F1 scores calculated. Using max error as fallback threshold.")
     optimal_threshold = normal_mse.max() + 1e-6 # Set threshold slightly above max normal error
     best_f1 = 0

print(f"\n[*] Optimal Threshold found: {optimal_threshold:.6f} (Maximizes F1 Score)")

predictions = (scores > optimal_threshold).astype(int)
accuracy = accuracy_score(true_labels, predictions)
final_precision = precision_score(true_labels, predictions, zero_division=0)
final_recall = recall_score(true_labels, predictions, zero_division=0)
conf_matrix = confusion_matrix(true_labels, predictions)

print("\n=== Performance Metrics at Optimal Threshold ===")
print(f" - Accuracy:  {accuracy * 100:.2f}%")
print(f" - Precision: {final_precision * 100:.2f}%")
print(f" - Recall:    {final_recall * 100:.2f}%")
print(f" - F1-Score:  {best_f1 * 100:.2f}%")
print("\nConfusion Matrix:")
print("             Predicted Normal  Predicted Attack")
# Ensure conf_matrix has expected shape before indexing
if conf_matrix.shape == (2, 2):
    print(f"Actual Normal:  {conf_matrix[0, 0]:<15} {conf_matrix[0, 1]:<15}")
    print(f"Actual Attack:  {conf_matrix[1, 0]:<15} {conf_matrix[1, 1]:<15}")
    print(f"( TN={conf_matrix[0, 0]}, FP={conf_matrix[0, 1]}, FN={conf_matrix[1, 0]}, TP={conf_matrix[1, 1]} )")
else:
    print("Could not compute full confusion matrix (likely only one class predicted).")


print("✅ Success! Metrics calculated.")

# --- Diagnostics: show top problematic samples ---
print('\n[*] Extracting top problematic samples for inspection...')
# Build a combined DataFrame mapping back to original indices
normal_idx = normal_df.index.to_numpy()
attack_idx = attack_df.index.to_numpy()
combined_idx = np.concatenate([normal_idx, attack_idx])

df_diag = pd.DataFrame({
    'original_index': combined_idx,
    'label': true_labels.astype(int),
    'metric_value': all_errors,
    'score': scores
})

# False negatives: attack samples with LOW score (missed attacks)
fn_attacks = df_diag[df_diag['label'] == 1].nsmallest(10, 'score')
# False positives: normal samples with HIGH score (misclassified normals)
fp_normals = df_diag[df_diag['label'] == 0].nlargest(10, 'score')

print('\nTop 10 missed attacks (attack label==1, lowest score):')
print(fn_attacks)
print('\nTop 10 false positives (normal label==0, highest score):')
print(fp_normals)

# Save diagnostics to CSV for deeper inspection
diag_out = 'diagnostics_top_samples.csv'
pd.concat([fn_attacks, fp_normals]).to_csv(diag_out, index=False)
print(f"Saved top problematic samples to '{diag_out}' for manual inspection.")

# === STEP 6: VISUALIZE THE RESULTS ===
print("\n[*] Generating histogram of reconstruction errors...")

# ----- Additional diagnostics -----
# 1) ROC AUC and ROC curve
try:
    auc_roc = roc_auc_score(true_labels, scores)
    fpr, tpr, roc_thresh = roc_curve(true_labels, scores)
    youden_idx = np.argmax(tpr - fpr)
    youden_thresh = roc_thresh[youden_idx]
    print(f"\nROC AUC: {auc_roc:.6f}")
    print(f"Youden optimal threshold (from ROC): {youden_thresh:.6f}")
except Exception as e:
    print(f"Warning: Could not compute ROC AUC. Error: {e}")

# 2) Cumulative distributions (CDF) for direct overlap visibility
plt.figure(figsize=(12, 6))
bins_cdf = np.linspace(min(all_errors), max(all_errors), 1000)
plt.hist(normal_mse, bins=bins_cdf, density=True, histtype='step', cumulative=True, label='Normal CDF', color='blue')
plt.hist(attack_mse, bins=bins_cdf, density=True, histtype='step', cumulative=True, label='Attack CDF', color='red')
plt.axvline(optimal_threshold, color='green', linestyle='dashed', linewidth=2, label=f'F1 Optimal ({optimal_threshold:.4f})')
if 'youden_thresh' in locals():
    plt.axvline(youden_thresh, color='orange', linestyle='dashdot', linewidth=2, label=f'Youden Optimal ({youden_thresh:.4f})')
plt.xlabel("Mean Squared Error (Reconstruction Error)")
plt.ylabel("Cumulative Density")
plt.title("CDF of Reconstruction Errors (Normal vs Attack)")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.show()

# 3) ROC Curve plot (useful when classes are imbalanced)
try:
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.scatter(fpr[youden_idx], tpr[youden_idx], c='orange', label='Youden Optimum')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.3)
    plt.show()
except Exception:
    pass

# 4) Sorted MSE scatter so you can see where attacks fall relative to normals
plt.figure(figsize=(12, 4))
all_idx = np.arange(len(all_errors))
# Sort by the detection scores (so highest-scoring samples are at the end)
sorted_idx = np.argsort(scores)
# For plotting, show the original metric values (all_errors) but sorted by score
sorted_errors = all_errors[sorted_idx]
sorted_labels = true_labels[sorted_idx]
colors = ['blue' if lab == 0 else 'red' for lab in sorted_labels]
plt.scatter(np.arange(len(sorted_errors)), sorted_errors, c=colors, s=6, alpha=0.7)
plt.axhline(optimal_threshold, color='green', linestyle='dashed', label=f'F1 Optimal ({optimal_threshold:.4f})')
if 'youden_thresh' in locals():
    plt.axhline(youden_thresh, color='orange', linestyle='dashdot', label=f'Youden ({youden_thresh:.4f})')
plt.yscale('log')
plt.xlabel('Samples (sorted by MSE)')
plt.ylabel('MSE (log scale)')
plt.title('Sorted Reconstruction Errors (blue=normal, red=attack)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.show()

print("\n--- Testing Complete ---")