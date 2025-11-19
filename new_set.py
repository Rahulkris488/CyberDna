import pandas as pd
import numpy as np
import os
import warnings
import hashlib
from collections import defaultdict
# Ensure you have: pip install scapy scapy-ssl_tls pandas numpy
from scapy.all import PcapReader, RadioTap, Dot11, IP, TCP, UDP
from scapy.layers.tls.handshake import TLSClientHello

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Folder containing your .pcap files
PCAP_FOLDER_PATH = './pcap_Folder/' 
# Name for the final output CSV
OUTPUT_FILE = 'rich_pcap_dataset.csv'
# Time window to aggregate packets (minutes)
TIME_WINDOW = 5 
# --------------------

def calculate_entropy(series_data):
    """Calculates Shannon Entropy for a list/series of values."""
    if not series_data:
        return 0
    series = pd.Series(series_data)
    counts = series.value_counts()
    probabilities = counts / len(series)
    log_probs = np.log2(probabilities[probabilities > 0])
    entropy = -np.sum(probabilities[probabilities > 0] * log_probs)
    return entropy if not np.isnan(entropy) else 0

def get_ja3_hash(packet):
    """Extracts JA3 fingerprint from TLS Client Hello packets."""
    try:
        if packet.haslayer(TLSClientHello):
            # Simplified JA3: Just hashing the Cipher Suites for this MVP
            # A full JA3 uses: SSLVersion,Cipher,SSLExtension,EllipticCurve,EllipticCurvePointFormat
            ciphers = packet[TLSClientHello].ciphers
            if ciphers:
                ciphers_str = ','.join(map(str, ciphers))
                return hashlib.md5(ciphers_str.encode()).hexdigest()
    except Exception:
        pass
    return None

print("--- CyberDNA PCAP Feature Extractor Started ---")

# Dictionary to hold packet data grouped by device (MAC) and time window
# Structure: device_data[mac][window_start] = { list of packet info }
device_data = defaultdict(lambda: defaultdict(lambda: {
    'timestamps': [],
    'sizes': [],
    'rssi_values': [],
    'dst_ips': [],
    'dst_ports': [],
    'ja3_hashes': set(),
    'syn_flags': 0,
    'fin_flags': 0
}))

# --- STEP 1: READ PCAP FILES AND EXTRACT RAW DATA ---
try:
    pcap_files = [os.path.join(PCAP_FOLDER_PATH, f) for f in os.listdir(PCAP_FOLDER_PATH) if f.endswith('.pcap')]
    if not pcap_files:
        print(f"❌ ERROR: No .pcap files found in {PCAP_FOLDER_PATH}")
        exit()
except FileNotFoundError:
    print(f"❌ ERROR: Folder {PCAP_FOLDER_PATH} not found.")
    exit()

print(f"[*] Found {len(pcap_files)} PCAP files. Processing...")

total_packets = 0
valid_packets = 0

for pcap_file in pcap_files:
    print(f"    -> Reading: {os.path.basename(pcap_file)} ...")
    try:
        # PcapReader is memory efficient (reads packet by packet)
        with PcapReader(pcap_file) as reader:
            for packet in reader:
                total_packets += 1
                
                # We need 802.11 frames with a Source Address (addr2)
                if not (packet.haslayer(Dot11) and packet.addr2):
                    continue
                
                valid_packets += 1
                mac = packet.addr2
                ts = float(packet.time)
                
                # Calculate Time Window Key (e.g., 1678880000)
                window_key = int(ts // (TIME_WINDOW * 60)) * (TIME_WINDOW * 60)
                
                # Point to the specific data bucket for this device+window
                data_bucket = device_data[mac][window_key]
                
                # 1. BASIC STATS
                data_bucket['timestamps'].append(ts)
                data_bucket['sizes'].append(len(packet))
                
                # 2. RSSI (Physical Fingerprint)
                if packet.haslayer(RadioTap):
                    try:
                        # Handle cases where signal might be a list or single value
                        signal = packet[RadioTap].dbm_antsignal
                        if isinstance(signal, list):
                            signal = signal[0]
                        data_bucket['rssi_values'].append(int(signal))
                    except (AttributeError, TypeError, IndexError):
                        pass # No RSSI in this packet

                # 3. IP/PORT (Network Behavior) - Requires IP layer inside 802.11
                # Note: Encrypted Wi-Fi data (most traffic) hides IP headers unless decrypted.
                # This part works best on open networks or if you have the key.
                if packet.haslayer(IP):
                    data_bucket['dst_ips'].append(packet[IP].dst)
                
                if packet.haslayer(TCP):
                    data_bucket['dst_ports'].append(packet[TCP].dport)
                    # Count Flags
                    if packet[TCP].flags.S: data_bucket['syn_flags'] += 1
                    if packet[TCP].flags.F: data_bucket['fin_flags'] += 1
                elif packet.haslayer(UDP):
                    data_bucket['dst_ports'].append(packet[UDP].dport)
                
                # 4. JA3 (Application Fingerprint)
                ja3 = get_ja3_hash(packet)
                if ja3:
                    data_bucket['ja3_hashes'].add(ja3)

    except Exception as e:
        print(f"        [!] Error reading file: {e}")

print(f"[*] Processed {total_packets} packets. Found {valid_packets} valid Wi-Fi frames.")


# --- STEP 2: CALCULATE GENOME FEATURES ---
print(f"\n[*] Calculating 10+ Genome features for each device/window...")

genome_list = []

for mac, windows in device_data.items():
    for window_start, data in windows.items():
        # Skip empty windows
        if len(data['sizes']) == 0:
            continue
            
        # Convert lists to numpy arrays for fast math
        sizes = np.array(data['sizes'])
        rssis = np.array(data['rssi_values'])
        timestamps = np.array(sorted(data['timestamps']))
        
        # Calculate Inter-Arrival Times (IAT)
        if len(timestamps) > 1:
            iats = np.diff(timestamps)
        else:
            iats = np.array([0])

        # --- FEATURE ENGINEERING ---
        genome = {
            # Identifiers
            'Src IP': mac, # Using MAC as ID
            'Timestamp': pd.to_datetime(window_start, unit='s'),
            
            # 1. Packet Size Features
            'Mean_Packet_Size': np.mean(sizes),
            'Packet_Size_Variance': np.var(sizes),
            
            # 2. Timing Features
            'Mean_IAT': np.mean(iats),
            'Mean_Flow_Duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            
            # 3. Traffic Volume
            'Mean_Tot_Fwd_Pkts': len(sizes), # Total packets seen
            
            # 4. Diversity Features
            'Port_Diversity': len(set(data['dst_ports'])),
            'IP_Entropy': calculate_entropy(data['dst_ips']),
            
            # 5. Flag Counts
            'Sum_SYN_Flag_Cnt': data['syn_flags'],
            'Sum_FIN_Flag_Cnt': data['fin_flags'],
            
            # 6. THE UNIQUE FEATURES (RSSI & JA3)
            'Mean_RSSI': np.mean(rssis) if len(rssis) > 0 else 0,
            'RSSI_Variance': np.var(rssis) if len(rssis) > 0 else 0,
            'JA3_Diversity': len(data['ja3_hashes']),
            
            # Label (Assuming normal for captured baseline data)
            'Label': 0 
        }
        genome_list.append(genome)

# --- STEP 3: SAVE TO CSV ---
print(f"\n[*] Generating final dataset...")
if len(genome_list) > 0:
    df_final = pd.DataFrame(genome_list)
    
    # Reorder columns to match your main dataset if needed
    # df_final = df_final[['Src IP', 'Timestamp', 'Mean_Packet_Size', ...]]
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 CONGRATULATIONS! Saved {len(df_final)} rich Genomes to '{OUTPUT_FILE}'")
    print("This dataset contains ALL your desired features, including RSSI and JA3.")
else:
    print("❌ Warning: No genomes were generated. Check your PCAP files.")