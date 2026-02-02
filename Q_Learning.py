import pandas as pd
import numpy as np

# --- 1. KONFIGURASI (Sesuai Manual Excel Anda) ---
ALPHA = 0.5   # Learning Rate
GAMMA = 0.8   # Discount Factor
FILENAME = 'dataset_qlearning_50baris.csv'

# --- 2. PERSIAPAN DATA ---
# Membaca file CSV (Pastikan pemisahnya titik koma ';' atau koma ',' sesuai file Anda)
try:
    df = pd.read_csv(FILENAME, delimiter=';') # Ganti delimiter jadi ',' jika error
except FileNotFoundError:
    print(f"File {FILENAME} tidak ditemukan. Pastikan nama file benar.")
    exit()

# Menyiapkan Q-Table Kosong
# Kita ambil semua State dan Action unik dari data untuk membuat tabel
states = sorted(df['state'].unique())
actions = sorted(df['action'].unique())

# Membuat DataFrame Q-Table dengan nilai awal 0.0
q_table = pd.DataFrame(0.0, index=states, columns=actions)

print("=== MULAI PERHITUNGAN Q-LEARNING ===\n")

# --- 3. LOOPING (ITERASI BARIS DEMI BARIS) ---
# List untuk menyimpan hasil hitungan agar bisa dicek mirip Excel
history_perhitungan = []

for index, row in df.iterrows():
    # Ambil data dari baris saat ini
    state_sekarang = row['state']
    aksi = row['action']
    next_state = row['next_state']
    reward = row['reward']
    episode = row['episode']
    step = row['step']

    # 1. Ambil Q Lama (G)
    # Sama seperti melihat sheet MODEL
    q_lama = q_table.at[state_sekarang, aksi]

    # 2. Ambil Q Max Depan (H)
    # Sama seperti melihat sheet MODEL baris Next State, cari yang paling besar
    # Jika next_state adalah Goal (S9) dan aksi 'Diam', biasanya max-nya 0
    q_max_depan = q_table.loc[next_state].max()

    # 3. Hitung Rumus (I)
    # Rumus: Q_Baru = Q_Lama + Alpha * (Reward + Gamma * Q_Max - Q_Lama)
    
    # Hitung Temporal Difference (Error)
    target = reward + (GAMMA * q_max_depan)
    td_error = target - q_lama
    
    # Nilai Q Baru
    q_baru = q_lama + (ALPHA * td_error)

    # 4. UPDATE Q-TABLE (MODEL)
    # Sama seperti Anda mengetik angka baru dan mewarnai sel
    q_table.at[state_sekarang, aksi] = q_baru

    # Simpan log untuk ditampilkan (opsional, biar mirip tabel Excel)
    history_perhitungan.append({
        'Ep': episode,
        'Step': step,
        'State': state_sekarang,
        'Action': aksi,
        'Reward': reward,
        'Q_Lama': round(q_lama, 4),
        'Q_Max_Depan': round(q_max_depan, 4),
        'Q_Baru': round(q_baru, 4)
    })

# --- 4. MENAMPILKAN HASIL ---

# Tampilkan 10 baris pertama (seperti Sheet HITUNGAN)
print("--- HASIL PERHITUNGAN ---")
result_df = pd.DataFrame(history_perhitungan)
print(result_df.head(10).to_string(index=False)) 
# .head(10) artinya cuma tampilkan 10 baris pertama biar tidak kepanjang

print("\n" + "="*50 + "\n")

# Tampilkan Q-Table Akhir (Seperti Sheet MODEL yang penuh warna)
print("--- Q-TABLE FINAL (Isi Otak Robot) ---")
print(q_table.round(4)) # Dibulatkan 4 angka belakang koma