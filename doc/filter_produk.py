# =============================================================================
# KODE UNTUK MENAMPILKAN ANALISIS PRODUK YANG DIGUNAKAN DALAM PENELITIAN
# Berdasarkan data prediksi 9 produk terlaris dari website
# =============================================================================

import pandas as pd
import tkinter as tk
from tkinter import ttk

# =============================================================================
# DATA PRODUK BERDASARKAN HASIL PREDIKSI WEBSITE (9 PRODUK)
# Berdasarkan screenshot dari website prediksi tren penjualan
# =============================================================================

data_produk = [
    {'Produk': 'Whey Protein', 'Prediksi Penjualan': 62951},
    {'Produk': 'Fish Oil', 'Prediksi Penjualan': 58792},
    {'Produk': 'Biotin', 'Prediksi Penjualan': 56847},
    {'Produk': 'Iron Supplement', 'Prediksi Penjualan': 56739},
    {'Produk': 'Vitamin C', 'Prediksi Penjualan': 56190},
    {'Produk': 'Ashwagandha', 'Prediksi Penjualan': 55683},
    {'Produk': 'Magnesium', 'Prediksi Penjualan': 54394},
    {'Produk': 'Zinc', 'Prediksi Penjualan': 50281},
    {'Produk': 'BCAA', 'Prediksi Penjualan': 50132}
]

# Buat DataFrame
df_analisis = pd.DataFrame(data_produk)

# Urutkan berdasarkan prediksi penjualan tertinggi ke terendah
df_analisis = df_analisis.sort_values('Prediksi Penjualan', ascending=False)

print("\n" + "="*80)
print("ANALISIS 9 PRODUK TERLARIS HASIL PREDIKSI")
print("Platform: Amazon | Satuan: Unit (per pcs) | Periode Prediksi: Tahun 2025")
print("="*80)

print("\nTabel Prediksi 9 Produk Terlaris:\n")
print(df_analisis.to_string(index=False))
print("\n" + "="*80)

# =============================================================================
# MENAMPILKAN POPUP UNTUK SCREENSHOT
# =============================================================================

def tampilkan_popup():
    root = tk.Tk()
    root.title("ANALISIS 9 PRODUK TERLARIS HASIL PREDIKSI")
    root.geometry("750x500")
    root.configure(bg='#f0f0f0')
    
    # Judul
    title_label = tk.Label(root, text="ANALISIS 9 PRODUK TERLARIS HASIL PREDIKSI", 
                           font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#1E3A8A')
    title_label.pack(pady=10)
    
    # Subtitle
    subtitle_label = tk.Label(root, text="Platform: Amazon | Satuan Penjualan: Unit (per pcs) | Periode Prediksi: Tahun 2025", 
                              font=("Arial", 10), bg='#f0f0f0')
    subtitle_label.pack(pady=5)
    
    # Frame untuk tabel
    frame_tabel = tk.LabelFrame(root, text="TABEL PREDIKSI 9 PRODUK TERLARIS", 
                                font=("Arial", 11, "bold"), bg='#f0f0f0', padx=10, pady=10)
    frame_tabel.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Treeview untuk tabel
    columns = ('No', 'Produk', 'Prediksi Penjualan (unit)')
    tree = ttk.Treeview(frame_tabel, columns=columns, show='headings', height=9)
    
    # Set column headings
    tree.heading('No', text='No')
    tree.heading('Produk', text='Nama Produk')
    tree.heading('Prediksi Penjualan (unit)', text='Prediksi Penjualan (unit)')
    
    # Set column widths
    tree.column('No', width=50, anchor='center')
    tree.column('Produk', width=250)
    tree.column('Prediksi Penjualan (unit)', width=180, anchor='center')
    
    # Insert data
    for idx, row in df_analisis.iterrows():
        tree.insert('', 'end', values=(
            idx+1, 
            row['Produk'], 
            f"{row['Prediksi Penjualan']:,}"
        ))
    
    tree.pack(fill="both", expand=True)
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(frame_tabel, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    
    # Informasi produk terlaris
    info_frame = tk.Frame(root, bg='#f0f0f0')
    info_frame.pack(fill="x", padx=10, pady=10)
    
    produk_terlaris = df_analisis.iloc[0]['Produk']
    nilai_terlaris = df_analisis.iloc[0]['Prediksi Penjualan']
    
    info_label = tk.Label(info_frame, 
                          text=f"PRODUK TERLARIS (PREDIKSI): {produk_terlaris} dengan {nilai_terlaris:,} unit", 
                          font=("Arial", 11, "bold"), bg='#E8F0FE', fg='#1E3A8A', padx=10, pady=10)
    info_label.pack(fill="x")
    
    # Informasi total
    total_penjualan = df_analisis['Prediksi Penjualan'].sum()
    
    total_label = tk.Label(info_frame, 
                           text=f"Total Prediksi 9 Produk: {total_penjualan:,} unit", 
                           font=("Arial", 10, "bold"), bg='#E8F0FE', fg='#1E3A8A', padx=10, pady=5)
    total_label.pack(fill="x", pady=(0, 10))
    
    # Tombol tutup
    def tutup():
        root.destroy()
    
    btn_close = tk.Button(root, text="Tutup", command=tutup, font=("Arial", 10, "bold"),
                          bg='#1E3A8A', fg='white', padx=20, pady=5)
    btn_close.pack(pady=10)
    
    root.mainloop()

# =============================================================================
# MENAMPILKAN RINGKASAN DAN MEMBUKA POPUP
# =============================================================================

print("\n" + "="*80)
print("RINGKASAN HASIL ANALISIS PREDIKSI")
print("="*80)
print(f"\nJumlah produk yang diprediksi: 9 produk")
print(f"Total prediksi penjualan keseluruhan: {df_analisis['Prediksi Penjualan'].sum():,} unit")
print("\nDaftar 9 produk terlaris hasil prediksi (urutan peringkat):")
for i, row in df_analisis.iterrows():
    print(f"   {i+1}. {row['Produk']}: {row['Prediksi Penjualan']:,} unit")
print("\n" + "="*80)
print("\nMembuka jendela popup untuk screenshot...")
print("Silakan screenshot jendela yang muncul untuk keperluan skripsi.")
print("="*80)

# Membuka popup window
tampilkan_popup()

print("\nProgram selesai.")