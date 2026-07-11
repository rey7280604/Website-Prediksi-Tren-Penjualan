# ==============================================================================
# SPLIT DATA BY YEAR - MEMISAHKAN DATA 2020-2024 DAN 2025
# ==============================================================================

import pandas as pd
import os
from datetime import datetime

# --- KONFIGURASI ---
INPUT_FILE = 'Amazon_Sales_Complete.csv'
OUTPUT_FOLDER = 'csv_upload'

# Folder output
FOLDER_2020_2024 = os.path.join(OUTPUT_FOLDER, '2020-2024')
FOLDER_2025 = os.path.join(OUTPUT_FOLDER, '2025')

# Nama file output
OUTPUT_FILE_2020_2024 = 'Amazon_Sales_2020_2024.csv'
OUTPUT_FILE_2025 = 'Amazon_Sales_2025.csv'

def create_folders():
    """Membuat folder yang diperlukan"""
    folders = [OUTPUT_FOLDER, FOLDER_2020_2024, FOLDER_2025]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder dibuat: {folder}")
        else:
            print(f"Folder sudah ada: {folder}")

def load_data(file_path):
    """Memuat data dari file CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"\nBerhasil memuat file: {file_path}")
        print(f"Total baris: {len(df):,}")
        return df
    except Exception as e:
        print(f"Gagal memuat file: {e}")
        return None

def split_data_by_year(df):
    """
    Memisahkan data menjadi dua bagian:
    1. Data tahun 2020-2024
    2. Data tahun 2025
    """
    # Konversi kolom Date ke datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Tambahkan kolom year sementara untuk filtering
    df['_year'] = df['Date'].dt.year
    
    # Filter data
    df_2020_2024 = df[df['_year'] <= 2024].copy()
    df_2025 = df[df['_year'] == 2025].copy()
    
    # Hapus kolom temporary
    df_2020_2024 = df_2020_2024.drop(columns=['_year'])
    df_2025 = df_2025.drop(columns=['_year'])
    
    return df_2020_2024, df_2025

def analyze_data(df_2020_2024, df_2025):
    """Menganalisis data yang sudah dipisahkan"""
    print("\n" + "="*70)
    print("ANALISIS DATA HASIL PEMISAHAN")
    print("="*70)
    
    # Analisis data 2020-2024
    print(f"\nDATA 2020-2024:")
    print(f"  Total baris: {len(df_2020_2024):,}")
    print(f"  Tanggal minimum: {df_2020_2024['Date'].min().date()}")
    print(f"  Tanggal maksimum: {df_2020_2024['Date'].max().date()}")
    
    if 'Product Name' in df_2020_2024.columns:
        print(f"  Jumlah produk unik: {df_2020_2024['Product Name'].nunique()}")
    
    if 'Units Sold' in df_2020_2024.columns:
        print(f"  Total unit terjual: {df_2020_2024['Units Sold'].sum():,.0f}")
    
    # Analisis per tahun untuk 2020-2024
    years_2020_2024 = pd.to_datetime(df_2020_2024['Date']).dt.year
    year_counts = years_2020_2024.value_counts().sort_index()
    print(f"\n  Rincian per tahun:")
    for year, count in year_counts.items():
        print(f"    Tahun {year}: {count:,} baris")
    
    # Analisis data 2025
    print(f"\nDATA 2025:")
    print(f"  Total baris: {len(df_2025):,}")
    
    if len(df_2025) > 0:
        print(f"  Tanggal minimum: {df_2025['Date'].min().date()}")
        print(f"  Tanggal maksimum: {df_2025['Date'].max().date()}")
        
        if 'Product Name' in df_2025.columns:
            print(f"  Jumlah produk unik: {df_2025['Product Name'].nunique()}")
        
        if 'Units Sold' in df_2025.columns:
            print(f"  Total unit terjual: {df_2025['Units Sold'].sum():,.0f}")
        
        # Analisis per bulan untuk 2025
        months_2025 = pd.to_datetime(df_2025['Date']).dt.month
        month_counts = months_2025.value_counts().sort_index()
        print(f"\n  Rincian per bulan:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in month_counts.items():
            print(f"    Bulan {month} ({month_names[month-1]}): {count:,} baris")
    else:
        print("  [WARNING] Tidak ada data untuk tahun 2025!")

def save_data(df, output_path):
    """Menyimpan data ke file CSV"""
    try:
        df.to_csv(output_path, index=False)
        print(f"  [OK] Disimpan ke: {output_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Gagal menyimpan: {e}")
        return False

def generate_summary_report(df_original, df_2020_2024, df_2025):
    """Membuat laporan ringkasan"""
    report_file = os.path.join(OUTPUT_FOLDER, 'split_summary_report.txt')
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("LAPORAN PEMISAHAN DATA\n")
            f.write("="*70 + "\n")
            f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"File Input: {INPUT_FILE}\n")
            f.write(f"Total baris awal: {len(df_original):,}\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("DATA 2020-2024\n")
            f.write("-"*70 + "\n")
            f.write(f"File: {os.path.join(FOLDER_2020_2024, OUTPUT_FILE_2020_2024)}\n")
            f.write(f"Total baris: {len(df_2020_2024):,}\n")
            f.write(f"Rentang tanggal: {df_2020_2024['Date'].min().date()} s/d {df_2020_2024['Date'].max().date()}\n")
            
            if 'Product Name' in df_2020_2024.columns:
                f.write(f"Jumlah produk: {df_2020_2024['Product Name'].nunique()}\n")
            
            # Rincian per tahun
            years = pd.to_datetime(df_2020_2024['Date']).dt.year
            year_counts = years.value_counts().sort_index()
            f.write("\nRincian per tahun:\n")
            for year, count in year_counts.items():
                f.write(f"  {year}: {count:,} baris\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("DATA 2025\n")
            f.write("-"*70 + "\n")
            f.write(f"File: {os.path.join(FOLDER_2025, OUTPUT_FILE_2025)}\n")
            f.write(f"Total baris: {len(df_2025):,}\n")
            
            if len(df_2025) > 0:
                f.write(f"Rentang tanggal: {df_2025['Date'].min().date()} s/d {df_2025['Date'].max().date()}\n")
                
                if 'Product Name' in df_2025.columns:
                    f.write(f"Jumlah produk: {df_2025['Product Name'].nunique()}\n")
                
                # Rincian per bulan
                months = pd.to_datetime(df_2025['Date']).dt.month
                month_counts = months.value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                f.write("\nRincian per bulan:\n")
                for month, count in month_counts.items():
                    f.write(f"  Bulan {month} ({month_names[month-1]}): {count:,} baris\n")
            else:
                f.write("Tidak ada data untuk tahun 2025\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("VERIFIKASI\n")
            f.write("-"*70 + "\n")
            f.write(f"Total baris input: {len(df_original):,}\n")
            f.write(f"Total baris output: {len(df_2020_2024) + len(df_2025):,}\n")
            f.write(f"Selisih: {len(df_original) - (len(df_2020_2024) + len(df_2025)):,}\n")
        
        print(f"\nLaporan ringkasan disimpan ke: {report_file}")
        
    except Exception as e:
        print(f"Gagal membuat laporan: {e}")

def main():
    print("\n" + "="*70)
    print("MEMISAHKAN DATA BERDASARKAN TAHUN")
    print("="*70)
    print(f"File Input: {INPUT_FILE}")
    print(f"Folder Output: {OUTPUT_FOLDER}/")
    print(f"  - 2020-2024: {FOLDER_2020_2024}/")
    print(f"  - 2025: {FOLDER_2025}/")
    print("="*70)
    
    # 1. Buat folder
    create_folders()
    
    # 2. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    df_original = df.copy()
    
    # 3. Pisahkan data
    print("\nMemisahkan data...")
    df_2020_2024, df_2025 = split_data_by_year(df)
    
    print(f"\nHasil pemisahan:")
    print(f"  Data 2020-2024: {len(df_2020_2024):,} baris")
    print(f"  Data 2025: {len(df_2025):,} baris")
    
    # 4. Analisis data
    analyze_data(df_2020_2024, df_2025)
    
    # 5. Simpan data
    print("\n" + "="*70)
    print("MENYIMPAN FILE")
    print("="*70)
    
    path_2020_2024 = os.path.join(FOLDER_2020_2024, OUTPUT_FILE_2020_2024)
    path_2025 = os.path.join(FOLDER_2025, OUTPUT_FILE_2025)
    
    success_2020_2024 = save_data(df_2020_2024, path_2020_2024)
    success_2025 = save_data(df_2025, path_2025)
    
    # 6. Buat laporan
    if success_2020_2024 and success_2025:
        generate_summary_report(df_original, df_2020_2024, df_2025)
    
    # 7. Ringkasan akhir
    print("\n" + "="*70)
    print("PROSES SELESAI!")
    print("="*70)
    
    if success_2020_2024 and success_2025:
        print(f"\nFile berhasil disimpan:")
        print(f"  1. {path_2020_2024}")
        print(f"     ({len(df_2020_2024):,} baris)")
        print(f"  2. {path_2025}")
        print(f"     ({len(df_2025):,} baris)")
        print(f"\nTotal baris: {len(df_2020_2024) + len(df_2025):,}")
    else:
        print("\n[WARNING] Beberapa file gagal disimpan!")

if __name__ == "__main__":
    main()