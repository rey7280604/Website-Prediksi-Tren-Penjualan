# ==============================================================================
# DATA CLEANING - FILTER PRODUK LENGKAP
# Membersihkan dataset dengan menghapus produk yang tidak memiliki data lengkap
# ==============================================================================

import pandas as pd
import os
from datetime import datetime

# --- KONFIGURASI ---
INPUT_FILE = 'Amazon_Sales_Cleaned.csv'  # Nama file input
OUTPUT_FILE = 'Supplement_Sales_Cleaned.csv'  # Nama file output
MIN_MONTHS_REQUIRED = 10  # Minimal bulan yang diperlukan agar produk dianggap lengkap

def load_data(file_path):
    """Memuat data dari file CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Berhasil memuat file: {file_path}")
        print(f"     Total baris: {len(df):,}")
        return df
    except Exception as e:
        print(f"[ERROR] Gagal memuat file: {e}")
        return None

def identify_date_column(df):
    """Mengidentifikasi kolom tanggal dalam dataset"""
    if 'Date' in df.columns:
        return 'Date'
    
    date_like = [col for col in df.columns if 'date' in col.lower() or 'tanggal' in col.lower()]
    if date_like:
        return date_like[0]
    
    return None

def identify_product_column(df):
    """Mengidentifikasi kolom produk dalam dataset"""
    if 'Product Name' in df.columns:
        return 'Product Name'
    
    product_like = [col for col in df.columns if 'product' in col.lower() or 'produk' in col.lower()]
    if product_like:
        return product_like[0]
    
    return None

def analyze_products(df, date_col, product_col):
    """
    Menganalisis kelengkapan data setiap produk per tahun.
    Mengembalikan daftar produk yang lengkap dan yang tidak lengkap.
    """
    df_analysis = df.copy()
    df_analysis['Year'] = df_analysis[date_col].dt.year
    df_analysis['Month'] = df_analysis[date_col].dt.month
    
    available_years = sorted(df_analysis['Year'].unique())
    
    print("\n" + "="*70)
    print("ANALISIS KELENGKAPAN DATA PRODUK PER TAHUN")
    print("="*70)
    
    complete_products_all_years = set()
    incomplete_products_info = []
    
    for year in available_years:
        year_data = df_analysis[df_analysis['Year'] == year]
        total_products_in_year = year_data[product_col].nunique()
        
        # Hitung berapa bulan setiap produk muncul
        product_month_counts = year_data.groupby(product_col)['Month'].nunique()
        
        complete_products = product_month_counts[product_month_counts >= MIN_MONTHS_REQUIRED].index.tolist()
        incomplete_products = product_month_counts[product_month_counts < MIN_MONTHS_REQUIRED].index.tolist()
        
        print(f"\n[TAHUN {year}]")
        print(f"  Total produk: {total_products_in_year}")
        print(f"  Produk lengkap (>= {MIN_MONTHS_REQUIRED} bulan): {len(complete_products)}")
        print(f"  Produk tidak lengkap (< {MIN_MONTHS_REQUIRED} bulan): {len(incomplete_products)}")
        
        if incomplete_products:
            print(f"\n  Produk yang akan DIHAPUS dari tahun {year}:")
            for prod in incomplete_products[:10]:  # Tampilkan maks 10
                month_count = product_month_counts[prod]
                print(f"    - {prod[:50]}... ({month_count} bulan)")
                incomplete_products_info.append({
                    'Tahun': year,
                    'Produk': prod,
                    'Jumlah_Bulan': month_count
                })
            if len(incomplete_products) > 10:
                print(f"    ... dan {len(incomplete_products) - 10} produk lainnya")
        
        complete_products_all_years.update(complete_products)
    
    return complete_products_all_years, incomplete_products_info

def filter_complete_products(df, date_col, product_col, valid_products):
    """
    Memfilter dataset hanya untuk produk yang memiliki data lengkap.
    """
    df_filtered = df[df[product_col].isin(valid_products)].copy()
    return df_filtered

def save_filtered_data(df, output_path):
    """Menyimpan data yang sudah difilter ke file CSV"""
    try:
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Data berhasil disimpan ke: {output_path}")
        print(f"     Total baris setelah filtering: {len(df):,}")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan file: {e}")
        return False

def save_removed_products_report(incomplete_products_info, output_path):
    """Menyimpan laporan produk yang dihapus"""
    if not incomplete_products_info:
        return
    
    report_df = pd.DataFrame(incomplete_products_info)
    report_file = output_path.replace('.csv', '_removed_products.csv')
    
    try:
        report_df.to_csv(report_file, index=False)
        print(f"[OK] Laporan produk yang dihapus disimpan ke: {report_file}")
    except Exception as e:
        print(f"[WARNING] Gagal menyimpan laporan: {e}")

def generate_summary_report(df_original, df_filtered, incomplete_products_info, output_path):
    """Membuat laporan ringkasan dalam format TXT"""
    report_file = output_path.replace('.csv', '_summary_report.txt')
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("LAPORAN DATA CLEANING - FILTER PRODUK LENGKAP\n")
            f.write("="*70 + "\n")
            f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"File Input: {INPUT_FILE}\n")
            f.write(f"File Output: {OUTPUT_FILE}\n")
            f.write(f"Minimal Bulan: {MIN_MONTHS_REQUIRED}\n")
            f.write("\n" + "-"*70 + "\n")
            f.write("RINGKASAN PERUBAHAN\n")
            f.write("-"*70 + "\n")
            f.write(f"Total baris sebelum filtering: {len(df_original):,}\n")
            f.write(f"Total baris setelah filtering: {len(df_filtered):,}\n")
            f.write(f"Baris yang dihapus: {len(df_original) - len(df_filtered):,}\n")
            f.write(f"Persentase data yang dipertahankan: {(len(df_filtered)/len(df_original)*100):.2f}%\n")
            
            if incomplete_products_info:
                f.write("\n" + "-"*70 + "\n")
                f.write("PRODUK YANG DIHAPUS\n")
                f.write("-"*70 + "\n")
                
                # Group by tahun
                report_df = pd.DataFrame(incomplete_products_info)
                for year in report_df['Tahun'].unique():
                    year_data = report_df[report_df['Tahun'] == year]
                    f.write(f"\nTAHUN {year}:\n")
                    f.write(f"  Jumlah produk dihapus: {len(year_data)}\n")
                    for _, row in year_data.iterrows():
                        f.write(f"    - {row['Produk']} ({row['Jumlah_Bulan']} bulan)\n")
        
        print(f"[OK] Laporan ringkasan disimpan ke: {report_file}")
        
    except Exception as e:
        print(f"[WARNING] Gagal membuat laporan ringkasan: {e}")

def main():
    print("\n" + "="*70)
    print("DATA CLEANING - FILTER PRODUK DENGAN DATA LENGKAP")
    print("="*70)
    print(f"File Input: {INPUT_FILE}")
    print(f"File Output: {OUTPUT_FILE}")
    print(f"Minimal bulan yang diperlukan: {MIN_MONTHS_REQUIRED}")
    print("="*70)
    
    # 1. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    df_original = df.copy()
    
    # 2. Identifikasi kolom penting
    date_col = identify_date_column(df)
    if date_col is None:
        print("[ERROR] Kolom tanggal tidak ditemukan dalam dataset!")
        print("        Pastikan ada kolom 'Date' atau kolom sejenis.")
        return
    
    product_col = identify_product_column(df)
    if product_col is None:
        print("[ERROR] Kolom produk tidak ditemukan dalam dataset!")
        print("        Pastikan ada kolom 'Product Name' atau kolom sejenis.")
        return
    
    print(f"\n[INFO] Kolom tanggal: '{date_col}'")
    print(f"[INFO] Kolom produk: '{product_col}'")
    
    # 3. Konversi kolom tanggal
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 4. Analisis kelengkapan produk
    valid_products, incomplete_products_info = analyze_products(df, date_col, product_col)
    
    if len(valid_products) == 0:
        print("\n[WARNING] Tidak ada produk yang memenuhi kriteria kelengkapan data!")
        print("          Coba turunkan nilai MIN_MONTHS_REQUIRED.")
        return
    
    print(f"\n[INFO] Total produk yang DIPERTAHANKAN: {len(valid_products)}")
    
    # 5. Filter data
    df_filtered = filter_complete_products(df, date_col, product_col, valid_products)
    
    # 6. Hapus kolom tambahan jika ada
    if 'Year' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['Year'])
    if 'Month' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['Month'])
    
    # 7. Simpan data yang sudah difilter
    if save_filtered_data(df_filtered, OUTPUT_FILE):
        # 8. Simpan laporan produk yang dihapus
        save_removed_products_report(incomplete_products_info, OUTPUT_FILE)
        
        # 9. Buat laporan ringkasan
        generate_summary_report(df_original, df_filtered, incomplete_products_info, OUTPUT_FILE)
        
        print("\n" + "="*70)
        print("PROSES DATA CLEANING SELESAI!")
        print("="*70)
        print(f"\nFile yang dihasilkan:")
        print(f"  1. {OUTPUT_FILE} - Data yang sudah dibersihkan")
        print(f"  2. {OUTPUT_FILE.replace('.csv', '_removed_products.csv')} - Daftar produk yang dihapus")
        print(f"  3. {OUTPUT_FILE.replace('.csv', '_summary_report.txt')} - Laporan ringkasan")
        print("\nGunakan file CSV yang sudah dibersihkan untuk analisis selanjutnya.")

if __name__ == "__main__":
    main()