# ==============================================================================
# DATA CLEANING - HAPUS PRODUK MULTIVITAMIN DAN ELECTROLYTE POWDER
# ==============================================================================

import pandas as pd
import os
from datetime import datetime

# --- KONFIGURASI ---
INPUT_FILE = 'Amazon_Sales_in_IDR.csv'
OUTPUT_FILE = 'Amazon_Sales_Cleaned.csv'

# Daftar produk yang akan dihapus (case insensitive)
PRODUCTS_TO_REMOVE = [
    'multivitamin',
    'electrolyte powder',
]

def load_data(file_path):
    """Memuat data dari file CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"Berhasil memuat file: {file_path}")
        print(f"Total baris awal: {len(df):,}")
        return df
    except Exception as e:
        print(f"Gagal memuat file: {e}")
        return None

def identify_product_column(df):
    """Mengidentifikasi kolom produk dalam dataset"""
    if 'Product Name' in df.columns:
        return 'Product Name'
    
    product_like = [col for col in df.columns if 'product' in col.lower() or 'produk' in col.lower()]
    if product_like:
        return product_like[0]
    
    return None

def remove_specific_products(df, product_col, products_to_remove):
    """
    Menghapus produk yang mengandung kata kunci tertentu.
    """
    df_filtered = df.copy()
    
    # Buat pattern untuk pencarian (case insensitive)
    pattern = '|'.join(products_to_remove)
    
    # Cari produk yang match dengan pattern
    mask = df_filtered[product_col].str.lower().str.contains(pattern, na=False)
    
    removed_df = df_filtered[mask]
    kept_df = df_filtered[~mask]
    
    # Hitung statistik
    unique_products_removed = removed_df[product_col].unique()
    total_rows_removed = len(removed_df)
    
    return kept_df, removed_df, unique_products_removed, total_rows_removed

def save_cleaned_data(df, output_path):
    """Menyimpan data yang sudah dibersihkan"""
    try:
        df.to_csv(output_path, index=False)
        print(f"\nData berhasil disimpan ke: {output_path}")
        print(f"Total baris setelah cleaning: {len(df):,}")
        return True
    except Exception as e:
        print(f"Gagal menyimpan file: {e}")
        return False

def save_removed_products_report(removed_df, unique_products, output_path):
    """Menyimpan laporan produk yang dihapus"""
    if len(removed_df) == 0:
        return
    
    report_file = output_path.replace('.csv', '_removed_products.csv')
    
    # Buat ringkasan per produk
    summary = removed_df.groupby('Product Name').agg({
        'Units Sold': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    summary.columns = ['Product Name', 'Total Units Sold', 'Total Revenue']
    summary = summary.sort_values('Total Units Sold', ascending=False)
    
    try:
        summary.to_csv(report_file, index=False)
        print(f"\nLaporan produk yang dihapus disimpan ke: {report_file}")
    except Exception as e:
        print(f"Gagal menyimpan laporan: {e}")

def generate_summary_report(df_original, df_filtered, removed_df, unique_products, output_path):
    """Membuat laporan ringkasan dalam format TXT"""
    report_file = output_path.replace('.csv', '_summary_report.txt')
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("LAPORAN DATA CLEANING\n")
            f.write("="*60 + "\n")
            f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"File Input: {INPUT_FILE}\n")
            f.write(f"File Output: {OUTPUT_FILE}\n")
            f.write("\n" + "-"*60 + "\n")
            f.write("PRODUK YANG DIHAPUS (Keyword)\n")
            f.write("-"*60 + "\n")
            for kw in PRODUCTS_TO_REMOVE:
                f.write(f"  - {kw}\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("RINGKASAN PERUBAHAN\n")
            f.write("-"*60 + "\n")
            f.write(f"Total baris sebelum cleaning: {len(df_original):,}\n")
            f.write(f"Total baris setelah cleaning: {len(df_filtered):,}\n")
            f.write(f"Baris yang dihapus: {len(removed_df):,}\n")
            f.write(f"Persentase data dihapus: {(len(removed_df)/len(df_original)*100):.2f}%\n")
            f.write(f"Persentase data dipertahankan: {(len(df_filtered)/len(df_original)*100):.2f}%\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("DAFTAR PRODUK YANG DIHAPUS\n")
            f.write("-"*60 + "\n")
            for i, prod in enumerate(sorted(unique_products), 1):
                # Hitung total units untuk produk ini
                prod_data = removed_df[removed_df['Product Name'] == prod]
                total_units = prod_data['Units Sold'].sum() if 'Units Sold' in prod_data.columns else 0
                f.write(f"{i}. {prod}\n")
                f.write(f"   - Baris dihapus: {len(prod_data):,}\n")
                f.write(f"   - Total unit terjual: {total_units:,.0f}\n")
            
            # Tampilkan 10 produk teratas yang dipertahankan
            f.write("\n" + "-"*60 + "\n")
            f.write("10 PRODUK TERATAS YANG DIPERTAHANKAN\n")
            f.write("-"*60 + "\n")
            if 'Product Name' in df_filtered.columns and 'Units Sold' in df_filtered.columns:
                top_products = df_filtered.groupby('Product Name')['Units Sold'].sum().sort_values(ascending=False).head(10)
                for i, (prod, units) in enumerate(top_products.items(), 1):
                    f.write(f"{i}. {prod[:50]}: {units:,.0f} unit\n")
        
        print(f"Laporan ringkasan disimpan ke: {report_file}")
        
    except Exception as e:
        print(f"Gagal membuat laporan ringkasan: {e}")

def main():
    print("\n" + "="*60)
    print("DATA CLEANING - HAPUS PRODUK TERTENTU")
    print("="*60)
    print(f"File Input: {INPUT_FILE}")
    print(f"File Output: {OUTPUT_FILE}")
    print("\nKeyword produk yang akan dihapus:")
    for kw in PRODUCTS_TO_REMOVE:
        print(f"  - {kw}")
    print("="*60)
    
    # 1. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    df_original = df.copy()
    
    # 2. Identifikasi kolom produk
    product_col = identify_product_column(df)
    if product_col is None:
        print("ERROR: Kolom produk tidak ditemukan!")
        return
    
    print(f"\nKolom produk: '{product_col}'")
    
    # 3. Cek apakah ada kolom Date dan tampilkan info
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Rentang tanggal: {df['Date'].min().date()} s/d {df['Date'].max().date()}")
    
    # 4. Hapus produk yang ditentukan
    df_filtered, removed_df, unique_products_removed, total_rows_removed = remove_specific_products(
        df, product_col, PRODUCTS_TO_REMOVE
    )
    
    print(f"\nHasil Cleaning:")
    print(f"  - Produk unik yang dihapus: {len(unique_products_removed)}")
    print(f"  - Total baris dihapus: {total_rows_removed:,}")
    print(f"  - Total baris dipertahankan: {len(df_filtered):,}")
    
    if len(unique_products_removed) > 0:
        print(f"\nDaftar produk yang dihapus:")
        for prod in sorted(unique_products_removed):
            count = len(removed_df[removed_df[product_col] == prod])
            print(f"  - {prod} ({count:,} baris)")
    
    # 5. Simpan data yang sudah dibersihkan
    if save_cleaned_data(df_filtered, OUTPUT_FILE):
        # 6. Simpan laporan
        save_removed_products_report(removed_df, unique_products_removed, OUTPUT_FILE)
        
        # 7. Buat laporan ringkasan
        generate_summary_report(df_original, df_filtered, removed_df, unique_products_removed, OUTPUT_FILE)
        
        print("\n" + "="*60)
        print("PROSES DATA CLEANING SELESAI!")
        print("="*60)
        print(f"\nFile yang dihasilkan:")
        print(f"  1. {OUTPUT_FILE} - Data bersih")
        print(f"  2. {OUTPUT_FILE.replace('.csv', '_removed_products.csv')} - Daftar produk dihapus")
        print(f"  3. {OUTPUT_FILE.replace('.csv', '_summary_report.txt')} - Laporan ringkasan")

if __name__ == "__main__":
    main()