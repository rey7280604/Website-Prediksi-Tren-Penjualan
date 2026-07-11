# ==============================================================================
# FILL MISSING DATES - MENGISI TANGGAL KOSONG DENGAN POLA MUSIMAN
# Rentang tanggal: dari data minimum sampai 31 Desember 2025
# Data setelah data terakhir diisi dengan pola musiman (bukan flat)
# TANPA MENAMBAHKAN KOLOM BARU
# ==============================================================================

import pandas as pd
import numpy as np
from datetime import datetime

# --- KONFIGURASI ---
INPUT_FILE = 'Amazon_Sales_Cleaned.csv'
OUTPUT_FILE = 'Amazon_Sales_Complete.csv'

# Target akhir data (bisa diubah sesuai kebutuhan)
TARGET_END_DATE = '2025-12-31'

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

def calculate_seasonal_pattern(df, product_data, numeric_col):
    """
    Menghitung pola musiman bulanan untuk suatu produk.
    Mengembalikan faktor pengali per bulan.
    """
    if len(product_data.dropna(subset=[numeric_col])) < 30:
        return None
    
    # Hitung rata-rata per bulan
    product_data_copy = product_data.copy()
    product_data_copy['Month'] = product_data_copy.index.month
    monthly_avg = product_data_copy.groupby('Month')[numeric_col].mean()
    
    # Hitung rata-rata keseluruhan
    overall_avg = product_data_copy[numeric_col].mean()
    
    if overall_avg == 0 or overall_avg is None or pd.isna(overall_avg):
        return None
    
    # Hitung faktor musiman (seasonal index)
    seasonal_factors = monthly_avg / overall_avg
    
    return seasonal_factors

def fill_missing_dates(df, target_end_date):
    """
    Mengisi tanggal yang kosong untuk setiap produk.
    Data akan dilengkapi dari tanggal minimum sampai target_end_date.
    - Data di antara tanggal yang ada: interpolasi linear
    - Data setelah tanggal terakhir: diisi dengan tren + pola musiman
    TIDAK menambahkan kolom baru apapun.
    """
    df_filled = df.copy()
    
    # Konversi kolom Date ke datetime
    df_filled['Date'] = pd.to_datetime(df_filled['Date'])
    
    # Tentukan rentang tanggal lengkap
    min_date = df_filled['Date'].min()
    max_date = pd.to_datetime(target_end_date)
    
    print(f"\nRentang tanggal data:")
    print(f"  Tanggal minimum (dari data): {min_date.date()}")
    print(f"  Tanggal maksimum (target): {max_date.date()}")
    
    # Hitung selisih hari
    days_diff = (max_date - min_date).days
    print(f"  Total hari dalam rentang: {days_diff + 1} hari")
    
    # Buat rentang tanggal lengkap (setiap hari)
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    print(f"  Jumlah tanggal yang akan di-generate: {len(all_dates)}")
    
    # Dapatkan daftar produk unik
    product_col = 'Product Name'
    if product_col not in df_filled.columns:
        product_like = [col for col in df_filled.columns if 'product' in col.lower()]
        if product_like:
            product_col = product_like[0]
        else:
            print("ERROR: Kolom produk tidak ditemukan!")
            return None
    
    unique_products = df_filled[product_col].unique()
    print(f"\nJumlah produk unik: {len(unique_products)}")
    
    # Hitung total data yang akan dihasilkan
    total_expected_rows = len(unique_products) * len(all_dates)
    print(f"Total baris yang akan dihasilkan: {total_expected_rows:,}")
    
    # Dapatkan daftar kolom original (untuk memastikan tidak ada kolom baru)
    original_columns = df_filled.columns.tolist()
    
    # List untuk menyimpan hasil
    filled_data = []
    
    # Proses setiap produk
    for i, product in enumerate(unique_products):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Memproses produk {i+1}/{len(unique_products)}: {product[:30]}...")
        
        # Filter data untuk produk ini
        product_data = df_filled[df_filled[product_col] == product].copy()
        
        # Set Date sebagai index
        product_data = product_data.set_index('Date')
        
        # Simpan tanggal terakhir yang memiliki data aktual
        actual_data_mask = ~product_data['Units Sold'].isna()
        if actual_data_mask.any():
            last_actual_date = product_data[actual_data_mask].index.max()
        else:
            last_actual_date = None
        
        # Reindex ke semua tanggal dalam rentang (sampai target_end_date)
        product_data = product_data.reindex(all_dates)
        
        # Isi kolom produk
        product_data[product_col] = product
        
        # Isi kolom Category dengan nilai yang konsisten
        if 'Category' in product_data.columns:
            category_value = product_data['Category'].dropna().iloc[0] if len(product_data['Category'].dropna()) > 0 else 'Unknown'
            product_data['Category'] = product_data['Category'].fillna(category_value)
        
        # Isi kolom Location dengan nilai yang paling sering muncul
        if 'Location' in product_data.columns:
            location_values = product_data['Location'].dropna()
            location_value = location_values.mode()[0] if len(location_values) > 0 else 'Unknown'
            product_data['Location'] = product_data['Location'].fillna(location_value)
        
        # Isi kolom Platform dengan nilai yang paling sering muncul
        if 'Platform' in product_data.columns:
            platform_values = product_data['Platform'].dropna()
            platform_value = platform_values.mode()[0] if len(platform_values) > 0 else 'Unknown'
            product_data['Platform'] = product_data['Platform'].fillna(platform_value)
        
        # Kolom numerik yang perlu diisi
        numeric_columns = ['Units Sold', 'Revenue', 'Price (IDR)']
        
        for col in numeric_columns:
            if col in product_data.columns:
                # Hitung pola musiman untuk kolom ini
                seasonal_factors = calculate_seasonal_pattern(df, product_data, col)
                
                # Hitung rata-rata keseluruhan untuk produk ini
                overall_avg = product_data[col].mean()
                
                # Hitung tren linear sederhana (untuk proyeksi ke depan)
                if last_actual_date is not None and len(product_data.loc[:last_actual_date, col].dropna()) >= 2:
                    # Ambil data aktual saja
                    actual_series = product_data.loc[:last_actual_date, col].dropna()
                    
                    if len(actual_series) >= 7:
                        # Hitung tren menggunakan moving average 7 hari
                        trend_series = actual_series.rolling(window=7, min_periods=1).mean()
                        
                        # Hitung slope (kecenderungan naik/turun)
                        if len(trend_series) >= 30:
                            # Ambil rata-rata 30 hari terakhir vs 30 hari sebelumnya
                            recent_avg = trend_series.iloc[-30:].mean() if len(trend_series) >= 30 else trend_series.mean()
                            older_avg = trend_series.iloc[:-30].mean() if len(trend_series) > 30 else trend_series.mean()
                            
                            if older_avg > 0:
                                trend_factor = recent_avg / older_avg
                                # Batasi faktor tren antara 0.7 dan 1.3 (tidak terlalu ekstrim)
                                trend_factor = max(0.7, min(1.3, trend_factor))
                            else:
                                trend_factor = 1.0
                        else:
                            trend_factor = 1.0
                    else:
                        trend_factor = 1.0
                else:
                    trend_factor = 1.0
                
                # Proses pengisian nilai
                if product_data[col].isna().any():
                    # Dapatkan index tanggal yang kosong
                    nan_indices = product_data[col].isna()
                    nan_dates = product_data.index[nan_indices]
                    
                    # Pisahkan: sebelum last_actual_date dan sesudahnya
                    if last_actual_date is not None:
                        before_mask = nan_dates <= last_actual_date
                        after_mask = nan_dates > last_actual_date
                        
                        # Untuk data sebelum last_actual_date: interpolasi linear
                        if before_mask.any():
                            # Interpolasi dulu untuk semua
                            interpolated = product_data[col].interpolate(method='linear', limit_direction='both')
                            product_data[col] = interpolated
                        
                        # Untuk data setelah last_actual_date: gunakan pola musiman
                        if after_mask.any() and seasonal_factors is not None:
                            # Ambil nilai terakhir yang sudah diisi
                            last_value = product_data.loc[last_actual_date, col]
                            
                            # Dapatkan rata-rata bulanan dari data historis
                            historical_monthly = product_data.loc[:last_actual_date, col].groupby(
                                product_data.loc[:last_actual_date].index.month
                            ).mean()
                            
                            # Isi setiap tanggal setelah last_actual_date
                            for date in nan_dates[after_mask]:
                                month = date.month
                                
                                # Gunakan pola musiman jika tersedia
                                if month in seasonal_factors.index and not pd.isna(seasonal_factors[month]):
                                    # Base value = rata-rata keseluruhan * faktor musiman * faktor tren
                                    base_value = overall_avg * seasonal_factors[month] * trend_factor
                                    
                                    # Tambahkan sedikit variasi random (±10%)
                                    random_variation = np.random.uniform(0.9, 1.1)
                                    value = base_value * random_variation
                                elif month in historical_monthly.index:
                                    # Fallback ke rata-rata bulanan historis
                                    value = historical_monthly[month] * trend_factor
                                    random_variation = np.random.uniform(0.9, 1.1)
                                    value = value * random_variation
                                else:
                                    # Fallback ke nilai terakhir dengan variasi
                                    value = last_value * np.random.uniform(0.85, 1.15)
                                
                                # Pastikan nilai tidak negatif
                                value = max(0, value)
                                
                                product_data.loc[date, col] = value
                    else:
                        # Jika tidak ada data aktual, interpolasi biasa
                        interpolated = product_data[col].interpolate(method='linear', limit_direction='both')
                        interpolated = interpolated.fillna(overall_avg if not pd.isna(overall_avg) else 0)
                        product_data[col] = interpolated
                
                # Final fallback: isi NaN yang tersisa dengan 0
                product_data[col] = product_data[col].fillna(0)
        
        # Kolom numerik lainnya (Price, Discount, Units Returned)
        other_numeric = ['Price', 'Discount', 'Units Returned']
        for col in other_numeric:
            if col in product_data.columns:
                # Cek apakah ini Price dalam USD (nilai kecil)
                is_usd_price = (col == 'Price' and product_data[col].max() < 100)
                
                if product_data[col].isna().any():
                    # Interpolasi dulu
                    interpolated = product_data[col].interpolate(method='linear', limit_direction='both')
                    
                    # Hitung rata-rata
                    avg_value = product_data[col].mean()
                    
                    # Isi sisa NaN
                    interpolated = interpolated.fillna(avg_value if not pd.isna(avg_value) else 0)
                    product_data[col] = interpolated
                    
                    # Untuk data setelah last_actual_date, tambahkan sedikit variasi
                    if last_actual_date is not None:
                        after_mask = product_data.index > last_actual_date
                        if after_mask.any() and not is_usd_price:
                            # Tambahkan variasi kecil (±5%)
                            for date in product_data.index[after_mask]:
                                current_val = product_data.loc[date, col]
                                if current_val > 0:
                                    variation = np.random.uniform(0.95, 1.05)
                                    product_data.loc[date, col] = current_val * variation
        
        # Reset index untuk mendapatkan kolom Date kembali
        product_data = product_data.reset_index()
        product_data = product_data.rename(columns={'index': 'Date'})
        
        # PASTIKAN hanya kolom original yang ada
        product_data = product_data[original_columns]
        
        filled_data.append(product_data)
    
    print(f"\nSelesai memproses semua produk!")
    
    # Gabungkan semua data
    df_result = pd.concat(filled_data, ignore_index=True)
    
    # Urutkan berdasarkan tanggal dan produk
    df_result = df_result.sort_values(['Date', product_col]).reset_index(drop=True)
    
    # Bulatkan nilai numerik
    if 'Units Sold' in df_result.columns:
        df_result['Units Sold'] = df_result['Units Sold'].round(0).astype(int)
    if 'Units Returned' in df_result.columns:
        df_result['Units Returned'] = df_result['Units Returned'].round(0).astype(int)
    if 'Revenue' in df_result.columns:
        df_result['Revenue'] = df_result['Revenue'].round(2)
    if 'Price' in df_result.columns:
        df_result['Price'] = df_result['Price'].round(2)
    if 'Price (IDR)' in df_result.columns:
        df_result['Price (IDR)'] = df_result['Price (IDR)'].round(0).astype(int)
    if 'Discount' in df_result.columns:
        df_result['Discount'] = df_result['Discount'].round(2)
    
    # PASTIKAN tidak ada kolom tambahan
    df_result = df_result[original_columns]
    
    return df_result, min_date, max_date, unique_products

def generate_report(df_original, df_filled, min_date, max_date, unique_products):
    """Membuat laporan perubahan data"""
    print("\n" + "="*70)
    print("LAPORAN PENGISIAN DATA KOSONG")
    print("="*70)
    
    # Statistik baris
    print(f"\nSTATISTIK BARIS:")
    print(f"  Total baris awal: {len(df_original):,}")
    print(f"  Total baris setelah pengisian: {len(df_filled):,}")
    print(f"  Baris yang ditambahkan: {len(df_filled) - len(df_original):,}")
    if len(df_original) > 0:
        print(f"  Persentase penambahan: {((len(df_filled) - len(df_original))/len(df_original)*100):.2f}%")
    
    # Statistik tanggal
    dates_original = pd.to_datetime(df_original['Date']).nunique()
    dates_filled = pd.to_datetime(df_filled['Date']).nunique()
    print(f"\nSTATISTIK TANGGAL:")
    print(f"  Jumlah hari unik awal: {dates_original}")
    print(f"  Jumlah hari unik setelah pengisian: {dates_filled}")
    print(f"  Rentang tanggal: {min_date.date()} s/d {max_date.date()}")
    
    # Statistik produk
    print(f"\nSTATISTIK PRODUK:")
    print(f"  Jumlah produk: {len(unique_products)}")
    print(f"  Setiap produk memiliki {dates_filled} hari data")
    
    # Cek data per tahun
    print(f"\nJUMLAH DATA PER TAHUN:")
    dates = pd.to_datetime(df_filled['Date'])
    years = dates.dt.year
    months = dates.dt.month
    
    temp_df = pd.DataFrame({'Year': years, 'Month': months})
    year_counts = temp_df['Year'].value_counts().sort_index()
    
    for year, count in year_counts.items():
        months_in_year = temp_df[temp_df['Year'] == year]['Month'].nunique()
        print(f"  Tahun {year}: {count:,} baris ({months_in_year} bulan)")
    
    # Cek variasi data setelah Maret 2025
    print(f"\nCEK VARIASI DATA SETELAH MARET 2025:")
    df_after_mar = df_filled[pd.to_datetime(df_filled['Date']) > '2025-03-31']
    if len(df_after_mar) > 0:
        # Cek Units Sold
        if 'Units Sold' in df_after_mar.columns:
            unique_values = df_after_mar['Units Sold'].nunique()
            total_rows = len(df_after_mar)
            print(f"  Units Sold - Nilai unik: {unique_values} dari {total_rows} baris")
            if unique_values > total_rows * 0.1:
                print(f"  [OK] Data bervariasi (tidak flat)")
            else:
                print(f"  [WARNING] Data cenderung flat/sama")
            
            # Tampilkan sample
            sample_data = df_after_mar[['Date', 'Product Name', 'Units Sold']].head(10)
            print(f"\n  Sample data setelah Maret 2025:")
            for _, row in sample_data.iterrows():
                print(f"    {row['Date']} | {row['Product Name'][:20]}... | {row['Units Sold']}")
    
    # Cek apakah tahun 2025 sudah sampai Desember
    if 2025 in year_counts.index:
        months_2025 = temp_df[temp_df['Year'] == 2025]['Month'].unique()
        print(f"\nTAHUN 2025:")
        print(f"  Bulan yang tersedia: {sorted(months_2025)}")
        if 12 in months_2025:
            print(f"  [OK] Data sudah mencapai Desember 2025")
        else:
            print(f"  [WARNING] Data belum mencapai Desember 2025!")
    
    # Tampilkan kolom yang ada di file output
    print(f"\nKOLOM OUTPUT:")
    for i, col in enumerate(df_filled.columns.tolist(), 1):
        print(f"  {i}. {col}")

def save_data(df, output_path):
    """Menyimpan data yang sudah dilengkapi"""
    try:
        df.to_csv(output_path, index=False)
        print(f"\nData berhasil disimpan ke: {output_path}")
        return True
    except Exception as e:
        print(f"Gagal menyimpan file: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("MENGISI TANGGAL KOSONG DENGAN POLA MUSIMAN")
    print("="*70)
    print(f"File Input: {INPUT_FILE}")
    print(f"File Output: {OUTPUT_FILE}")
    print(f"Target Tanggal Akhir: {TARGET_END_DATE}")
    print("="*70)
    print("\nMETODE PENGISIAN:")
    print("  - Data di antara tanggal yang ada: INTERPOLASI LINEAR")
    print("  - Data setelah tanggal terakhir: POLA MUSIMAN + TREN + VARIASI")
    print("  - Ini memastikan data setelah Maret 2025 TIDAK FLAT/SAMA SEMUA")
    print("="*70)
    
    # 1. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    df_original = df.copy()
    
    # Tampilkan kolom original
    print(f"\nKolom original ({len(df_original.columns)}):")
    for col in df_original.columns:
        print(f"  - {col}")
    
    # 2. Isi tanggal kosong sampai target_end_date
    print("\nMemproses pengisian data kosong...")
    print("(Proses ini mungkin memakan waktu beberapa saat)")
    
    result = fill_missing_dates(df, TARGET_END_DATE)
    
    if result is None:
        return
    
    df_filled, min_date, max_date, unique_products = result
    
    # 3. Generate laporan
    generate_report(df_original, df_filled, min_date, max_date, unique_products)
    
    # 4. Simpan data
    if save_data(df_filled, OUTPUT_FILE):
        print("\n" + "="*70)
        print("PROSES SELESAI!")
        print("="*70)
        print(f"\nFile output: {OUTPUT_FILE}")
        print(f"Rentang tanggal: {min_date.date()} s/d {max_date.date()}")
        print("\nFITUR PENGISIAN DATA:")
        print("  ✓ Data di antara tanggal yang ada: interpolasi linear")
        print("  ✓ Data setelah Maret 2025: pola musiman + variasi acak")
        print("  ✓ Hasil: data TIDAK FLAT, bervariasi seperti data asli")
        print(f"\nTotal data: {len(df_filled):,} baris")
        print(f"TIDAK ADA KOLOM BARU YANG DITAMBAHKAN.")
        print(f"Jumlah kolom tetap: {len(df_filled.columns)}")

if __name__ == "__main__":
    main()