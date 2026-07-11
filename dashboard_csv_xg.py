# ==============================================================================
# SISTEM PREDIKSI TREN PENJUALAN E-COMMERCE XGBOOST
# Dasbor analisis data penjualan dan prediksi tren
# ==============================================================================

# --- 1. IMPORT PUSTAKA ---
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from datetime import timedelta, datetime
import os
import re
import calendar
import locale

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from prophet import Prophet

# --- 2. PENGATURAN ---
DEFAULT_FILE_PATH = 'Amazon_Sales_in_IDR.csv'
FOLDER_UPLOAD = 'csv_upload'
FOLDER_AKTUAL = 'csv_aktual'
NILAI_TUKAR_USD_KE_IDR = 17000

# Set locale ke Indonesia
try:
    locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'Indonesian_Indonesia.1252')
    except:
        pass

# Daftar nama bulan dalam bahasa Indonesia
NAMA_BULAN_INDONESIA = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

def dapatkan_nama_bulan(indeks_bulan):
    """Mengembalikan nama bulan dalam bahasa Indonesia berdasarkan indeks (1-12)"""
    if 1 <= indeks_bulan <= 12:
        return NAMA_BULAN_INDONESIA[indeks_bulan - 1]
    return ""

def format_tanggal_indonesia(tgl):
    """Format tanggal ke dalam bahasa Indonesia: 31 Januari 2025"""
    return f"{tgl.day} {dapatkan_nama_bulan(tgl.month)} {tgl.year}"

st.set_page_config(
    page_title="Prediksi Penjualan E-Commerce",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not os.path.exists(FOLDER_UPLOAD):
    os.makedirs(FOLDER_UPLOAD)

if not os.path.exists(FOLDER_AKTUAL):
    os.makedirs(FOLDER_AKTUAL)

# --- 3. FUNGSI PEMBANTU ---
def format_angka(nilai):
    return f"{nilai:,.0f}"

def format_angka_dengan_titik(nilai):
    """Format angka dengan titik sebagai pemisah ribuan"""
    return f"{nilai:,.0f}".replace(',', '.')

def format_angka_indonesia(nilai, desimal=0):
    """Format angka dengan titik sebagai pemisah ribuan dan koma sebagai pemisah desimal"""
    if desimal == 0:
        return f"{nilai:,.0f}".replace(',', '.')
    else:
        formatted = f"{nilai:,.{desimal}f}"
        if '.' in formatted:
            int_part, dec_part = formatted.split('.')
            int_part = int_part.replace(',', '.')
            return f"{int_part},{dec_part}"
        return formatted.replace(',', '.')

def format_angka_selisih(nilai):
    """Format angka selisih tanpa tanda + untuk positif, tanda - tetap untuk negatif"""
    if nilai < 0:
        return f"-{format_angka_indonesia(abs(nilai))}"
    else:
        return f"{format_angka_indonesia(nilai)}"

def hitung_mape(y_aktual, y_prediksi):
    """Menghitung Mean Absolute Percentage Error"""
    mask = y_aktual != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_aktual[mask] - y_prediksi[mask]) / y_aktual[mask])) * 100

def hitung_rmse(y_aktual, y_prediksi):
    """Menghitung Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_aktual, y_prediksi))

def interpretasi_mape(mape):
    """Memberikan interpretasi nilai MAPE"""
    if mape < 10:
        return "Sangat Baik - Model memiliki kemampuan prediksi yang sangat akurat"
    elif mape < 20:
        return "Baik - Model memiliki kemampuan prediksi yang baik"
    elif mape < 50:
        return "Cukup - Model memiliki kemampuan prediksi yang cukup memadai"
    else:
        return "Kurang Baik - Model memiliki kemampuan prediksi yang kurang akurat"

def interpretasi_rmse(rmse, rata_rata_aktual):
    """Memberikan interpretasi nilai RMSE"""
    if rata_rata_aktual == 0:
        return "Tidak dapat diinterpretasikan (rata-rata aktual = 0)"
    rasio = rmse / rata_rata_aktual
    if rasio < 0.1:
        return "Sangat Baik - Kesalahan prediksi sangat kecil dibandingkan rata-rata nilai aktual"
    elif rasio < 0.2:
        return "Baik - Kesalahan prediksi relatif kecil"
    elif rasio < 0.5:
        return "Cukup - Kesalahan prediksi dalam batas yang dapat diterima"
    else:
        return "Kurang Baik - Kesalahan prediksi cukup besar dibandingkan rata-rata nilai aktual"

def terjemahkan_nama_kolom(df):
    peta_terjemahan = {
        'Date': 'Tanggal', 'Product Name': 'Nama Produk', 'Units Sold': 'Unit Terjual',
        'Price': 'Harga', 'Revenue': 'Pendapatan', 'Category': 'Kategori',
        'Location': 'Lokasi', 'Country': 'Negara', 'City': 'Kota',
        'Order ID': 'ID Pesanan', 'Customer': 'Pelanggan'
    }
    dict_ganti_nama = {}
    for kol in df.columns:
        for ing, ind in peta_terjemahan.items():
            if kol == ing or kol.lower() == ing.lower():
                dict_ganti_nama[kol] = ind
                break
    if dict_ganti_nama:
        df = df.rename(columns=dict_ganti_nama)
    return df

def simpan_file_unggahan(file_unggahan, folder=FOLDER_UPLOAD):
    try:
        jalur_file = os.path.join(folder, file_unggahan.name)
        with open(jalur_file, "wb") as f:
            f.write(file_unggahan.getbuffer())
        return jalur_file
    except Exception as e:
        st.error(f"Galat: {e}")
        return None

def dapatkan_file_tersimpan(folder=FOLDER_UPLOAD):
    if os.path.exists(folder):
        return [f for f in os.listdir(folder) if f.endswith('.csv')]
    return []

def hapus_file(nama_file, folder=FOLDER_UPLOAD):
    try:
        jalur_file = os.path.join(folder, nama_file)
        if os.path.exists(jalur_file):
            os.remove(jalur_file)
            return True
    except Exception as e:
        st.error(f"Galat: {e}")
    return False

def dapatkan_info_file(jalur_file):
    try:
        df = pd.read_csv(jalur_file)
        if 'Date' in df.columns:
            kol_tanggal = 'Date'
        else:
            kol_seperti_tanggal = [kol for kol in df.columns if 'date' in kol.lower() or 'tanggal' in kol.lower()]
            if not kol_seperti_tanggal:
                return None, None, None, None, None, None
            kol_tanggal = kol_seperti_tanggal[0]
        df[kol_tanggal] = pd.to_datetime(df[kol_tanggal])
        tgl_min = df[kol_tanggal].min().date()
        tgl_maks = df[kol_tanggal].max().date()
        total_baris = len(df)
        tgl_unik = df[kol_tanggal].nunique()
        if 'Units Sold' in df.columns:
            total_penjualan = df['Units Sold'].sum()
        else:
            kol_seperti_unit = [kol for kol in df.columns if 'unit' in kol.lower()]
            if kol_seperti_unit:
                total_penjualan = df[kol_seperti_unit[0]].sum()
            else:
                total_penjualan = None
        tahun_tersedia = sorted(df[kol_tanggal].dt.year.unique())
        return tgl_min, tgl_maks, total_baris, tgl_unik, total_penjualan, tahun_tersedia
    except Exception as e:
        st.error(f"Galat: {e}")
        return None, None, None, None, None, None

def deteksi_dan_konversi_mata_uang(df):
    df_terkonversi = df.copy()
    kolom_harga = [kol for kol in df.columns if 'price' in kol.lower() or 'harga' in kol.lower()]
    if not kolom_harga:
        kolom_pendapatan = [kol for kol in df.columns if 'revenue' in kol.lower() or 'pendapatan' in kol.lower()]
        if kolom_pendapatan and 'Units Sold' in df.columns:
            df_terkonversi['Harga (IDR)'] = df[kolom_pendapatan[0]] / df['Units Sold']
            return df_terkonversi, 'IDR'
    if kolom_harga:
        kol_harga = kolom_harga[0]
        contoh_harga = df[kol_harga].iloc[0] if len(df) > 0 else 0
        if isinstance(contoh_harga, str):
            contoh_harga_bersih = re.sub(r'[^\d.,-]', '', contoh_harga).replace(',', '')
            try: contoh_harga = float(contoh_harga_bersih)
            except: contoh_harga = 0
        if contoh_harga < 1000:
            def konversi_ke_idr(nilai_harga):
                if pd.isna(nilai_harga): return 0
                if isinstance(nilai_harga, str):
                    harga_bersih = re.sub(r'[^\d.,-]', '', nilai_harga).replace(',', '')
                    try: angka_harga = float(harga_bersih)
                    except: angka_harga = 0
                else: angka_harga = float(nilai_harga)
                return angka_harga * NILAI_TUKAR_USD_KE_IDR
            df_terkonversi['Harga (IDR)'] = df[kol_harga].apply(konversi_ke_idr)
            return df_terkonversi, 'USD'
        else:
            def bersihkan_harga(nilai_harga):
                if pd.isna(nilai_harga): return 0
                if isinstance(nilai_harga, str):
                    harga_bersih = re.sub(r'[^\d.,-]', '', nilai_harga).replace(',', '')
                    try: return float(harga_bersih)
                    except: return 0
                return float(nilai_harga)
            df_terkonversi['Harga (IDR)'] = df[kol_harga].apply(bersihkan_harga)
            return df_terkonversi, 'IDR'
    st.error("Kolom harga tidak ditemukan")
    return None, None

def dapatkan_kolom_harga(df, df_asli):
    if 'Harga (IDR)' in df.columns:
        return 'Harga (IDR)'
    kolom_harga = [kol for kol in df.columns if 'price' in kol.lower() or 'harga' in kol.lower()]
    if kolom_harga:
        kol_harga = kolom_harga[0]
        contoh_harga = df[kol_harga].iloc[0] if len(df) > 0 else 0
        if isinstance(contoh_harga, str):
            contoh_harga_bersih = re.sub(r'[^\d.,-]', '', contoh_harga).replace(',', '')
            try: contoh_harga = float(contoh_harga_bersih)
            except: contoh_harga = 0
        if contoh_harga < 1000:
            def konversi_ke_idr(nilai_harga):
                if pd.isna(nilai_harga): return 0
                if isinstance(nilai_harga, str):
                    harga_bersih = re.sub(r'[^\d.,-]', '', nilai_harga).replace(',', '')
                    try: angka_harga = float(harga_bersih)
                    except: angka_harga = 0
                else: angka_harga = float(nilai_harga)
                return angka_harga * NILAI_TUKAR_USD_KE_IDR
            df['Harga (IDR)'] = df[kol_harga].apply(konversi_ke_idr)
            return 'Harga (IDR)'
        else:
            def bersihkan_harga(nilai_harga):
                if pd.isna(nilai_harga): return 0
                if isinstance(nilai_harga, str):
                    harga_bersih = re.sub(r'[^\d.,-]', '', nilai_harga).replace(',', '')
                    try: return float(harga_bersih)
                    except: return 0
                return float(nilai_harga)
            df['Harga (IDR)'] = df[kol_harga].apply(bersihkan_harga)
            return 'Harga (IDR)'
    return None

# --- 4. FUNGSI PEMROSESAN DATA ---
@st.cache_data
def muat_dan_siapkan_data(jalur_file, tahun_mulai=None, tahun_akhir=None):
    try:
        df = pd.read_csv(jalur_file)
        if 'Date' not in df.columns:
            kol_seperti_tanggal = [kol for kol in df.columns if 'date' in kol.lower() or 'tanggal' in kol.lower()]
            if kol_seperti_tanggal:
                df = df.rename(columns={kol_seperti_tanggal[0]: 'Date'})
            else:
                st.error("Kolom Tanggal tidak ditemukan")
                st.stop()
        df['Date'] = pd.to_datetime(df['Date'])
        if tahun_mulai and tahun_akhir:
            df = df[(df['Date'].dt.year >= tahun_mulai) & (df['Date'].dt.year <= tahun_akhir)]
        df_asli = df.copy()
        df, jenis_mata_uang = deteksi_dan_konversi_mata_uang(df)
        if df is None: st.stop()
        if 'Units Sold' not in df.columns:
            kol_seperti_unit = [kol for kol in df.columns if 'unit' in kol.lower()]
            if kol_seperti_unit:
                df = df.rename(columns={kol_seperti_unit[0]: 'Units Sold'})
            else:
                st.error("Kolom Unit Terjual tidak ditemukan")
                st.stop()
        df['Pendapatan (IDR)'] = df['Harga (IDR)'] * df['Units Sold']
        penjualan_harian = df.groupby('Date').agg({'Units Sold': 'sum', 'Pendapatan (IDR)': 'sum'}).asfreq('D')
        penjualan_harian.fillna(0, inplace=True)
        penjualan_harian['hari_dalam_minggu'] = penjualan_harian.index.dayofweek + 1
        penjualan_harian['bulan'] = penjualan_harian.index.month
        penjualan_harian['kuartal'] = penjualan_harian.index.quarter
        penjualan_harian['tahun'] = penjualan_harian.index.year
        penjualan_harian['lag_1'] = penjualan_harian['Units Sold'].shift(1)
        penjualan_harian['lag_7'] = penjualan_harian['Units Sold'].shift(7)
        penjualan_harian['rata_bergerak_7'] = penjualan_harian['Units Sold'].rolling(window=7).mean()
        penjualan_harian['rata_bergerak_30'] = penjualan_harian['Units Sold'].rolling(window=30).mean()
        penjualan_harian['adalah_akhir_pekan'] = (penjualan_harian.index.dayofweek >= 5).astype(int)
        penjualan_harian.dropna(inplace=True)
        return penjualan_harian, df_asli, jenis_mata_uang
    except Exception as e:
        st.error(f"Galat: {e}")
        st.stop()

def buat_fitur_untuk_prediksi(data_series):
    """Membuat fitur untuk prediksi XGBoost dari Series penjualan harian"""
    df = pd.DataFrame(data_series)
    df.columns = ['Units Sold']
    
    df['hari_dalam_minggu'] = df.index.dayofweek + 1
    df['bulan'] = df.index.month
    df['kuartal'] = df.index.quarter
    df['tahun'] = df.index.year
    df['hari_dalam_bulan'] = df.index.day
    df['minggu_dalam_tahun'] = df.index.isocalendar().week.astype(int)
    df['adalah_akhir_pekan'] = (df.index.dayofweek >= 5).astype(int)
    
    df['lag_1'] = df['Units Sold'].shift(1)
    df['lag_2'] = df['Units Sold'].shift(2)
    df['lag_3'] = df['Units Sold'].shift(3)
    df['lag_7'] = df['Units Sold'].shift(7)
    df['lag_14'] = df['Units Sold'].shift(14)
    df['lag_30'] = df['Units Sold'].shift(30)
    
    df['rata_bergerak_7'] = df['Units Sold'].rolling(window=7).mean()
    df['rata_bergerak_14'] = df['Units Sold'].rolling(window=14).mean()
    df['rata_bergerak_30'] = df['Units Sold'].rolling(window=30).mean()
    df['std_7'] = df['Units Sold'].rolling(window=7).std()
    df['max_7'] = df['Units Sold'].rolling(window=7).max()
    df['min_7'] = df['Units Sold'].rolling(window=7).min()
    
    return df.dropna()

def prediksi_dengan_xgboost(data_series, hari_prediksi, tgl_terakhir):
    """Melakukan prediksi menggunakan XGBoost - MODEL UTAMA"""
    try:
        if isinstance(data_series, pd.DataFrame):
            data_series = data_series.iloc[:, 0] if data_series.shape[1] > 0 else data_series
        
        df_fitur = buat_fitur_untuk_prediksi(data_series)
        
        if len(df_fitur) < 30:
            rata_harian = data_series.mean() if len(data_series) > 0 else 0
            df_kosong = pd.DataFrame({'ds': [], 'yhat': []})
            return df_kosong, rata_harian * hari_prediksi
        
        kolom_fitur = [col for col in df_fitur.columns if col != 'Units Sold']
        X = df_fitur[kolom_fitur].values
        y = df_fitur['Units Sold'].values
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=0
        )
        model.fit(X, y)
        
        y_pred_historis = model.predict(X)
        
        tgl_terakhir_dt = pd.to_datetime(tgl_terakhir)
        tgl_prediksi = pd.date_range(
            start=tgl_terakhir_dt + timedelta(days=1),
            periods=hari_prediksi,
            freq='D'
        )
        
        data_terakhir = df_fitur.iloc[-1:].copy()
        prediksi_list = []
        
        for i in range(hari_prediksi):
            if i > 0:
                tgl_baru = tgl_prediksi[i-1]
                data_terakhir['hari_dalam_minggu'] = tgl_baru.dayofweek + 1
                data_terakhir['bulan'] = tgl_baru.month
                data_terakhir['kuartal'] = tgl_baru.quarter
                data_terakhir['tahun'] = tgl_baru.year
                data_terakhir['hari_dalam_bulan'] = tgl_baru.day
                data_terakhir['minggu_dalam_tahun'] = tgl_baru.isocalendar().week
                data_terakhir['adalah_akhir_pekan'] = 1 if tgl_baru.dayofweek >= 5 else 0
                
                if len(prediksi_list) >= 1:
                    data_terakhir['lag_1'] = prediksi_list[-1]
                if len(prediksi_list) >= 2:
                    data_terakhir['lag_2'] = prediksi_list[-2]
                if len(prediksi_list) >= 3:
                    data_terakhir['lag_3'] = prediksi_list[-3]
                if len(prediksi_list) >= 7:
                    data_terakhir['lag_7'] = prediksi_list[-7]
                if len(prediksi_list) >= 14:
                    data_terakhir['lag_14'] = np.mean(prediksi_list[-14:])
                if len(prediksi_list) >= 30:
                    data_terakhir['lag_30'] = np.mean(prediksi_list[-30:])
                
                if len(prediksi_list) >= 7:
                    data_terakhir['rata_bergerak_7'] = np.mean(prediksi_list[-7:])
                    data_terakhir['std_7'] = np.std(prediksi_list[-7:])
                    data_terakhir['max_7'] = np.max(prediksi_list[-7:])
                    data_terakhir['min_7'] = np.min(prediksi_list[-7:])
                if len(prediksi_list) >= 14:
                    data_terakhir['rata_bergerak_14'] = np.mean(prediksi_list[-14:])
                if len(prediksi_list) >= 30:
                    data_terakhir['rata_bergerak_30'] = np.mean(prediksi_list[-30:])
            
            X_pred = data_terakhir[kolom_fitur].values
            prediksi = model.predict(X_pred)[0]
            prediksi = max(0, prediksi)
            prediksi_list.append(prediksi)
        
        # Gabungkan data historis dan prediksi
        semua_tanggal = list(df_fitur.index) + list(tgl_prediksi)
        semua_nilai = list(y_pred_historis) + prediksi_list
        
        df_hasil = pd.DataFrame({
            'ds': semua_tanggal,
            'yhat': semua_nilai
        })
        
        total_terprediksi = sum(prediksi_list)
        return df_hasil, total_terprediksi
        
    except Exception as e:
        if isinstance(data_series, pd.DataFrame):
            data_series = data_series.iloc[:, 0] if data_series.shape[1] > 0 else data_series
        rata_harian = data_series.mean() if len(data_series) > 0 else 0
        df_kosong = pd.DataFrame({'ds': [], 'yhat': []})
        return df_kosong, rata_harian * hari_prediksi

def prediksi_dengan_prophet_visual(data_series, hari_prediksi):
    """Prophet HANYA untuk visualisasi smoothing"""
    try:
        df_prophet = data_series.reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet['y'] = df_prophet['y'].clip(lower=0)
        
        if len(df_prophet) < 10:
            return None
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(df_prophet)
        masa_depan = model.make_future_dataframe(periods=hari_prediksi)
        ramalan = model.predict(masa_depan)
        ramalan['yhat'] = ramalan['yhat'].clip(lower=0)
        
        return ramalan
        
    except Exception as e:
        return None

def prediksi_kombinasi(data_series, hari_prediksi, tgl_terakhir):
    """
    XGBoost untuk nilai prediksi (teknologi utama)
    Prophet HANYA untuk smoothing visualisasi grafik
    """
    # Dapatkan prediksi XGBoost (model utama)
    df_xgb, total_xgb = prediksi_dengan_xgboost(data_series, hari_prediksi, tgl_terakhir)
    
    # Dapatkan prediksi Prophet untuk visualisasi
    ramalan_prophet = prediksi_dengan_prophet_visual(data_series, hari_prediksi)
    
    # Jika Prophet berhasil, gunakan untuk visualisasi
    if ramalan_prophet is not None and len(ramalan_prophet) > 0:
        return ramalan_prophet, total_xgb
    else:
        # Jika Prophet gagal, gunakan XGBoost untuk visualisasi juga
        return df_xgb, total_xgb

def hitung_metrik_prediksi(df_ramalan):
    if df_ramalan is None or len(df_ramalan) == 0:
        return {'total_unit': 0}
    total_unit = df_ramalan['yhat'].sum()
    return {'total_unit': total_unit}

def prediksi_produk_terlaris(df_asli, hari_prediksi, tgl_terakhir, tahun_mulai=None, tahun_akhir=None, kolom_harga=None):
    df_terfilter = df_asli.copy()
    if tahun_mulai and tahun_akhir:
        df_terfilter = df_terfilter[(df_terfilter['Date'].dt.year >= tahun_mulai) & (df_terfilter['Date'].dt.year <= tahun_akhir)]
    kolom_produk = None
    if 'Product Name' in df_terfilter.columns:
        kolom_produk = 'Product Name'
    else:
        kol_seperti_produk = [kol for kol in df_terfilter.columns if 'product' in kol.lower() or 'produk' in kol.lower()]
        if kol_seperti_produk: kolom_produk = kol_seperti_produk[0]
        else: return None, None, None, None
    
    penjualan_produk = df_terfilter.groupby(kolom_produk)['Units Sold'].sum().sort_values(ascending=False)
    if len(penjualan_produk) == 0: return None, None, None, None
    
    sembilan_produk_teratas = penjualan_produk.head(9)
    prediksi_produk = {}
    produk_historis = {}
    
    for produk in sembilan_produk_teratas.index:
        data_produk = df_terfilter[df_terfilter[kolom_produk] == produk]
        penjualan_harian = data_produk.groupby('Date')['Units Sold'].sum().asfreq('D').fillna(0)
        produk_historis[produk] = penjualan_harian
        
        ramalan, total_terprediksi = prediksi_kombinasi(
            penjualan_harian, hari_prediksi, tgl_terakhir
        )
        prediksi_produk[produk] = {'unit_terprediksi': total_terprediksi, 'ramalan': ramalan}
    
    produk_terlaris_terprediksi = max(prediksi_produk.items(), key=lambda x: x[1]['unit_terprediksi'])
    return prediksi_produk, produk_terlaris_terprediksi, kolom_produk, produk_historis

# --- 5. FUNGSI TAMPILAN ANTARMUKA ---
def render_bagian_unggah_file():
    st.markdown("### Unggah Berkas CSV")
    kol1, kol2 = st.columns([2, 1])
    with kol1:
        berkas_diunggah = st.file_uploader("Pilih berkas CSV", type=['csv'])
        if berkas_diunggah is not None:
            try:
                df_pratinjau = pd.read_csv(berkas_diunggah)
                st.write("Pratinjau data (9 baris pertama):")
                df_tampil = df_pratinjau.head(9).copy()
                df_tampil.insert(0, 'No', range(1, len(df_tampil) + 1))
                st.dataframe(df_tampil, use_container_width=True, hide_index=True)
                if st.button("Simpan Berkas", type="primary", use_container_width=True):
                    jalur_file = simpan_file_unggahan(berkas_diunggah)
                    if jalur_file:
                        st.success(f"Berkas tersimpan!")
                        st.rerun()
            except Exception as e:
                st.error(f"Galat: {e}")
    with kol2:
        st.write("Berkas Tersimpan")
        berkas_tersimpan = dapatkan_file_tersimpan()
        if berkas_tersimpan:
            for berkas in berkas_tersimpan:
                kol_berkas, kol_tombol = st.columns([3, 1])
                with kol_berkas: st.write(f"{berkas}")
                with kol_tombol:
                    if st.button("Hapus", key=f"hapus_{berkas}"):
                        if hapus_file(berkas): st.rerun()
            berkas_terpilih = st.selectbox("Pilih berkas:", berkas_tersimpan)
            if berkas_terpilih:
                jalur_file = os.path.join(FOLDER_UPLOAD, berkas_terpilih)
                tgl_min, tgl_maks, total_baris, tgl_unik, total_penjualan, tahun_tersedia = dapatkan_info_file(jalur_file)
                if tgl_min and tgl_maks:
                    st.info(f"Data: {tgl_min} s/d {tgl_maks} | {total_baris} baris")
                    if total_penjualan: st.info(f"Total penjualan: {format_angka(total_penjualan)} unit")
                    st.session_state['tahun_tersedia'] = tahun_tersedia
                    if st.button("Mulai Analisis", type="primary", use_container_width=True):
                        st.session_state['berkas_terpilih'] = berkas_terpilih
                        st.session_state['jalur_berkas'] = jalur_file
                        st.session_state['analisis_dikonfirmasi'] = True
                        st.rerun()
                else:
                    st.error("Tidak dapat membaca informasi berkas")
        else:
            st.info("Belum ada berkas tersimpan")

def render_bagian_unggah_aktual():
    st.markdown("### Unggah Data Aktual 2025")
    kol1, kol2 = st.columns([2, 1])
    with kol1:
        berkas_diunggah = st.file_uploader("Pilih berkas CSV aktual 2025", type=['csv'], key='upload_aktual')
        if berkas_diunggah is not None:
            try:
                df_pratinjau = pd.read_csv(berkas_diunggah)
                st.write("Pratinjau data (9 baris pertama):")
                df_tampil = df_pratinjau.head(9).copy()
                df_tampil.insert(0, 'No', range(1, len(df_tampil) + 1))
                st.dataframe(df_tampil, use_container_width=True, hide_index=True)
                if st.button("Simpan Berkas Aktual", type="primary", use_container_width=True, key='simpan_aktual'):
                    jalur_file = simpan_file_unggahan(berkas_diunggah, FOLDER_AKTUAL)
                    if jalur_file:
                        st.success(f"Berkas aktual tersimpan!")
                        st.session_state['file_aktual'] = jalur_file
                        st.rerun()
            except Exception as e:
                st.error(f"Galat: {e}")
    with kol2:
        st.write("Berkas Aktual Tersimpan")
        berkas_tersimpan = dapatkan_file_tersimpan(FOLDER_AKTUAL)
        if berkas_tersimpan:
            for berkas in berkas_tersimpan:
                kol_berkas, kol_tombol = st.columns([3, 1])
                with kol_berkas: st.write(f"{berkas}")
                with kol_tombol:
                    if st.button("Hapus", key=f"hapus_aktual_{berkas}"):
                        if hapus_file(berkas, FOLDER_AKTUAL): 
                            st.session_state['file_aktual'] = None
                            st.rerun()
            berkas_terpilih = st.selectbox("Pilih berkas aktual:", berkas_tersimpan, key='select_aktual')
            if berkas_terpilih:
                jalur_file = os.path.join(FOLDER_AKTUAL, berkas_terpilih)
                st.session_state['file_aktual'] = jalur_file
                st.info(f"File aktual siap: {berkas_terpilih}")
        else:
            st.info("Belum ada berkas aktual tersimpan")

def render_bagian_kpi(data_penjualan, df_asli, tahun_mulai, tahun_akhir):
    st.markdown("### Ukuran Kinerja - Data Riwayat Penjualan")
    st.caption(f"Periode: {tahun_mulai} - {tahun_akhir}")
    total_unit = data_penjualan['Units Sold'].sum()
    st.metric("Total Unit Terjual", format_angka(total_unit))

def render_bagian_kpi_prediksi(metrik_prediksi, hari_prediksi, tahun_prediksi, bulan_prediksi=None):
    st.markdown("### Metrik Kinerja - Prediksi")
    if bulan_prediksi:
        nama_bulan = dapatkan_nama_bulan(bulan_prediksi)
        st.caption(f"Periode Prediksi: {nama_bulan} {tahun_prediksi}")
    else:
        st.caption(f"Periode Prediksi: Tahun {tahun_prediksi}")
    
    total_unit_bulat = int(round(metrik_prediksi['total_unit']))
    
    st.markdown(f"""
    <div style="background-color: #E8F0FE; padding: 20px; border-radius: 8px; border-left: 5px solid #1E3A8A;">
        <p style="color: #1E3A8A; margin: 0; font-size: 16px;">Prediksi Total Unit (9 Produk Terlaris)</p>
        <h1 style="color: #1E3A8A; margin: 10px 0;">{format_angka_dengan_titik(total_unit_bulat)}</h1>
    </div>
    """, unsafe_allow_html=True)

def render_evaluasi_model(prediksi_produk, df_asli, kolom_produk):
    st.markdown("## Perhitungan Mean Absolute Percentage Error (MAPE) dan Root Mean Square Error (RMSE)")
    st.markdown("Untuk mengukur tingkat akurasi hasil prediksi sistem, digunakan dua metrik evaluasi yaitu Mean Absolute Percentage Error (MAPE) dan Root Mean Square Error (RMSE) berdasarkan data perbandingan.")
    
    st.markdown("### Unggah Data Aktual")
    st.markdown("Unggah data aktual untuk mengevaluasi kinerja model prediksi.")
    
    render_bagian_unggah_aktual()
    
    if 'file_aktual' in st.session_state and st.session_state['file_aktual']:
        try:
            df_aktual = pd.read_csv(st.session_state['file_aktual'])
            
            if 'Date' not in df_aktual.columns:
                kol_seperti_tanggal = [kol for kol in df_aktual.columns if 'date' in kol.lower() or 'tanggal' in kol.lower()]
                if kol_seperti_tanggal:
                    df_aktual = df_aktual.rename(columns={kol_seperti_tanggal[0]: 'Date'})
                else:
                    st.error("Kolom Tanggal tidak ditemukan pada data aktual")
                    return
            
            df_aktual['Date'] = pd.to_datetime(df_aktual['Date'])
            df_aktual_2025 = df_aktual[df_aktual['Date'].dt.year == 2025]
            
            if kolom_produk is None:
                if 'Product Name' in df_aktual_2025.columns:
                    kolom_produk = 'Product Name'
                else:
                    kol_seperti_produk = [kol for kol in df_aktual_2025.columns if 'product' in kol.lower() or 'produk' in kol.lower()]
                    if kol_seperti_produk:
                        kolom_produk = kol_seperti_produk[0]
                    else:
                        st.error("Kolom produk tidak ditemukan pada data aktual")
                        return
            
            if 'Units Sold' not in df_aktual_2025.columns:
                kol_seperti_unit = [kol for kol in df_aktual_2025.columns if 'unit' in kol.lower()]
                if kol_seperti_unit:
                    df_aktual_2025 = df_aktual_2025.rename(columns={kol_seperti_unit[0]: 'Units Sold'})
                else:
                    st.error("Kolom Unit Terjual tidak ditemukan pada data aktual")
                    return
            
            data_aktual_per_produk = df_aktual_2025.groupby(kolom_produk)['Units Sold'].sum()
            
            data_prediksi_per_produk = {}
            for produk, data in prediksi_produk.items():
                data_prediksi_per_produk[produk] = data['unit_terprediksi']
            
            produk_aktual = set(data_aktual_per_produk.index)
            produk_prediksi = set(data_prediksi_per_produk.keys())
            produk_interseksi = produk_aktual.intersection(produk_prediksi)
            
            if len(produk_interseksi) == 0:
                st.warning("Tidak ada produk yang cocok antara data aktual dan data prediksi.")
                return
            
            data_aktual_terfilter = {}
            data_prediksi_terfilter = {}
            for produk in produk_interseksi:
                data_aktual_terfilter[produk] = data_aktual_per_produk[produk]
                data_prediksi_terfilter[produk] = data_prediksi_per_produk[produk]
            
            y_aktual = np.array(list(data_aktual_terfilter.values()))
            y_prediksi = np.array(list(data_prediksi_terfilter.values()))
            
            n = len(produk_interseksi)
            
            # --- TABEL PERBANDINGAN DATA AKTUAL DAN HASIL PREDIKSI ---
            st.markdown("---")
            st.markdown("### Tabel Perbandingan Data Aktual dan Hasil Prediksi (9 Produk Terlaris)")
            
            data_tabel = []
            total_aktual = 0
            total_prediksi = 0
            total_selisih = 0
            
            for produk in produk_interseksi:
                aktual = data_aktual_terfilter[produk]
                prediksi = data_prediksi_terfilter[produk]
                selisih = aktual - prediksi
                total_aktual += aktual
                total_prediksi += prediksi
                total_selisih += selisih
                
                data_tabel.append({
                    'No': len(data_tabel) + 1,
                    'Nama Produk': produk,
                    'Aktual 2025': format_angka_dengan_titik(aktual),
                    'Prediksi 2025': format_angka_dengan_titik(prediksi),
                    'Selisih (Aktual - Prediksi)': format_angka_selisih(selisih)
                })
            
            # Tambahkan baris total
            data_tabel.append({
                'No': '',
                'Nama Produk': 'Total',
                'Aktual 2025': format_angka_dengan_titik(total_aktual),
                'Prediksi 2025': format_angka_dengan_titik(total_prediksi),
                'Selisih (Aktual - Prediksi)': format_angka_selisih(total_selisih)
            })
            
            df_tabel = pd.DataFrame(data_tabel)
            st.dataframe(df_tabel, use_container_width=True, hide_index=True)
            
            # --- a. Mean Absolute Percentage Error (MAPE) ---
            st.markdown("---")
            st.markdown("### a. Mean Absolute Percentage Error (MAPE)")
            
            st.markdown("**Rumus MAPE:**")
            st.latex(r"MAPE = \frac{\sum_{t=1}^n \left| \left( \frac{A_t - F_t}{A_t} \right) \right| \times 100}{n}")
            
            st.markdown("**Keterangan:**")
            st.markdown("- n = jumlah produk yang dievaluasi (9 produk)")
            st.markdown("- A_t = nilai aktual penjualan pada produk ke-t")
            st.markdown("- F_t = nilai prediksi penjualan pada produk ke-t")
            
            st.markdown("**Perhitungan MAPE:**")
            
            detail_mape = []
            mape_values = []
            
            for produk in produk_interseksi:
                aktual = data_aktual_terfilter[produk]
                prediksi = data_prediksi_terfilter[produk]
                selisih = aktual - prediksi
                # Hitung MAPE dan LANGSUNG BULATKAN ke 2 desimal
                mape_produk = round(abs(selisih / aktual) * 100, 2) if aktual > 0 else 0
                mape_values.append(mape_produk)
                
                detail_mape.append({
                    'produk': produk,
                    'aktual': aktual,
                    'prediksi': prediksi,
                    'selisih': selisih,
                    'mape': mape_produk
                })
            
            for d in detail_mape:
                st.latex(rf"""
                \text{{{d['produk']}}} = \frac{{|{format_angka_dengan_titik(d['aktual'])} - {format_angka_dengan_titik(d['prediksi'])}|}}{{{format_angka_dengan_titik(d['aktual'])}}} \times 100 = \frac{{|{format_angka_dengan_titik(d['selisih'])}|}}{{{format_angka_dengan_titik(d['aktual'])}}} \times 100 = {format_angka_indonesia(d['mape'], 2)}\%
                """)
            
            # Hitung total menggunakan nilai yang SUDAH DIBULATKAN ke 2 desimal
            total_mape = sum(mape_values)
            mape_akhir = round(total_mape / n, 2) if n > 0 else 0
            
            mape_str = " + ".join([f"{format_angka_indonesia(v, 2)}\%" for v in mape_values])
            
            st.latex(rf"""
            \text{{MAPE}} = \frac{{{mape_str}}}{{{n}}} = \frac{{{format_angka_indonesia(total_mape, 2)}\%}}{{{n}}} = {format_angka_indonesia(mape_akhir, 2)}\%
            """)
            
            st.markdown("**Interpretasi Nilai MAPE:**")
            st.markdown("""
            | Rentang | Kategori |
            |---------|----------|
            | < 10% | Sangat Akurat |
            | 10% - 20% | Akurat |
            | 20% - 50% | Cukup Baik |
            | > 50% | Tidak Akurat |
            """)
            
            interpretasi = interpretasi_mape(mape_akhir)
            st.markdown(f"""
            <div style="background-color: #E8F0FE; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A; margin-top: 10px;">
                <p style="color: #1E3A8A; margin: 0; font-size: 14px;">
                    Berdasarkan hasil perhitungan, diperoleh nilai MAPE sebesar <strong>{format_angka_indonesia(mape_akhir, 2)}%</strong>. 
                    Nilai tersebut berada pada rentang <strong>{'< 10%' if mape_akhir < 10 else '10% - 20%' if mape_akhir < 20 else '20% - 50%' if mape_akhir < 50 else '> 50%'}</strong>, 
                    sehingga dapat disimpulkan bahwa model prediksi yang digunakan tergolong <strong>{interpretasi}</strong> 
                    dalam memperkirakan penjualan produk.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- b. Root Mean Squared Error (RMSE) ---
            st.markdown("---")
            st.markdown("### b. Root Mean Squared Error (RMSE)")
            
            st.markdown("**Rumus RMSE:**")
            st.latex(r"RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}")
            
            st.markdown("**Keterangan:**")
            st.markdown("- n = jumlah produk yang dievaluasi (9 produk)")
            st.markdown("- y_i = nilai aktual penjualan pada produk ke-i")
            st.markdown("- ŷ_i = nilai prediksi penjualan pada produk ke-i")
            
            st.markdown("**Perhitungan RMSE:**")
            
            # BULATKAN NILAI TERLEBIH DAHULU UNTUK PERHITUNGAN YANG AKURAT
            detail_rmse = []
            squared_errors = []
            
            for produk in produk_interseksi:
                aktual = data_aktual_terfilter[produk]
                prediksi = data_prediksi_terfilter[produk]
                
                # Bulatkan nilai ke integer terdekat
                aktual_bulat = int(round(aktual))
                prediksi_bulat = int(round(prediksi))
                selisih_bulat = aktual_bulat - prediksi_bulat
                selisih_kuadrat_bulat = selisih_bulat ** 2
                
                squared_errors.append(selisih_kuadrat_bulat)
                
                detail_rmse.append({
                    'produk': produk,
                    'aktual': aktual_bulat,
                    'prediksi': prediksi_bulat,
                    'selisih': selisih_bulat,
                    'selisih_kuadrat': selisih_kuadrat_bulat
                })
            
            for d in detail_rmse:
                st.latex(rf"""
                \text{{{d['produk']}}} = ({format_angka_dengan_titik(d['aktual'])} - {format_angka_dengan_titik(d['prediksi'])})^2 = ({format_angka_dengan_titik(d['selisih'])})^2 = {format_angka_dengan_titik(d['selisih_kuadrat'])}
                """)
            
            total_squared = sum(squared_errors)
            mse = total_squared / n if n > 0 else 0
            rmse = np.sqrt(mse)
            
            squared_sum_str = " + ".join([f"{format_angka_dengan_titik(v)}" for v in squared_errors])
            
            st.latex(rf"""
            \text{{MSE}} = \frac{{{squared_sum_str}}}{{{n}}} = \frac{{{format_angka_dengan_titik(total_squared)}}}{{{n}}} = {format_angka_indonesia(mse, 2)}
            """)
            
            # PEMBULATAN RMSE MENJADI BILANGAN BULAT (TANPA DESIMAL)
            rmse_bulat = int(round(rmse))
            
            st.latex(rf"""
            \text{{RMSE}} = \sqrt{{{format_angka_indonesia(mse, 2)}}} = {format_angka_dengan_titik(rmse_bulat)} \text{{ unit}}
            """)
            
            # Ringkasan
            st.markdown("---")
            st.markdown("### Ringkasan Metrik Evaluasi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: #E8F0FE; padding: 20px; border-radius: 8px; border-left: 5px solid #1E3A8A;">
                    <p style="color: #1E3A8A; margin: 0; font-size: 14px; font-weight: bold;">MAPE (Mean Absolute Percentage Error)</p>
                    <h2 style="color: #1E3A8A; margin: 10px 0;">{format_angka_indonesia(mape_akhir, 2)}%</h2>
                    <p style="color: #555; margin: 5px 0 0 0; font-size: 13px;">Kategori: {interpretasi}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                rata_rata_aktual = np.mean(y_aktual) if len(y_aktual) > 0 else 0
                interpretasi_rmse_text = interpretasi_rmse(rmse, rata_rata_aktual)
                st.markdown(f"""
                <div style="background-color: #E8F0FE; padding: 20px; border-radius: 8px; border-left: 5px solid #1E3A8A;">
                    <p style="color: #1E3A8A; margin: 0; font-size: 14px; font-weight: bold;">RMSE (Root Mean Squared Error)</p>
                    <h2 style="color: #1E3A8A; margin: 10px 0;">{format_angka_dengan_titik(rmse_bulat)} unit</h2>
                    <p style="color: #555; margin: 5px 0 0 0; font-size: 13px;">{interpretasi_rmse_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Kesimpulan RMSE dengan nilai aktual
            st.markdown(f"""
            <div style="background-color: #E8F0FE; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A; margin-top: 10px;">
                <p style="color: #1E3A8A; margin: 0; font-size: 14px;">
                    Berdasarkan hasil perhitungan, diperoleh nilai RMSE sebesar <strong>{format_angka_dengan_titik(rmse_bulat)}</strong> unit. 
                    Nilai ini menunjukkan bahwa model prediksi memiliki tingkat kesalahan sebesar {format_angka_dengan_titik(rmse_bulat)} unit 
                    dari nilai aktual penjualan, yang termasuk dalam kategori <strong>{interpretasi_rmse_text}</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Galat saat membaca data aktual: {e}")
            import traceback
            st.error(traceback.format_exc())

def render_tab_produk(df_asli, hari_prediksi, tgl_akhir_pred, tahun_prediksi, tgl_terakhir, tahun_mulai=None, kolom_harga=None, tahun_akhir=None, bulan_prediksi=None):
    st.markdown("### Analisis Produk")
    
    kolom_produk = None
    if 'Product Name' in df_asli.columns:
        kolom_produk = 'Product Name'
    else:
        kol_seperti_produk = [kol for kol in df_asli.columns if 'product' in kol.lower() or 'produk' in kol.lower()]
        if kol_seperti_produk: kolom_produk = kol_seperti_produk[0]
    
    if kolom_produk is None:
        st.warning("Kolom produk tidak ditemukan")
        return
    
    with st.spinner("Memprediksi produk terlaris..."):
        prediksi_produk, produk_terlaris, kolom_produk, produk_historis = prediksi_produk_terlaris(
            df_asli, hari_prediksi, tgl_terakhir, tahun_mulai, 
            tahun_akhir, kolom_harga
        )
    
    if prediksi_produk is None:
        st.warning("Tidak dapat melakukan analisis produk")
        return
    
    total_unit_9_produk = sum(data['unit_terprediksi'] for data in prediksi_produk.values())
    st.session_state['total_prediksi_9_produk'] = total_unit_9_produk
    st.session_state['prediksi_produk'] = prediksi_produk
    st.session_state['df_asli'] = df_asli
    st.session_state['kolom_produk'] = kolom_produk
    
    st.markdown("#### Prediksi 9 Produk")
    data_pred = []
    for produk, data in prediksi_produk.items():
        data_pred.append({'Produk': produk, 'PREDIKSI PENJUALAN': data['unit_terprediksi']})
    df_pred = pd.DataFrame(data_pred).sort_values('PREDIKSI PENJUALAN', ascending=False)
    df_pred.insert(0, 'No', range(1, len(df_pred) + 1))
    df_pred['PREDIKSI PENJUALAN'] = df_pred['PREDIKSI PENJUALAN'].apply(format_angka)
    st.dataframe(df_pred, use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    <div style="background-color: #E8F0FE; padding: 20px; border-radius: 8px; border-left: 5px solid #1E3A8A; margin: 20px 0;">
        <p style="color: #1E3A8A; margin: 0; font-size: 14px; font-weight: bold;">PRODUK TERLARIS (PREDIKSI)</p>
        <h2 style="color: #1E3A8A; margin: 5px 0;">{produk_terlaris[0]}</h2>
        <p style="color: #1E3A8A; margin: 10px 0 0 0; font-size: 18px;">Prediksi Total Penjualan: <strong>{format_angka(produk_terlaris[1]['unit_terprediksi'])} unit</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    daftar_produk = list(prediksi_produk.keys())
    produk_dipilih = st.selectbox(
        "Pilih Produk untuk Melihat Grafik Prediksi:",
        options=daftar_produk,
        index=daftar_produk.index(produk_terlaris[0])
    )
    
    if produk_dipilih and prediksi_produk[produk_dipilih]['ramalan'] is not None:
        st.markdown(f"#### Grafik Prediksi - {produk_dipilih}")
        
        ramalan_pt = prediksi_produk[produk_dipilih]['ramalan'].copy()
        historis_pt = produk_historis[produk_dipilih]
        historis_pt = historis_pt[historis_pt.index <= pd.to_datetime(tgl_terakhir)]
        
        if tahun_akhir:
            tgl_akhir_tahun_analisis = pd.to_datetime(f"{tahun_akhir}-12-31")
        else:
            tgl_akhir_tahun_analisis = pd.to_datetime(tgl_terakhir)
        
        historis_pt = historis_pt[historis_pt.index <= tgl_akhir_tahun_analisis]
        
        prediksi_setelah_batas = ramalan_pt[ramalan_pt['ds'] > tgl_akhir_tahun_analisis]
        
        if bulan_prediksi:
            tgl_akhir_bulan = datetime(tahun_prediksi, bulan_prediksi, calendar.monthrange(tahun_prediksi, bulan_prediksi)[1])
            prediksi_setelah_batas = prediksi_setelah_batas[prediksi_setelah_batas['ds'] <= tgl_akhir_bulan]
        else:
            prediksi_setelah_batas = prediksi_setelah_batas[prediksi_setelah_batas['ds'].dt.year <= tahun_prediksi]
        
        if len(prediksi_setelah_batas) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=historis_pt.index, y=historis_pt.values,
                mode='lines', name='Data Historis',
                line=dict(color='#FFA500', width=2)
            ))
            
            if len(historis_pt) > 0:
                tgl_historis_terakhir = historis_pt.index[-1]
                nilai_historis_terakhir = historis_pt.values[-1]
                
                tgl_prediksi_full = [tgl_historis_terakhir] + list(prediksi_setelah_batas['ds'])
                nilai_prediksi_full = [nilai_historis_terakhir] + list(prediksi_setelah_batas['yhat'])
                
                fig.add_trace(go.Scatter(
                    x=tgl_prediksi_full, y=nilai_prediksi_full,
                    mode='lines', name='Prediksi',
                    line=dict(color='#1E3A8A', width=3)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=prediksi_setelah_batas['ds'], y=prediksi_setelah_batas['yhat'],
                    mode='lines', name='Prediksi',
                    line=dict(color='#1E3A8A', width=3)
                ))
            
            fig.add_shape(type="line", x0=tgl_akhir_tahun_analisis, x1=tgl_akhir_tahun_analisis, y0=0, y1=1, yref="paper",
                         line=dict(color="gray", width=1.5, dash="dot"))
            fig.add_annotation(x=tgl_akhir_tahun_analisis, y=1.02, yref="paper", text="Akhir Tahun Analisis",
                             showarrow=False, font=dict(size=11, color="gray"))
            
            fig.update_layout(
                title=f"Prediksi Penjualan {produk_dipilih}",
                xaxis_title="Tanggal", yaxis_title="Jumlah Unit",
                plot_bgcolor='white', hovermode='x', height=400
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#EEE')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEE')
            st.plotly_chart(fig, use_container_width=True)
            
            # --- EVALUASI MODEL ---
            render_evaluasi_model(prediksi_produk, df_asli, kolom_produk)
            
    elif produk_dipilih:
        st.warning(f"Data prediksi untuk {produk_dipilih} tidak tersedia.")

# --- 6. FUNGSI HALAMAN ABOUT ---
def render_halaman_about():
    st.markdown("---")
    st.header("Tentang Aplikasi")
    
    st.markdown("""
    <div style="background-color: #F8F9FA; padding: 30px; border-radius: 12px; margin-bottom: 30px;">
        <h2 style="color: #1E3A8A; margin-top: 0;">Sistem Prediksi Tren Penjualan E-Commerce</h2>
        <p style="font-size: 16px; line-height: 1.8; color: #333;">
            Aplikasi ini dirancang untuk membantu pelaku bisnis e-commerce dalam menganalisis data penjualan historis 
            dan menghasilkan prediksi tren penjualan produk. Sistem ini menyediakan dasbor interaktif yang memungkinkan 
            pengguna untuk mengunggah data penjualan dalam format CSV, memilih rentang waktu analisis, dan melihat 
            hasil prediksi untuk produk-produk terlaris. Aplikasi ini menggunakan algoritma XGBoost sebagai model utama 
            untuk melakukan prediksi penjualan berdasarkan pola dan tren dari data historis.
        </p>
        <p style="font-size: 16px; line-height: 1.8; color: #333; margin-bottom: 0;">
            Fitur utama yang tersedia dalam aplikasi ini meliputi:
        </p>
        <ul style="font-size: 15px; line-height: 1.8; color: #444; padding-left: 20px;">
            <li>Unggah dan kelola berkas data penjualan (CSV)</li>
            <li>Pemilihan rentang tahun analisis sesuai kebutuhan</li>
            <li>Visualisasi data historis dan prediksi dalam grafik interaktif</li>
            <li>Identifikasi 9 produk dengan prediksi penjualan tertinggi</li>
            <li>Prediksi penjualan per produk dengan opsi pemilihan tahun dan bulan</li>
            <li>Ekspor dan penyimpanan berkas data yang telah diunggah</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 25px; border-radius: 10px; border: 1px solid #E0E0E0; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E3A8A; width: 4px; height: 30px; margin-right: 15px; border-radius: 2px;"></div>
            <h3 style="color: #1E3A8A; margin: 0; font-size: 22px;">Manajemen Stok</h3>
        </div>
        <p style="font-size: 15px; line-height: 1.8; color: #444; margin-bottom: 20px;">
            Aplikasi ini membantu dalam perencanaan stok dengan menyediakan data prediksi penjualan untuk produk-produk terlaris. 
            Pengguna dapat melihat estimasi jumlah unit yang akan terjual pada periode tertentu sehingga dapat menyesuaikan 
            jumlah persediaan barang secara lebih tepat.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Prediksi Kebutuhan Stok</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Menampilkan perkiraan jumlah unit yang akan terjual berdasarkan analisis tren penjualan historis menggunakan model XGBoost.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Identifikasi Produk Unggulan</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Menampilkan 9 produk dengan volume penjualan tertinggi untuk membantu prioritas pengadaan stok.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Analisis Periode Tertentu</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Memungkinkan pemilihan rentang tahun dan bulan prediksi untuk perencanaan stok jangka pendek maupun panjang.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Visualisasi Data Stok</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Grafik interaktif yang menampilkan data historis dan prediksi untuk memudahkan analisis kebutuhan stok.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 25px; border-radius: 10px; border: 1px solid #E0E0E0; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E3A8A; width: 4px; height: 30px; margin-right: 15px; border-radius: 2px;"></div>
            <h3 style="color: #1E3A8A; margin: 0; font-size: 22px;">Alokasi Sales Person (SP)</h3>
        </div>
        <p style="font-size: 15px; line-height: 1.8; color: #444; margin-bottom: 20px;">
            Data prediksi penjualan dari aplikasi ini dapat digunakan sebagai dasar dalam menentukan alokasi tenaga penjualan. 
            Dengan mengetahui produk mana yang diprediksi akan mengalami peningkatan permintaan, manajemen dapat 
            menyesuaikan penempatan sales person untuk fokus pada produk atau wilayah dengan potensi penjualan tertinggi.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Informasi Produk Prioritas</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Menyediakan daftar produk terlaris yang dapat menjadi acuan dalam mengarahkan fokus tenaga penjualan.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Analisis Tren Permintaan</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Menampilkan grafik tren penjualan per produk untuk membantu identifikasi produk yang memerlukan dukungan penjualan lebih.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Data Dasar Pengambilan Keputusan</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Hasil prediksi memberikan gambaran kuantitatif yang dapat digunakan sebagai pertimbangan dalam strategi alokasi SP.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Pemantauan Kinerja Produk</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Memungkinkan pemantauan performa penjualan produk dari waktu ke waktu untuk evaluasi efektivitas alokasi SP.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 25px; border-radius: 10px; border: 1px solid #E0E0E0; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="background-color: #1E3A8A; width: 4px; height: 30px; margin-right: 15px; border-radius: 2px;"></div>
            <h3 style="color: #1E3A8A; margin: 0; font-size: 22px;">Target Penjualan</h3>
        </div>
        <p style="font-size: 15px; line-height: 1.8; color: #444; margin-bottom: 20px;">
            Aplikasi ini menyediakan hasil prediksi penjualan yang dapat dijadikan acuan dalam menetapkan target penjualan 
            yang realistis. Data prediksi dihitung berdasarkan pola penjualan historis sehingga target yang ditetapkan 
            memiliki dasar analisis yang terukur.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Target Berbasis Data Historis</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Prediksi dihasilkan dari analisis data penjualan masa lalu sehingga memberikan estimasi yang terukur.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Proyeksi Fleksibel</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Pengguna dapat memilih periode prediksi (tahunan atau bulanan) sesuai dengan kebutuhan penetapan target.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Perbandingan Historis vs Prediksi</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Grafik menampilkan data historis dan prediksi dalam satu tampilan untuk memudahkan analisis gap.
                </p>
            </div>
            <div style="background-color: #F0F4FF; padding: 15px; border-radius: 8px;">
                <h4 style="color: #1E3A8A; margin: 0 0 8px 0; font-size: 16px;">Ringkasan Metrik Prediksi</h4>
                <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.6;">
                    Menampilkan total prediksi unit untuk 9 produk terlaris sebagai gambaran umum target penjualan.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #1E3A8A; padding: 25px; border-radius: 10px; margin-top: 30px; color: white;">
        <h3 style="color: white; margin-top: 0; text-align: center;">Fitur Utama Aplikasi</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 20px;">
            <div style="text-align: center; padding: 15px; background-color: rgba(255,255,255,0.1); border-radius: 8px;">
                <p style="font-size: 20px; font-weight: bold; margin: 0 0 8px 0;">Unggah Data CSV</p>
                <p style="font-size: 14px; margin: 0; opacity: 0.9;">Mengunggah dan menyimpan berkas data penjualan untuk dianalisis</p>
            </div>
            <div style="text-align: center; padding: 15px; background-color: rgba(255,255,255,0.1); border-radius: 8px;">
                <p style="font-size: 20px; font-weight: bold; margin: 0 0 8px 0;">Prediksi Penjualan</p>
                <p style="font-size: 14px; margin: 0; opacity: 0.9;">Menghasilkan prediksi menggunakan algoritma XGBoost</p>
            </div>
            <div style="text-align: center; padding: 15px; background-color: rgba(255,255,255,0.1); border-radius: 8px;">
                <p style="font-size: 20px; font-weight: bold; margin: 0 0 8px 0;">Visualisasi Interaktif</p>
                <p style="font-size: 14px; margin: 0; opacity: 0.9;">Grafik interaktif untuk analisis data historis dan prediksi</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 7. FUNGSI UTAMA ---
def utama():
    st.title("Prediksi Tren Penjualan Produk E-Commerce Menggunakan Teknik Analisis Big Data")
    
    if 'berkas_terpilih' not in st.session_state: st.session_state['berkas_terpilih'] = None
    if 'jalur_berkas' not in st.session_state: st.session_state['jalur_berkas'] = None
    if 'tahun_mulai_analisis' not in st.session_state: st.session_state['tahun_mulai_analisis'] = None
    if 'tahun_akhir_analisis' not in st.session_state: st.session_state['tahun_akhir_analisis'] = None
    if 'analisis_dikonfirmasi' not in st.session_state: st.session_state['analisis_dikonfirmasi'] = False
    if 'total_prediksi_9_produk' not in st.session_state: st.session_state['total_prediksi_9_produk'] = 0
    if 'tahun_tersedia' not in st.session_state: st.session_state['tahun_tersedia'] = []
    if 'bulan_prediksi' not in st.session_state: st.session_state['bulan_prediksi'] = None
    if 'halaman_aktif' not in st.session_state: st.session_state['halaman_aktif'] = 'analisis'
    if 'file_aktual' not in st.session_state: st.session_state['file_aktual'] = None
    if 'prediksi_produk' not in st.session_state: st.session_state['prediksi_produk'] = None
    if 'df_asli' not in st.session_state: st.session_state['df_asli'] = None
    if 'kolom_produk' not in st.session_state: st.session_state['kolom_produk'] = None
    
    with st.sidebar:
        st.header("Navigasi")
        
        if st.button("Analisis Prediksi", use_container_width=True, 
                     type="primary" if st.session_state['halaman_aktif'] == 'analisis' else "secondary"):
            st.session_state['halaman_aktif'] = 'analisis'
            st.rerun()
        
        if st.button("Evaluasi Model", use_container_width=True,
                     type="primary" if st.session_state['halaman_aktif'] == 'evaluasi' else "secondary"):
            st.session_state['halaman_aktif'] = 'evaluasi'
            st.rerun()
        
        if st.button("Tentang Aplikasi", use_container_width=True,
                     type="primary" if st.session_state['halaman_aktif'] == 'about' else "secondary"):
            st.session_state['halaman_aktif'] = 'about'
            st.rerun()
        
        st.markdown("---")
        
        if st.session_state['halaman_aktif'] == 'analisis':
            st.header("Pengaturan")
            if st.session_state['berkas_terpilih']:
                st.success(f"Berkas: {st.session_state['berkas_terpilih']}")
                
                if 'tahun_tersedia' in st.session_state and st.session_state['tahun_tersedia']:
                    tahun_tersedia = st.session_state['tahun_tersedia']
                    st.markdown("### Pilih Rentang Tahun Analisis")
                    kol_mulai, kol_akhir = st.columns(2)
                    with kol_mulai:
                        tahun_mulai = st.selectbox("Tahun Mulai", options=tahun_tersedia, 
                                                  index=0 if not st.session_state['tahun_mulai_analisis'] else tahun_tersedia.index(st.session_state['tahun_mulai_analisis']) if st.session_state['tahun_mulai_analisis'] in tahun_tersedia else 0,
                                                  key='tahun_mulai_sidebar')
                    with kol_akhir:
                        tahun_akhir_valid = [y for y in tahun_tersedia if y >= tahun_mulai]
                        tahun_akhir = st.selectbox("Tahun Akhir", options=tahun_akhir_valid, 
                                                  index=len(tahun_akhir_valid)-1 if not st.session_state['tahun_akhir_analisis'] or st.session_state['tahun_akhir_analisis'] not in tahun_akhir_valid else tahun_akhir_valid.index(st.session_state['tahun_akhir_analisis']),
                                                  key='tahun_akhir_sidebar')
                    
                    if tahun_mulai > tahun_akhir:
                        st.error("Tahun mulai harus lebih kecil atau sama dengan tahun akhir")
                        tahun_akhir = tahun_mulai
                    
                    st.info(f"Rentang Analisis: {tahun_mulai} - {tahun_akhir}")
                    
                    st.session_state['tahun_mulai_analisis'] = tahun_mulai
                    st.session_state['tahun_akhir_analisis'] = tahun_akhir
                
                st.markdown("---")
                
                if st.button("Ganti Berkas", use_container_width=True):
                    for kunci in ['berkas_terpilih', 'jalur_berkas', 'tahun_mulai_analisis', 'tahun_akhir_analisis', 'analisis_dikonfirmasi', 'total_prediksi_9_produk', 'tahun_tersedia', 'bulan_prediksi', 'file_aktual', 'prediksi_produk', 'df_asli', 'kolom_produk']:
                        st.session_state[kunci] = None if kunci != 'tahun_tersedia' else []
                    st.rerun()
    
    if st.session_state['halaman_aktif'] == 'about':
        render_halaman_about()
        return
    
    if st.session_state['halaman_aktif'] == 'evaluasi':
        if st.session_state.get('prediksi_produk') and st.session_state.get('df_asli') is not None:
            render_evaluasi_model(
                st.session_state['prediksi_produk'], 
                st.session_state['df_asli'], 
                st.session_state.get('kolom_produk')
            )
        else:
            st.warning("Silakan lakukan analisis prediksi terlebih dahulu sebelum mengakses halaman evaluasi model.")
        return
    
    with st.expander("Kelola Berkas CSV", expanded=not st.session_state['berkas_terpilih']):
        render_bagian_unggah_file()
    
    if st.session_state['berkas_terpilih'] and st.session_state['analisis_dikonfirmasi']:
        tahun_mulai_analisis = st.session_state.get('tahun_mulai_analisis')
        tahun_akhir_analisis = st.session_state.get('tahun_akhir_analisis')
        
        if not tahun_mulai_analisis and 'tahun_tersedia' in st.session_state and st.session_state['tahun_tersedia']:
            tahun_mulai_analisis = st.session_state['tahun_tersedia'][0]
            tahun_akhir_analisis = st.session_state['tahun_tersedia'][-1]
            st.session_state['tahun_mulai_analisis'] = tahun_mulai_analisis
            st.session_state['tahun_akhir_analisis'] = tahun_akhir_analisis
        
        with st.spinner("Memproses data..."):
            try:
                data_penjualan, df_asli, _ = muat_dan_siapkan_data(
                    st.session_state['jalur_berkas'],
                    tahun_mulai_analisis,
                    tahun_akhir_analisis
                )
                
                if len(data_penjualan) == 0:
                    st.warning("Tidak ada data dalam rentang tahun yang dipilih")
                    return
                
                kolom_harga = dapatkan_kolom_harga(df_asli, df_asli)
                tgl_terakhir = data_penjualan.index.max().date()
                
                with st.sidebar:
                    st.markdown("---")
                    st.markdown("### Periode Prediksi")
                    
                    tahun_prediksi_min = tahun_akhir_analisis + 1
                    tahun_prediksi_maks = 2100
                    
                    daftar_tahun_prediksi = list(range(tahun_prediksi_min, tahun_prediksi_maks + 1))
                    
                    tahun_prediksi = st.selectbox(
                        "Pilih Tahun Prediksi",
                        options=daftar_tahun_prediksi,
                        index=0,
                        key='tahun_prediksi_sidebar'
                    )
                    
                    daftar_bulan = list(range(1, 13))
                    
                    bulan_prediksi = st.selectbox(
                        "Pilih Bulan Prediksi",
                        options=daftar_bulan,
                        format_func=lambda x: dapatkan_nama_bulan(x),
                        index=None,
                        placeholder="Pilih bulan (opsional)...",
                        key='bulan_prediksi_sidebar'
                    )
                    
                    st.session_state['bulan_prediksi'] = bulan_prediksi
                    
                    if bulan_prediksi:
                        tgl_akhir_bulan = datetime(tahun_prediksi, bulan_prediksi, calendar.monthrange(tahun_prediksi, bulan_prediksi)[1]).date()
                        hari_prediksi = (tgl_akhir_bulan - tgl_terakhir).days
                        
                        if hari_prediksi > 0:
                            st.success(f"Prediksi hingga {format_tanggal_indonesia(tgl_akhir_bulan)}")
                        else:
                            tahun_prediksi += 1
                            tgl_akhir_bulan = datetime(tahun_prediksi, bulan_prediksi, calendar.monthrange(tahun_prediksi, bulan_prediksi)[1]).date()
                            hari_prediksi = (tgl_akhir_bulan - tgl_terakhir).days
                            st.info(f"Prediksi hingga {format_tanggal_indonesia(tgl_akhir_bulan)}")
                    else:
                        tgl_akhir_pred = datetime(tahun_prediksi, 12, 31).date()
                        hari_prediksi = (tgl_akhir_pred - tgl_terakhir).days
                        
                        if hari_prediksi > 0:
                            st.success(f"Prediksi hingga 31 Des {tahun_prediksi}")
                        else:
                            hari_prediksi = 365
                            tahun_prediksi = tgl_akhir_pred.year + 1
                            st.info(f"Prediksi 1 tahun ke depan")
                    
                    if st.button("Ubah Rentang Analisis", use_container_width=True):
                        st.session_state['analisis_dikonfirmasi'] = False
                        st.rerun()
                
                prediksi_produk, _, _, _ = prediksi_produk_terlaris(
                    df_asli, hari_prediksi, tgl_terakhir, 
                    tahun_mulai_analisis,
                    tahun_akhir_analisis, 
                    kolom_harga
                )
                
                if prediksi_produk:
                    total_unit_9_produk = sum(data['unit_terprediksi'] for data in prediksi_produk.values())
                    st.session_state['total_prediksi_9_produk'] = total_unit_9_produk
                else:
                    st.session_state['total_prediksi_9_produk'] = 0
                
                st.session_state['prediksi_produk'] = prediksi_produk
                st.session_state['df_asli'] = df_asli
                st.session_state['kolom_produk'] = kolom_produk
                
                metrik_prediksi = {'total_unit': st.session_state['total_prediksi_9_produk']}
                
                st.markdown("---")
                st.header("Analisis Data Riwayat")
                render_bagian_kpi(data_penjualan, df_asli, tahun_mulai_analisis, tahun_akhir_analisis)
                
                st.markdown("---")
                st.header("Prediksi Data Mendatang")
                render_bagian_kpi_prediksi(metrik_prediksi, hari_prediksi, tahun_prediksi, bulan_prediksi)
                
                st.markdown("---")
                
                render_tab_produk(df_asli, hari_prediksi, tgl_akhir_pred if not bulan_prediksi else tgl_akhir_bulan, 
                                 tahun_prediksi, tgl_terakhir,
                                 tahun_mulai_analisis, kolom_harga, tahun_akhir_analisis, bulan_prediksi)
                
            except Exception as e:
                st.error(f"Galat: {e}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    utama()