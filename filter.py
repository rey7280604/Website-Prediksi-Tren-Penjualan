import pandas as pd

# Baca dataset awal
df = pd.read_csv('Amazon_Sales_Complete.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Daftar produk yang diinginkan
daftar_produk = [
    'Whey Protein', 'Fish Oil', 'Biotin', 'Iron Supplement', 
    'Vitamin C', 'Ashwagandha', 'Magnesium', 'Zinc', 'BCAA'
]

# Filter produk
df = df[df['Product Name'].isin(daftar_produk)]

# Split tahun
df_2020_2024 = df[(df['Date'].dt.year >= 2020) & (df['Date'].dt.year <= 2024)]
df_2025 = df[df['Date'].dt.year == 2025]

# Simpan
df_2020_2024.to_csv('Amazon_Sales_2020-2024.csv', index=False)
df_2025.to_csv('Amazon_Sales_2025.csv', index=False)

print("Selesai!")
print(f"File 2020-2024: {len(df_2020_2024)} baris")
print(f"File 2025: {len(df_2025)} baris")