# ==============================================================================
# MENAMPILKAN DATA AMAZON_SALES_COMPLETE.CSV DALAM POPUP WINDOW
# Menggunakan Tkinter Treeview untuk tampilan tabel interaktif
# ==============================================================================

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime

# --- KONFIGURASI ---
DATA_FILE = 'Amazon_Sales_Complete.csv'

class DataViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 DATA AMAZON SALES COMPLETE")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variabel
        self.df = None
        self.current_view = 'all'
        
        # Cek file
        if not os.path.exists(DATA_FILE):
            messagebox.showerror("Error", f"File '{DATA_FILE}' tidak ditemukan!\n\nPastikan file berada di direktori:\n{os.getcwd()}")
            root.destroy()
            return
        
        # Load data
        self.load_data()
        
        # Setup GUI
        self.setup_gui()
        
        # Tampilkan data
        self.show_all_data()
    
    def load_data(self):
        """Memuat data dari CSV"""
        self.df = pd.read_csv(DATA_FILE)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
    
    def setup_gui(self):
        """Membuat komponen GUI"""
        
        # ===== FRAME ATAS (INFO & TOMBOL) =====
        top_frame = tk.Frame(self.root, bg='#2c3e50', height=120)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)
        
        # Judul
        title_label = tk.Label(top_frame, text="📊 DATA AMAZON SALES COMPLETE", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=5)
        
        # Info dataset
        info_text = f"Total: {len(self.df):,} baris | {self.df['Product Name'].nunique()} produk | {self.df['Category'].nunique()} kategori | {self.df['Date'].min().strftime('%d %b %Y')} s/d {self.df['Date'].max().strftime('%d %b %Y')}"
        info_label = tk.Label(top_frame, text=info_text, 
                             font=('Arial', 10), fg='#bdc3c7', bg='#2c3e50')
        info_label.pack(pady=2)
        
        # ===== FRAME TOMBOL NAVIGASI =====
        nav_frame = tk.Frame(self.root, bg='#ecf0f1', height=50)
        nav_frame.pack(fill=tk.X, pady=2)
        nav_frame.pack_propagate(False)
        
        # Tombol-tombol
        btn_style = {'font': ('Arial', 10, 'bold'), 'width': 20, 'height': 1, 'cursor': 'hand2'}
        
        tk.Button(nav_frame, text="📋 Semua Data", bg='#3498db', fg='white',
                 command=self.show_all_data, **btn_style).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(nav_frame, text="📦 Daftar Produk", bg='#2ecc71', fg='white',
                 command=self.show_product_list, **btn_style).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(nav_frame, text="📂 Daftar Kategori", bg='#e67e22', fg='white',
                 command=self.show_category_list, **btn_style).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(nav_frame, text="📅 Data Per Tahun", bg='#9b59b6', fg='white',
                 command=self.show_yearly_data, **btn_style).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(nav_frame, text="📊 Data Per Bulan", bg='#1abc9c', fg='white',
                 command=self.show_monthly_data, **btn_style).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(nav_frame, text="📈 Statistik", bg='#e74c3c', fg='white',
                 command=self.show_statistics, **btn_style).pack(side=tk.LEFT, padx=5, pady=5)
        
        # ===== FRAME PENCARIAN =====
        search_frame = tk.Frame(self.root, bg='#ecf0f1')
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(search_frame, text="🔍 Cari:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1').pack(side=tk.LEFT, padx=5)
        
        self.search_entry = tk.Entry(search_frame, font=('Arial', 11), width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', self.search_data)
        
        tk.Button(search_frame, text="Cari", bg='#3498db', fg='white',
                 font=('Arial', 10), width=10, command=self.search_data).pack(side=tk.LEFT, padx=5)
        
        tk.Button(search_frame, text="Reset", bg='#95a5a6', fg='white',
                 font=('Arial', 10), width=10, command=self.reset_search).pack(side=tk.LEFT, padx=5)
        
        # Label jumlah data
        self.count_label = tk.Label(search_frame, text="", font=('Arial', 10), bg='#ecf0f1')
        self.count_label.pack(side=tk.RIGHT, padx=10)
        
        # ===== FRAME TABEL (TREEVIEW) =====
        table_frame = tk.Frame(self.root, bg='white')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        y_scroll = tk.Scrollbar(table_frame, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scroll = tk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        self.tree = ttk.Treeview(table_frame, yscrollcommand=y_scroll.set, 
                                 xscrollcommand=x_scroll.set, selectmode='extended')
        
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)
        
        # Style Treeview
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', font=('Arial', 9), rowheight=25)
        style.configure('Treeview.Heading', font=('Arial', 10, 'bold'), background='#2c3e50', foreground='white')
        style.map('Treeview.Heading', background=[('active', '#34495e')])
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # ===== FRAME BAWAH (STATUS) =====
        bottom_frame = tk.Frame(self.root, bg='#2c3e50', height=30)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)
        
        self.status_label = tk.Label(bottom_frame, text="Ready", 
                                     font=('Arial', 9), fg='white', bg='#2c3e50')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    def update_count_label(self, count):
        """Update label jumlah data yang ditampilkan"""
        self.count_label.config(text=f"Menampilkan: {count:,} baris")
    
    def search_data(self, event=None):
        """Fungsi pencarian"""
        query = self.search_entry.get().lower().strip()
        
        if not query:
            self.show_all_data()
            return
        
        # Filter data
        filtered_df = self.df[self.df.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)]
        self.display_dataframe(filtered_df)
        self.status_label.config(text=f"Hasil pencarian: '{query}' - {len(filtered_df):,} data ditemukan")
    
    def reset_search(self):
        """Reset pencarian"""
        self.search_entry.delete(0, tk.END)
        self.show_all_data()
    
    def display_dataframe(self, df, title=None):
        """Menampilkan dataframe ke treeview"""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Reset columns
        self.tree['columns'] = list(df.columns)
        self.tree['show'] = 'headings'
        
        # Set column headings
        for col in df.columns:
            self.tree.heading(col, text=col)
            # Adjust column width
            max_width = max(df[col].astype(str).str.len().max(), len(str(col))) * 8
            self.tree.column(col, width=min(max_width, 200), anchor='center')
        
        # Insert data
        for idx, row in df.iterrows():
            values = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, pd.Timestamp):
                    val = val.strftime('%Y-%m-%d')
                elif isinstance(val, (np.floating, float)):
                    val = f'{val:.2f}'
                elif isinstance(val, np.integer):
                    val = f'{val:,}'
                values.append(val)
            
            # Color alternating rows
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            self.tree.insert('', tk.END, values=values, tags=(tag,))
        
        # Configure row colors
        self.tree.tag_configure('evenrow', background='#f8f9fa')
        self.tree.tag_configure('oddrow', background='white')
        
        # Update count
        self.update_count_label(len(df))
        
        if title:
            self.status_label.config(text=title)
    
    def show_all_data(self):
        """Tampilkan semua data (20 baris pertama)"""
        self.current_view = 'all'
        display_df = self.df.head(100).copy()
        # Format kolom Date
        if 'Date' in display_df.columns:
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        self.display_dataframe(display_df, "Menampilkan 100 baris pertama dari semua data")
    
    def show_product_list(self):
        """Tampilkan daftar produk"""
        self.current_view = 'products'
        
        product_stats = self.df.groupby('Product Name').agg({
            'Units Sold': ['count', 'mean', 'sum'],
            'Revenue': ['mean', 'sum']
        }).round(2)
        
        product_stats.columns = ['Jumlah Data', 'Rata2 Units', 'Total Units', 'Rata2 Revenue', 'Total Revenue']
        product_stats = product_stats.sort_values('Total Units', ascending=False).reset_index()
        
        self.display_dataframe(product_stats, f"Menampilkan {len(product_stats)} produk")
    
    def show_category_list(self):
        """Tampilkan daftar kategori"""
        self.current_view = 'categories'
        
        if 'Category' in self.df.columns:
            cat_stats = self.df.groupby('Category').agg({
                'Units Sold': ['count', 'mean', 'sum'],
                'Revenue': ['mean', 'sum'],
                'Product Name': 'nunique'
            }).round(2)
            
            cat_stats.columns = ['Jumlah Data', 'Rata2 Units', 'Total Units', 
                                'Rata2 Revenue', 'Total Revenue', 'Jumlah Produk']
            cat_stats = cat_stats.sort_values('Total Units', ascending=False).reset_index()
            
            self.display_dataframe(cat_stats, f"Menampilkan {len(cat_stats)} kategori")
    
    def show_yearly_data(self):
        """Tampilkan data per tahun"""
        self.current_view = 'yearly'
        
        yearly_stats = self.df.groupby('Year').agg({
            'Units Sold': ['count', 'mean', 'sum'],
            'Revenue': ['mean', 'sum'],
            'Product Name': 'nunique'
        }).round(2)
        
        yearly_stats.columns = ['Jumlah Data', 'Rata2 Units', 'Total Units', 
                               'Rata2 Revenue', 'Total Revenue', 'Jumlah Produk']
        yearly_stats = yearly_stats.sort_index().reset_index()
        
        self.display_dataframe(yearly_stats, f"Menampilkan data per tahun ({len(yearly_stats)} tahun)")
    
    def show_monthly_data(self):
        """Tampilkan data per bulan"""
        self.current_view = 'monthly'
        
        months_name = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 
                      5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
                      9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}
        
        monthly_stats = self.df.groupby('Month').agg({
            'Units Sold': ['count', 'mean', 'std', 'min', 'max', 'sum'],
            'Revenue': 'sum'
        }).round(2)
        
        monthly_stats.columns = ['Jumlah Data', 'Rata2 Units', 'Std Units', 
                                'Min Units', 'Max Units', 'Total Units', 'Total Revenue']
        monthly_stats = monthly_stats.reset_index()
        monthly_stats['Bulan'] = monthly_stats['Month'].map(months_name)
        
        # Reorder columns
        cols = ['Bulan', 'Month', 'Jumlah Data', 'Rata2 Units', 'Std Units', 
               'Min Units', 'Max Units', 'Total Units', 'Total Revenue']
        monthly_stats = monthly_stats[cols]
        
        self.display_dataframe(monthly_stats, "Menampilkan data per bulan")
    
    def show_statistics(self):
        """Tampilkan statistik deskriptif"""
        self.current_view = 'statistics'
        
        # Statistik Units Sold
        stats_units = self.df['Units Sold'].describe()
        
        stats_data = {
            'Metrik': ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max'],
            'Units Sold': [
                f'{stats_units["count"]:,.0f}',
                f'{stats_units["mean"]:,.2f}',
                f'{stats_units["std"]:,.2f}',
                f'{stats_units["min"]:,.0f}',
                f'{stats_units["25%"]:,.2f}',
                f'{stats_units["50%"]:,.2f}',
                f'{stats_units["75%"]:,.2f}',
                f'{stats_units["max"]:,.0f}'
            ]
        }
        
        if 'Revenue' in self.df.columns:
            stats_rev = self.df['Revenue'].describe()
            stats_data['Revenue'] = [
                f'{stats_rev["count"]:,.0f}',
                f'{stats_rev["mean"]:,.2f}',
                f'{stats_rev["std"]:,.2f}',
                f'{stats_rev["min"]:,.2f}',
                f'{stats_rev["25%"]:,.2f}',
                f'{stats_rev["50%"]:,.2f}',
                f'{stats_rev["75%"]:,.2f}',
                f'{stats_rev["max"]:,.0f}'
            ]
        
        stats_df = pd.DataFrame(stats_data)
        self.display_dataframe(stats_df, "Menampilkan statistik deskriptif")
        
        # Popup tambahan untuk informasi
        self.show_info_popup()
    
    def show_info_popup(self):
        """Tampilkan popup informasi tambahan"""
        popup = tk.Toplevel(self.root)
        popup.title("📊 Informasi Dataset")
        popup.geometry("500x400")
        popup.configure(bg='white')
        
        # Title
        tk.Label(popup, text="INFORMASI DATASET", font=('Arial', 14, 'bold'), 
                bg='#2c3e50', fg='white', pady=10).pack(fill=tk.X)
        
        # Info text
        info_text = f"""
═══════════════════════════════════════
📂 FILE INFORMATION
═══════════════════════════════════════
Nama File        : {DATA_FILE}
Total Baris      : {len(self.df):,}
Total Kolom      : {len(self.df.columns)}

═══════════════════════════════════════
📅 DATE RANGE
═══════════════════════════════════════
Tanggal Awal     : {self.df['Date'].min().strftime('%d %B %Y')}
Tanggal Akhir    : {self.df['Date'].max().strftime('%d %B %Y')}
Total Hari       : {self.df['Date'].nunique():,}

═══════════════════════════════════════
📦 PRODUCTS & CATEGORIES
═══════════════════════════════════════
Jumlah Produk    : {self.df['Product Name'].nunique()}
Jumlah Kategori  : {self.df['Category'].nunique()}

═══════════════════════════════════════
📊 DATA QUALITY
═══════════════════════════════════════
Missing Values   : {self.df.isnull().sum().sum():,}
Duplicate Rows   : {self.df.duplicated().sum():,}

═══════════════════════════════════════
📈 TOP 5 PRODUCTS (by Total Units)
═══════════════════════════════════════
"""
        
        # Top 5 products
        top5 = self.df.groupby('Product Name')['Units Sold'].sum().sort_values(ascending=False).head(5)
        for i, (product, total) in enumerate(top5.items(), 1):
            info_text += f"{i}. {product[:40]}: {total:,.0f} units\n"
        
        info_text += """
═══════════════════════════════════════
📅 DATA PER TAHUN
═══════════════════════════════════════
"""
        
        for year in sorted(self.df['Year'].unique()):
            year_data = self.df[self.df['Year'] == year]
            info_text += f"Tahun {year}: {len(year_data):,} baris ({year_data['Date'].dt.month.nunique()} bulan)\n"
        
        # Text widget
        text_widget = scrolledtext.ScrolledText(popup, font=('Consolas', 10), 
                                                bg='white', fg='#2c3e50', wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        tk.Button(popup, text="Tutup", bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                 command=popup.destroy, cursor='hand2').pack(pady=10)

def main():
    """Fungsi utama"""
    root = tk.Tk()
    
    # Cek file
    if not os.path.exists(DATA_FILE):
        root.withdraw()  # Sembunyikan window utama
        messagebox.showerror("❌ Error", 
                             f"File '{DATA_FILE}' tidak ditemukan!\n\n"
                             f"Pastikan file berada di direktori:\n{os.getcwd()}\n\n"
                             f"Jalankan 'fill_missing_dates.py' terlebih dahulu.")
        return
    
    app = DataViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()