import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox

# --- Pengaturan Tampilan Aplikasi ---
ctk.set_appearance_mode("System") 
ctk.set_default_color_theme("blue")

def proses_data_distribusi_platform(file_csv):
    """
    Membaca CSV dan menghitung total REVENUE per PLATFORM.
    """
    try:
        df = pd.read_csv(file_csv)
        
        # --- Menggunakan nama kolom yang BENAR dari file Anda ---
        # Kolom Platform: 'Platform'
        # Kolom Total Penjualan: 'Revenue'
        distribusi = df.groupby('Platform')['Revenue'].sum()
        
        # Urutkan dari nilai terbesar ke terkecil
        distribusi_diurutkan = distribusi.sort_values(ascending=False)
        
        return distribusi_diurutkan

    except FileNotFoundError:
        # Tampilkan pesan error jika file tidak ada
        messagebox.showerror("Error File", 
                             f"File '{file_csv}' tidak ditemukan.\nPastikan file CSV ada di folder yang sama dengan script ini.")
        return None
    except Exception as e:
        # Tangani error lain yang mungkin terjadi
        messagebox.showerror("Error", f"Terjadi error tidak terduga: {e}")
        return None


class AplikasiGrafik(ctk.CTk):
    def __init__(self, data_distribusi):
        super().__init__()

        # --- Konfigurasi Jendela Utama ---
        self.title("Distribusi Total Revenue per Platform E-commerce")
        self.geometry("900x600")

        # --- Buat Frame untuk Grafik ---
        self.frame_grafik = ctk.CTkFrame(self)
        self.frame_grafik.pack(fill="both", expand=True, padx=20, pady=20)

        # --- Buat Grafik menggunakan Matplotlib ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6), facecolor='#f0f0f0')
        
        platforms = data_distribusi.index
        revenues = data_distribusi.values

        # Buat grafik batang
        bars = self.ax.bar(platforms, revenues, color='#3498db')

        # --- Kustomisasi Grafik ---
        self.ax.set_title('Total Revenue per Platform', fontsize=16, fontweight='bold', pad=20)
        self.ax.set_xlabel('Platform E-commerce', fontsize=12)
        self.ax.set_ylabel('Total Revenue', fontsize=12)
        
        # Putar label sumbu X agar tidak tumpang tindih
        plt.xticks(rotation=45, ha="right")
        
        # Tambahkan grid untuk memudahkan pembacaan
        self.ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_axisbelow(True)
        
        # Format label sumbu Y agar lebih mudah dibaca (misalnya dengan pemisah ribuan)
        self.ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Atur layout agar tidak ada elemen yang terpotong
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- Tampilkan Grafik di dalam Jendela CustomTkinter ---
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafik)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


# --- Bagian Utama Program ---
if __name__ == "__main__":
    file_csv = 'Amazon_Sales_in_IDR.csv'
    
    # Proses data terlebih dahulu
    hasil_distribusi = proses_data_distribusi_platform(file_csv)
    
    # Jika data berhasil diproses (tidak None), jalankan aplikasi GUI
    if hasil_distribusi is not None:
        app = AplikasiGrafik(hasil_distribusi)
        app.mainloop()