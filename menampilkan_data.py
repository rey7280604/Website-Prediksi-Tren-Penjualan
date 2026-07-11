import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def tampilkan_data_dalam_popup(file_csv):
    """
    Membaca file CSV dan menampilkannya dalam jendela popup yang bisa di-scroll.
    """
    try:
        # Baca file CSV menggunakan pandas
        df = pd.read_csv(file_csv)

        # --- Buat Jendela Utama (Root Window) ---
        root = tk.Tk()
        root.title(f"Data dari {file_csv}")
        root.geometry("1200x700")  # Atur ukuran awal jendela

        # --- Buat Frame untuk Tabel dan Scrollbar ---
        # Frame ini membantu mengatur posisi tabel dan scrollbar
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Buat Widget Tabel (Treeview) ---
        # Treeview adalah widget yang ideal untuk menampilkan data dalam bentuk tabel
        tree = ttk.Treeview(main_frame)

        # --- Konfigurasi Kolom Tabel ---
        # Ambil nama kolom dari DataFrame pandas
        kolom = list(df.columns)
        tree["columns"] = kolom

        # Sembunyikan kolom kosong pertama (#0) yang biasanya ada di Treeview
        tree.column("#0", width=0, stretch=tk.NO)

        # Atur setiap kolom berdasarkan nama dari CSV
        for nama_kolom in kolom:
            tree.column(nama_kolom, anchor="w", width=120) # 'w' = rata kiri
            tree.heading(nama_kolom, text=nama_kolom, anchor="w")

        # --- Masukkan Data Baris ke Tabel ---
        # Konversi DataFrame ke list dan masukkan baris per baris
        df_rows = df.to_numpy().tolist() # Lebih cepat dari iterasi biasa
        for row in df_rows:
            tree.insert("", tk.END, values=row)

        # --- Tambahkan Scrollbar ---
        # Scrollbar Vertikal
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=tree.yview)
        # Scrollbar Horizontal
        hsb = ttk.Scrollbar(main_frame, orient="horizontal", command=tree.xview)
        
        # Hubungkan scrollbar dengan tabel
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # --- Atur Posisi Widget di dalam Frame ---
        # Menggunakan grid untuk tata letak yang lebih rapi
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        # Agar frame bisa menyesuaikan ukuran saat jendela di-resize
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Jalankan jendela popup
        root.mainloop()

    except FileNotFoundError:
        # Tampilkan pesan error jika file tidak ditemukan
        messagebox.showerror("Error File", 
                             f"File '{file_csv}' tidak ditemukan.\nPastikan file CSV ada di folder yang sama dengan script ini.")
    except Exception as e:
        # Tangani error lain yang mungkin terjadi
        messagebox.showerror("Error", f"Terjadi error tidak terduga: {e}")

# --- Bagian Utama yang Dijalankan ---
if __name__ == "__main__":
    nama_file = 'Amazon_Sales_in_IDR.csv'
    tampilkan_data_dalam_popup(nama_file)