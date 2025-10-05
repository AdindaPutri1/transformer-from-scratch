# 🧠 Transformer from Scratch (Decoder-Only, GPT-style)

Implementasi arsitektur **Transformer Decoder-Only (GPT-style)** dari nol menggunakan **NumPy**, tanpa menggunakan library deep learning seperti PyTorch atau TensorFlow.  
Proyek ini bertujuan untuk mendemonstrasikan pemahaman mendalam tentang mekanisme dasar **Self-Attention** dan arsitektur generatif modern.

---

## Fitur Kunci yang Diimplementasikan

Model ini adalah model **Autoregresif** yang terdiri dari beberapa lapisan `DecoderLayer`.  
Komponen-komponen utama yang berhasil diimplementasikan adalah:

- **Token Embedding** — Mengubah ID token menjadi vektor berdimensi *d<sub>model</sub>* dengan scaling × √*d<sub>model</sub>*  
- **Positional Encoding** — Menggunakan pola sinusoidal untuk menyuntikkan informasi posisi  
- **Scaled Dot-Product Attention** — Inti dari mekanisme attention, menghitung skor Query dan Key  
- **Multi-Head Attention (MHA)** — Memungkinkan model menangkap berbagai representasi informasi secara paralel  
- **Feed-Forward Network (FFN)** — Transformasi non-linear per posisi  
- **Residual Connection + Layer Normalization** — Menggunakan arsitektur *Pre-Norm* untuk stabilitas training (forward pass)  
- **Causal Masking** — Wajib untuk decoder-only model; mencegah token saat ini mengakses informasi token masa depan  
- **Output Layer** — Proyeksi akhir ke ukuran vocabulary dan *Softmax* untuk mendapatkan distribusi probabilitas token berikutnya

---

## Hasil Pengujian dan Validasi (Dari `test_transformer.py` dan `demo.ipynb`)

Semua komponen model berhasil melewati unit test fungsional dan diverifikasi melalui visualisasi di notebook.

### Status Fungsional

| Komponen | Status | Hasil Verifikasi Kritis |
|-----------|--------|--------------------------|
| Logika Model | SUCCESS | Semua unit test lulus tanpa error |
| Causal Mask | Passed | Masker terverifikasi sebagai Matriks Segitiga Bawah; bobot attention ke masa depan diatur ke nol |
| Stabilitas Numerik | Stable | Tidak ada nilai NaN atau Inf terdeteksi pada tensor |
| Probabilitas Output | Valid | Sum dari probabilitas Softmax token berikutnya terverifikasi ≈1.0 |

---

## Temuan Utama (Key Insights)

- **Autoregresi Terpenuhi** — Implementasi Causal Mask memastikan bahwa arsitektur bekerja dalam mode autoregresif, ideal untuk tugas pembangkitan teks.  
- **Layer Normalization Esensial** — Verifikasi menunjukkan bahwa Layer Normalization menjaga rata-rata feature ≈ 0 dan standar deviasi ≈ 1 untuk stabilitas jaringan yang dalam.  
- **Validasi Visual** — Pola sinusoidal pada Positional Encoding berhasil divisualisasikan, membuktikan injeksi informasi posisi berjalan benar.

---

## Struktur Folder

```
transformer-from-scratch/
├── README.md              # Dokumentasi (file ini)
├── requirements.txt       # Dependency (NumPy, Matplotlib, Seaborn)
├── .gitignore             # Daftar file yang diabaikan Git (.npy, .ipynb_checkpoints)
├── transformer.py         # Implementasi utama Transformer (kelas-kelas komponen)
├── test_transformer.py    # Unit test komprehensif untuk setiap komponen
├── demo.ipynb             # Demo interaktif, visualisasi, dan hasil eksekusi
└── laporan.pdf            # Laporan 2 halaman (Dokumen Tugas)
```

---

## Cara Menjalankan Proyek

### Prasyarat
- Python **3.8+**
- Git

---

### 1️. Clone Repository dan Setup Lingkungan

```bash
# Clone repository
git clone <URL_REPO_ANDA>
cd transformer-from-scratch

# Buat virtual environment (Disarankan)
python -m venv venv
source venv/bin/activate   # Linux/Mac OS
# venv\Scripts\activate  # Windows
```

---

### 2️. Instal Dependency

Instal semua library yang diperlukan (terutama NumPy, Matplotlib, dan Seaborn untuk visualisasi):

```bash
pip install -r requirements.txt
```

---

### 3️. Jalankan Unit Test

Verifikasi bahwa setiap komponen bekerja dengan benar sebelum menjalankan demo.  
Output harus menunjukkan:

```
🎉 ALL TESTS PASSED! 🎉
```

Jalankan dengan perintah:

```bash
python test_transformer.py
```

---

### 4️. Coba Forward Pass dan Generasi Token Sederhana

Jalankan script utama untuk melihat simulasi forward pass dan demonstrasi prediksi token berurutan:

```bash
python transformer.py
```

---

### 5️. Jalankan Demo Interaktif

Buka file `demo.ipynb` di **Jupyter Notebook** atau **VS Code (Jupyter Extension)** untuk melihat:

- Visualisasi **attention weights**
- Pola **positional encoding**
- Perubahan **shape tensor** di setiap lapisan

---

## Kontributor

Implementasi ini dibuat sebagai bagian dari tugas mata kuliah **Pemrosesan Bahasa Alami (Natural Language Processing)**.

| Role | Nama | NIM | Mata Kuliah |
|------|------|-----|--------------|
| Author | Adinda Putri Romadhon | 22/505508/TK/55321 | Pemrosesan Bahasa Alami |

---

> Notebook ini dibuat untuk tugas implementasi Arsitektur Transformer dari nol.

© 2025 — Transformer NumPy Implementation Project