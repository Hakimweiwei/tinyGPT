# TinyGPT - Mini Corpus Project

Proyek ini adalah tugas mata kuliah Pemrosesan Data Multimedia (PDM) yang bertujuan untuk membangun model bahasa (Language Model) berarsitektur dasar Transformer (TinyGPT) dari nol tanpa menggunakan fungsi `nn.Transformer` bawaan.

## Deskripsi Tugas
1. **Corpus:** Proyek ini menggunakan sebuah *mini corpus* berukuran >2000 kata mengenai "Strategi Tiktok Affiliate" (`corpus.txt`).
2. **Tokenisasi:** Mendukung 3 pendekatan tokenisasi berbeda menggunakan algoritma `sentencepiece`:
   - BPE (Byte-Pair Encoding)
   - Karakter (Char)
   - Unigram
3. **Training:** Melatih arsitektur Transformer murni secara lokal untuk menirukan pola teks pada corpus.
4. **Analisis:** Membandingkan hasil generasi teks dari ketiga model tokenisasi tersebut.

## Instalasi dan Setup Environment
Sangat disarankan menjalankan proyek ini di dalam virtual environment. Jika perangkat Anda mendukung GPU NVIDIA, ikuti langkah berikut:

```powershell
uv venv --python 3.12 venv_gpu
.\venv_gpu\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

## Cara Menjalankan

### 1. Melatih Model (Training)
Latih model dengan salah satu tipe *tokenizer*. Model akan otomatis dilatih selama 3000 epoch.
```powershell
python train.py --tokenizer bpe
python train.py --tokenizer char
python train.py --tokenizer unigram
```

### 2. Generasi Teks (Generation)
Setelah model selesai dilatih, file `.pt` akan otomatis tersimpan. Kini Anda bisa menguji kemampuannya dalam melanjutkan pola kalimat:
```powershell
python generate.py --tokenizer bpe --prompt "strategi tiktok affiliate yang sukses sangat" --max_len 100
```
*Catatan: Ubah argumen `--tokenizer` ke `char` atau `unigram` untuk memanggil model yang berbeda dan melihat perbedaan kualitas hasil akhirnya.*
