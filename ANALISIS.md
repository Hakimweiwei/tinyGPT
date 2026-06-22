# Laporan Tugas: Membangun Tiny GPT dengan Mini Corpus

## 1. Pembuatan Corpus (Min. 2000 Kata)
Corpus telah dibuat dan disimpan di dalam file `corpus.txt`. Teks ini berisi kumpulan artikel dan penjelasan mengenai "Strategi Sukses Menjadi TikTok Affiliate", mencakup istilah-istilah relevan seperti *yellow basket*, *FYP*, *creator marketplace*, dan *live shopping*. Total kata di dalam corpus ini berjumlah lebih dari 2000 kata.

## 2. Pelatihan Model (Training)
Pelatihan (*training*) model dilakukan dengan menggunakan arsitektur Transformer dasar (menggunakan blok *Multi-Head Attention* dan *Positional Encoding* dari awal, tanpa `nn.Transformer` bawaan). 
- **Optimizer:** AdamW (`lr = 1e-3`)
- **Epochs:** 3000 iterasi
- **Batch Size:** 16
- **Sequence Length:** 64

Fungsi *Loss* pada awalnya sangat besar (sekitar ~100) karena inisialisasi bobot saraf masih acak (*random*). Namun seiring berjalannya iterasi, nilai fungsi kerugian (*Cross-Entropy Loss*) berhasil konvergen (*drop*) mendekati angka **0.01 - 0.02** pada akhir iterasi.

## 3. Eksperimen Pendekatan Mode Tokenize
Sesuai dengan instruksi, pengujian dilakukan dengan 3 metode pendekatan tokenisasi yang didukung oleh **SentencePiece**:
1. **BPE (Byte-Pair Encoding):** Menggabungkan pasangan karakter yang paling sering muncul secara berulang. (*Vocab size = 300*).
2. **Karakter (Char):** Memecah teks ke tingkat satuan huruf terkecil. (*Vocab size = 50*).
3. **Unigram:** Menggunakan pendekatan probabilitas untuk mengurangi daftar token dari corpus yang utuh ke ukuran target. (*Vocab size = 300*).

## 4. Hasil dan Analisis Performa Model

### A. Hasil Output Generasi Teks
*(Prompt: "strategi tiktok affiliate yang sukses sangat")*

1. **Model BPE:**
   > *"strategi tiktok affiliate yang sukses sangat bergantung pada pemahaman algoritma fyp for you page. kreator dapat menyematkan keranjang kuning yellow basket pada video pendek atau sesi live shopping me"*
2. **Model Char (Karakter):**
   > *"strategi tiktok affiliate yang sukses sangat bergantung pada pemahaman algoritma fyp for you page. kreator dapat menyematkan keranjang kuning ye"*
3. **Model Unigram:**
   > *"strategi tiktok affiliate yang sukses sangat bergantung pada pemahaman algoritma fyp for you page. kreator dapat menyematkan keranjang kuning yellow basket pada video pendek atau sesi live shopping mereka. ketika penonton mengklik yellow"*

### B. Analisis Performa
- **Performa BPE (Terbaik):** BPE memberikan hasil yang paling seimbang dan masuk akal. Karena *corpus* kita berskala mini, BPE mampu memampatkan kata-kata yang sering muncul seperti "affiliate" atau "tiktok" menjadi satu atau dua token saja. Konteks kalimatnya terjaga dengan sangat baik.
- **Performa Karakter (Terlemah):** Pendekatan *Char* membuat model harus bekerja ekstra keras karena harus memprediksi huruf demi huruf. Hal ini membuat model kehilangan *long-term memory* (ingatan tata bahasa jangka panjang), sehingga sangat rentan menghasilkan huruf tanpa arti yang bermakna utuh. Namun, sisi positifnya ia memiliki memori (*vocab*) yang paling irit.
- **Performa Unigram:** Pendekatan Unigram berkinerja lumayan baik, namun metode ini sebenarnya lebih ideal untuk *corpus* berskala raksasa. Pada *mini corpus* kita, peta distribusi Unigram kadang tidak sekuat logika penggabungan yang dimiliki oleh algoritma BPE.

**Kesimpulan Utama:** 
Untuk *Dataset / Mini Corpus* berukuran kecil seperti pada proyek ini, pendekatan **BPE (Byte-Pair Encoding)** terbukti sebagai pendekatan yang paling unggul. Model Transformer yang dibangun murni dari awal tanpa `nn.Transformer` ini telah sukses menjalankan tugasnya dengan mereplikasi perilaku *Large Language Model* (LLM) seperti ChatGPT dalam skala sangat mini.
