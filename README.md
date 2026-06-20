# tinyGPT

Implementasi model GPT sederhana (TinyGPT) yang dapat dilatih dari awal.

## Prasyarat

Pastikan Anda memiliki Python yang sudah ter-install di sistem Anda.
Sangat disarankan untuk menggunakan package manager `uv` agar proses instalasi lebih cepat dan mudah dikelola.

```bash
pip install uv
```

## 1. Setup Environment

Proyek ini dapat dijalankan menggunakan CPU maupun GPU (NVIDIA).

### A. Untuk Pengguna CPU
Jalankan perintah ini di terminal untuk membuat dan mengaktifkan environment:
```powershell
uv venv venv
.\venv\Scripts\activate
uv pip install -r requirements.txt
```

### B. Untuk Pengguna GPU (NVIDIA / CUDA)
Jika Anda memiliki GPU NVIDIA dan ingin mempercepat proses training, buat environment khusus menggunakan Python 3.12 (untuk memastikan kompatibilitas PyTorch CUDA terbaru) lalu install versi GPU:
```powershell
uv venv --python 3.12 venv_gpu
.\venv_gpu\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

## 2. Melatih Model (Training)

Sebelum menyuruh model menghasilkan teks, model harus dilatih terlebih dahulu. Pastikan virtual environment pilihan Anda (CPU/GPU) sudah aktif.

```powershell
python train.py
```

Beberapa argumen opsional yang bisa diatur pada `train.py`:
- `--epochs`: Jumlah iterasi training (default: 50). Semakin tinggi, model semakin pintar tapi memakan waktu lebih lama.
- `--batch_size`: Ukuran batch (default: 16).
- `--tokenizer`: Tipe tokenizer (`char`, `bpe`, atau `unigram`. default: `bpe`).

## 3. Menghasilkan Teks (Generation)

Setelah model selesai dilatih, file checkpoint berekstensi `.pt` (misal: `tinygpt_bpe.pt`) akan otomatis dibuat di folder proyek Anda. Kini Anda bisa mulai menghasilkan teks:

```powershell
python generate.py --prompt "strategi tiktok affiliate yang sukses sangat" --max_len 100
```

Beberapa argumen opsional untuk `generate.py`:
- `--prompt`: Teks awalan untuk memicu generate dari model.
- `--max_len`: Jumlah panjang maksimal karakter/kata yang dihasilkan (default: 50).
- `--temperature`: Mengatur tingkat "kreativitas" AI. Nilai yang lebih kecil (< 1.0) membuatnya lebih fokus/tertebak, nilai yang lebih besar (> 1.0) membuatnya lebih variatif (default: 0.8).
- `--repetition_penalty`: Mencegah model mengulang kata yang sama terus-menerus (default: 1.2).
- `--top_k`: Membatasi jumlah probabilitas kata yang bisa diprediksi untuk menghindari kalimat aneh (default: 10).
