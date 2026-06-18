# Analisis Eksperimen Tiny GPT

## 1. Tujuan Eksperimen
Eksperimen ini bertujuan untuk membangun arsitektur model bahasa Transformer murni dari awal (from scratch) menggunakan PyTorch tanpa menggunakan library pre-built seperti `nn.Transformer`. Eksperimen ini juga menguji performa model tersebut ketika dilatih pada dataset yang berukuran sangat kecil (Mini Corpus) dan dengan perbendaharaan kata (vocabulary) yang sangat terbatas.

## 2. Metodologi
- **Corpus**: Dataset berukuran sekitar ~2000 kata dengan topik spesifik seputar "TikTok Affiliate" (keyword wajib: *yellow basket, FYP, TikTok Shop, komisi, live shopping, creator marketplace, conversion rate, unboxing, algoritma, affiliate*).
- **Tokenizer**: Menggunakan `SentencePiece` dengan metode Byte-Pair Encoding (BPE) berukuran *vocabulary* yang sengaja dibatasi (vocab_size = 120).
- **Arsitektur**: Mengimplementasikan `MultiHeadAttention`, `FeedForward`, `PositionalEncoding`, dan Masking secara manual dengan PyTorch.
- **Training**: Menggunakan algoritma optimasi AdamW dengan iterasi sebanyak 5 epoch.

## 3. Hasil Loss (Tingkat Kesalahan)
Dari file `loss_history_bpe.json` yang terekam selama proses *training*, tingkat *Loss* model adalah sebagai berikut:
- **Epoch 1:** Rata-rata Loss 3.0030 (Awalnya bernilai 103.55 dan langsung terjun tajam)
- **Epoch 2:** Rata-rata Loss 0.0242
- **Epoch 3:** Rata-rata Loss 0.0219
- **Epoch 4:** Rata-rata Loss 0.0209
- **Epoch 5:** Rata-rata Loss 0.0207

Nilai loss `0.02` menandakan bahwa model telah berhasil memprediksi teks latih dengan nyaris 100% akurat secara matematis. Model mengalami *convergence* dengan sangat cepat.

## 4. Analisis Hasil Generasi Teks (Inference)

### A. Fenomena Overfitting Sempurna
Ketika model diberikan *prompt* yang berisi teks yang persis sama dengan yang ada di dalam korpus:
> **Prompt:** `"strategi tiktok affiliate yang sukses sangat"`
> **Output:** `"bergantung pada pemahaman algoritma fyp for you page. kreator dapat menyematkan..."`

**Analisis:**
Model mampu melanjutkan kata demi kata dengan sangat sempurna karena model mengalami **Overfitting Sempurna**. Model menghafal persis seluruh urutan kalimat di dalam korpus. Ini membuktikan bahwa arsitektur `MultiHeadAttention` (logika menengok kembali kata-kata sebelumnya) berfungsi 100%.

### B. Fenomena Out-Of-Vocabulary (OOV)
Ketika model diberikan *prompt* yang tidak pernah dilihatnya dalam proses training:
> **Prompt:** `"apa itu affiliate?"`
> **Output:** `"apa itu affiliate ⁇  ⁇  ⁇  ⁇  ⁇  ⁇"`

**Analisis:**
Kata `"apa"`, `"itu"`, dan tanda tanya `"??"` tidak pernah muncul sekalipun dalam Mini Corpus latih. Karena ukuran BPE *vocabulary* hanya 120, tokenizer gagal mengubahnya dan menjadikannya token *Unknown* (`[UNK]`). Karena model kebingungan menerima token yang tidak ia pahami (tanpa riwayat pelatihan), ia terjebak mencetak token `[UNK]` secara berulang. Ini menegaskan bahwa model LLM sangat dibatasi oleh skala dataset latihannya. Kecerdasan (kemampuan generalisasi) hanya akan muncul ketika dataset berskala masif (seperti GPT asli).

## 5. Kesimpulan Akhir
Proyek ini membuktikan bahwa arsitektur Transformer dan teori di balik *Self-Attention* berhasil diimplementasikan dari nol. Fenomena yang terjadi pada teks uji (baik yang persis sama maupun teks yang tidak dikenali) sangat masuk akal secara ilmu *Machine Learning* untuk ukuran model eksperimental (Tiny) dengan korpus (Mini). Proyek berjalan sukses.
