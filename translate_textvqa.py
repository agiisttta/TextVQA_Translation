"""
Terjemahan dataset TextVQA (EN -> ID) menggunakan Google Translate,
dengan fallback MyMemory, cache, dan resume antar-batch.

Versi lokal (bukan Colab) dari notebook:
https://colab.research.google.com/drive/1lM4xnrCWkWRK4ljKVHw3ApPKAcVOVwpQ

Perbedaan dari Colab:
- Dataset diambil dari Hugging Face "lmms-lab-encoder/textvqa" (split "validation",
  5000 baris, setara TextVQA_0.5.1_val.json) lewat REST API datasets-server,
  bukan file JSON hasil mount Google Drive.
- Path hasil ditulis ke folder lokal (BASE_DIR), bukan /content/drive/...
- TRANSLATE_DELAY diturunkan ke 0.5 detik karena jalan dari IP lokal
  (bukan IP Colab yang sudah banyak dipakai bareng, jadi lebih cepat kena 429).
- Endpoint Google Translate memakai clients5.google.com (client=dict-chrome-ex)
  yang menerima banyak teks sekaligus, karena endpoint scraping milik
  deep-translator (translate.google.com/m) sekarang diblokir CAPTCHA.
"""

import json
import os
import random
import time

import requests
from tqdm.auto import tqdm
from deep_translator import MyMemoryTranslator

# =============================================================================
# 1. KONFIGURASI
# =============================================================================

HF_DATASET = "lmms-lab-encoder/textvqa"
HF_CONFIG = "default"
HF_SPLIT = "validation"  # 5000 baris, setara TextVQA_0.5.1_val.json
HF_ROWS_PER_REQUEST = 100  # batas maksimum API datasets-server

# Folder untuk menyimpan hasil (relatif ke lokasi script ini)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

DATASET_CACHE_PATH = os.path.join(BASE_DIR, "textvqa_validation_raw.json")
BATCH_DIR = os.path.join(BASE_DIR, "translation_batches")
CACHE_PATH = os.path.join(BASE_DIR, "translation_cache.json")
FINAL_CSV_PATH = os.path.join(BASE_DIR, "textvqa_5000_translated.csv")

os.makedirs(BATCH_DIR, exist_ok=True)

BATCH_SIZE = 1250
# Delay lebih kecil dari versi Colab (1.2 -> 2.0) karena jalan dari IP lokal
# yang belum "kotor" seperti IP Colab yang dipakai bareng-bareng.
# Kalau mulai muncul banyak TooManyRequests, naikkan nilai ini.
TRANSLATE_DELAY = 0.5

MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_WAIT = 30  # detik, bertambah tiap percobaan

# Endpoint Google Translate yang masih jalan dan bisa menerjemahkan
# banyak teks dalam satu request (dipakai juga oleh googletrans).
GOOGLE_BATCH_URL = "https://clients5.google.com/translate_a/t"
SOURCE_LANG = "en"
TARGET_LANG = "id"

# Batas per request batch: jumlah teks dan total karakter.
MAX_TEXTS_PER_REQUEST = 50
MAX_CHARS_PER_REQUEST = 1200

print("Folder hasil:", BASE_DIR)
print("Folder batch:", BATCH_DIR)


# =============================================================================
# 2. AMBIL DATASET DARI HUGGING FACE (ganti pengganti mount Drive + JSON lokal)
# =============================================================================

def normalize_answers(answers):
    result = []

    for answer in answers:
        if isinstance(answer, dict):
            result.append(str(answer.get("answer", "")))
        else:
            result.append(str(answer))

    return result


def fetch_dataset_from_hf():
    """Ambil seluruh baris split 'validation' lewat REST API datasets-server.

    Hanya mengambil kolom teks (question_id, image_id, question, answers),
    TIDAK mengunduh gambar (field 'image' diabaikan) supaya ringan dan cepat.
    """
    if os.path.isfile(DATASET_CACHE_PATH):
        with open(DATASET_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print("Dataset dimuat dari cache lokal:", DATASET_CACHE_PATH)
        print("Jumlah data:", len(cached))
        return cached

    print(f"Mengambil dataset '{HF_DATASET}' (split={HF_SPLIT}) dari Hugging Face...")

    dataset = []
    offset = 0

    # Ambil total baris dulu dari response pertama
    first = _fetch_rows_page(offset=0, length=1)
    total_rows = first["num_rows_total"]
    print("Total baris di split ini:", total_rows)

    pbar = tqdm(total=total_rows, desc="Mengambil dataset dari HF")

    while offset < total_rows:
        length = min(HF_ROWS_PER_REQUEST, total_rows - offset)
        page = _fetch_rows_page(offset=offset, length=length)

        for item in page["rows"]:
            row = item["row"]
            dataset.append({
                "question_id": str(row["question_id"]),
                "image_id": str(row["image_id"]),
                "question": str(row["question"]),
                "answers": normalize_answers(row.get("answers", [])),
            })

        pbar.update(length)
        offset += length

    pbar.close()

    with open(DATASET_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    print("Dataset disimpan ke cache lokal:", DATASET_CACHE_PATH)
    print("Jumlah data:", len(dataset))

    return dataset


def _fetch_rows_page(offset, length, max_retries=5):
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": HF_SPLIT,
        "offset": offset,
        "length": length,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) * 3
            print(f"Gagal ambil halaman offset={offset}: {e}. "
                  f"Coba lagi setelah {wait_time}s.")
            time.sleep(wait_time)


# =============================================================================
# 3. CACHE TERJEMAHAN
# =============================================================================

def load_cache(path):
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Peringatan: cache tidak dapat dibaca.")
        print("File cache lama tidak dihapus.")
        raise


def save_cache(cache, path):
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    os.replace(temp_path, path)


# =============================================================================
# 4. TRANSLATOR: Google Translate (batch) + retry/backoff + fallback MyMemory
# =============================================================================

# MyMemoryTranslator butuh kode bahasa berformat region (en-GB/id-ID).
fallback_translator = MyMemoryTranslator(source="en-GB", target="id-ID")


class RateLimitedError(RuntimeError):
    """Google membalas 429 atau halaman blokir (CAPTCHA)."""


def _request_google_batch(texts):
    """Terjemahkan beberapa teks dalam satu request. Hasil urutannya sama."""
    params = [
        ("client", "dict-chrome-ex"),
        ("sl", SOURCE_LANG),
        ("tl", TARGET_LANG),
    ]
    params.extend(("q", text) for text in texts)

    response = requests.get(
        GOOGLE_BATCH_URL,
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )

    if response.status_code == 429:
        raise RateLimitedError("Google membatasi permintaan (HTTP 429).")

    response.raise_for_status()

    try:
        data = response.json()
    except ValueError:
        # Saat IP diblokir sementara, Google membalas halaman HTML "Sorry...".
        raise RateLimitedError(
            "Google mengembalikan halaman blokir, bukan JSON."
        )

    # Kadang hasil dibungkus satu level lagi ([[...]]) -> diratakan.
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]

    if not isinstance(data, list):
        raise RuntimeError(
            f"Format respons Google tidak dikenal: {str(data)[:80]!r}"
        )

    if len(data) != len(texts):
        raise RuntimeError(
            f"Jumlah hasil ({len(data)}) tidak sesuai "
            f"jumlah teks ({len(texts)})."
        )

    return [str(item) for item in data]


def _translate_with_fallback(text):
    """Penerjemah cadangan untuk satu teks (MyMemory)."""
    print(
        f"\nGoogle Translate gagal terus. "
        f"Beralih ke MyMemory Translator untuk: {text[:50]!r}"
    )

    try:
        result = fallback_translator.translate(text)
    except Exception as e:
        raise RuntimeError(
            f"Penerjemahan gagal total untuk teks: {text[:50]!r}"
        ) from e

    if not result:
        raise RuntimeError(
            f"Penerjemahan gagal total untuk teks: {text[:50]!r}"
        )

    return result


def _translate_chunk(texts):
    """Terjemahkan satu chunk teks lewat Google, dengan retry + split."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return _request_google_batch(texts)

        except RateLimitedError:
            wait_time = (
                RATE_LIMIT_BASE_WAIT * (attempt + 1)
                + random.uniform(0, 5)
            )
            print(
                f"\nGoogle Translate membatasi permintaan "
                f"(percobaan {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}). "
                f"Menunggu {wait_time:.0f} detik sebelum mencoba lagi."
            )
            time.sleep(wait_time)

        except Exception:
            if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                break

            wait_time = (2 ** attempt) * 5 + random.uniform(0, 2)
            print(
                f"\nPercobaan {attempt + 1} gagal. "
                f"Mencoba kembali setelah {wait_time:.1f} detik."
            )
            time.sleep(wait_time)

    # Google tetap gagal. Kalau chunk berisi banyak teks, pecah jadi dua
    # supaya teks yang bermasalah tidak menjatuhkan seluruh chunk.
    if len(texts) > 1:
        middle = len(texts) // 2
        print(
            f"\nChunk {len(texts)} teks gagal. "
            f"Membelah menjadi {middle} + {len(texts) - middle} teks."
        )
        return (
            _translate_chunk(texts[:middle])
            + _translate_chunk(texts[middle:])
        )

    return [_translate_with_fallback(texts[0])]


def translate_many(texts, translation_cache):
    """Terjemahkan list teks sekaligus; hasil urutannya sama dengan input."""
    texts = [str(text).strip() for text in texts]

    results = {}
    pending = []

    for text in texts:
        if not text:
            results[text] = ""
        elif text in translation_cache:
            results[text] = translation_cache[text]
        elif text not in pending:
            pending.append(text)

    chunk = []
    chunk_chars = 0

    def flush_chunk():
        nonlocal chunk, chunk_chars

        if not chunk:
            return

        translated = _translate_chunk(chunk)

        for source_text, translated_text in zip(chunk, translated):
            translation_cache[source_text] = translated_text
            results[source_text] = translated_text

        save_cache(translation_cache, CACHE_PATH)

        chunk = []
        chunk_chars = 0

        time.sleep(TRANSLATE_DELAY + random.uniform(0, 0.8))

    for text in pending:
        too_many = len(chunk) >= MAX_TEXTS_PER_REQUEST
        too_long = chunk_chars + len(text) > MAX_CHARS_PER_REQUEST

        if chunk and (too_many or too_long):
            flush_chunk()

        chunk.append(text)
        chunk_chars += len(text)

    flush_chunk()

    return [results[text] for text in texts]


def translate_text_cached(text, translation_cache):
    return translate_many([text], translation_cache)[0]



# =============================================================================
# 5. BATCH + RESUME (identik dengan logika di Colab)
# =============================================================================

def get_batch_path(batch_id):
    return os.path.join(BATCH_DIR, f"textvqa_{batch_id}.jsonl")


def load_batch(path):
    records = {}

    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records[str(record["question_id"])] = record

    return records


def append_batch_record(record, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_batches(dataset, translation_cache):
    total_data = len(dataset)
    total_batches = (total_data + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_index in range(total_batches):
        start = batch_index * BATCH_SIZE
        end = min(start + BATCH_SIZE, total_data)

        batch_id = f"B{batch_index + 1:02d}"
        batch_path = get_batch_path(batch_id)
        completed = load_batch(batch_path)

        print(f"\n{'=' * 50}")
        print(f"{batch_id}: data {start + 1}-{end}")
        print(f"Sudah tersimpan: {len(completed)}/{end - start}")
        print(f"{'=' * 50}")

        try:
            for row in tqdm(
                dataset[start:end],
                desc=f"Menerjemahkan {batch_id}"
            ):
                qid = str(row["question_id"])

                if qid in completed:
                    continue

                question_en = row["question"]
                answers_en = row["answers"]

                translated = translate_many(
                    [question_en] + answers_en,
                    translation_cache,
                )
                question_idn = translated[0]
                answers_idn = translated[1:]

                record = {
                    "batch_id": batch_id,
                    "question_id": qid,
                    "image_id": row["image_id"],
                    "question": question_en,
                    "answers": answers_en,
                    "question_idn": question_idn,
                    "answers_idn": answers_idn,
                }

                append_batch_record(record, batch_path)
                completed[qid] = record

            print(
                f"{batch_id} selesai: "
                f"{len(completed)}/{end - start} data."
            )

        except Exception as e:
            print("\nPenerjemahan terhenti.")
            print("Error:", type(e).__name__, repr(e))
            print("Batch:", batch_id)
            print("File progres:", batch_path)
            print(
                "Hasil yang sudah tersimpan tetap dipertahankan. "
                "Jalankan kembali setelah masalah teratasi."
            )
            break


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    dataset = fetch_dataset_from_hf()

    translation_cache = load_cache(CACHE_PATH)
    print("Jumlah teks dalam cache:", len(translation_cache))

    # Tes 1 kalimat dulu sebelum lanjut ke batch (sama seperti cell tes di Colab)
    try:
        test_result = translate_text_cached(
            "What is written on the sign?", translation_cache
        )
        print("Tes berhasil:", test_result)
    except Exception as e:
        print("Tes gagal:", type(e).__name__, repr(e))
        print(
            "Jangan lanjutkan ke batch sebelum masalah akses "
            "Google Translate teratasi."
        )
        return

    run_batches(dataset, translation_cache)


if __name__ == "__main__":
    main()
