from flask import Flask, request, jsonify
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pymysql
from pymysql.cursors import DictCursor
from flask_cors import CORS
import os
import tempfile
import re
import traceback
import xgboost as xgb

app = Flask(__name__)
CORS(app)  # Enable CORS if needed
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150MB


# Database configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "prediksi_svr",
}


# Kolom yang diharapkan untuk upload file
# Kolom yang diharapkan untuk upload file
EXPECTED_COLUMNS = [
    "Nosi",
    "Posisi Saat Ini",
    "Status Kiriman",
    "Produk",
    "SLA",
    "Kantor Kirim",
    "Tgl Kirim",
    "Tgl Antaran Pertama",
    "Tgl Update",
    "Petugas",
    "Nama Penerima",
    "Alamat ",
    "Kota ",
    "Alasan Gagal",
    "Alasan Irregulitas",
    "Status SWP",
    "Berat",
    "Cek",
]
# Kolom tanggal yang perlu diformat
DATE_COLUMNS = ["tgl_kirim", "tgl_antaran_pertama", "tgl_update"]


def get_db_connection():
    return pymysql.connect(**db_config)


def get_db_connection_with_cursor():
    """Membuat dan mengembalikan koneksi database dengan DictCursor"""
    db_params = db_config.copy()
    db_params["cursorclass"] = DictCursor
    return pymysql.connect(**db_params)


def parse_yyyymm(date_str):
    """Parse YYYY-MM format into year and month integers"""
    try:
        year, month = map(int, date_str.split("-"))
        return year, month
    except:
        return None, None


def clean_column_name(column):
    """Membersihkan nama kolom dengan penanganan khusus untuk 'kota' dan 'alamat'"""
    column = column.strip().lower()

    if column.startswith("kota"):
        return column.replace(" ", "")
    elif column.startswith("alamat"):
        return column.replace(" ", "")
    elif any(date_col in column for date_col in ["tgl", "tanggal"]):
        return re.sub(r"\s+", "_", column)
    else:
        return column.replace(" ", "_")


def format_date(date_val):
    """Mencoba mengkonversi berbagai format tanggal ke format yyyy-mm-dd"""
    if pd.isna(date_val) or date_val == "":
        return None

    try:
        # Tambahkan format eksplisit untuk membaca format tanggal yang benar
        parsed_date = pd.to_datetime(date_val, format="%d/%m/%Y %H:%M", errors="coerce")
        if pd.isna(parsed_date):
            return None
        return parsed_date.strftime("%Y-%m-%d")
    except:
        return None


def validate_dataframe(df):
    """Validasi dataframe memiliki kolom yang diharapkan"""
    cleaned_cols_from_file = set(clean_column_name(col) for col in df.columns)
    expected_cleaned_cols = set()
    for expected_col in EXPECTED_COLUMNS:
        cleaned_expected_col = expected_col.strip().lower()
        if cleaned_expected_col.startswith("kota"):
            cleaned_expected_col = cleaned_expected_col.replace(" ", "")
        elif cleaned_expected_col.startswith("alamat"):
            cleaned_expected_col = cleaned_expected_col.replace(" ", "")
        elif any(date_col in cleaned_expected_col for date_col in ["tgl", "tanggal"]):
            cleaned_expected_col = cleaned_expected_col.replace(" ", "_")
        else:
            cleaned_expected_col = cleaned_expected_col.replace(" ", "_")
        expected_cleaned_cols.add(cleaned_expected_col)

    missing_columns = []
    for cleaned_expected in expected_cleaned_cols:
        if cleaned_expected not in cleaned_cols_from_file:
            original_missing_col = next(
                (
                    col
                    for col in EXPECTED_COLUMNS
                    if clean_column_name(col) == cleaned_expected
                ),
                cleaned_expected,
            )
            missing_columns.append(original_missing_col)

    if missing_columns:
        return False, f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}"

    return True, "Validasi berhasil"


def clean_dataframe(df):
    """Membersihkan dataframe dan mengganti nama kolom sesuai target"""
    # Membersihkan nama kolom awal
    df.columns = [clean_column_name(col) for col in df.columns]
    # Format kolom tanggal yang diperlukan
    for date_column in DATE_COLUMNS:
        if date_column in df.columns:
            df[date_column] = df[date_column].apply(format_date)
    return df


def save_df_to_mysql(df, chunk_size=5000):
    """Menyimpan dataframe ke MySQL dengan chunking menggunakan nama kolom dari dataframe"""
    connection = get_db_connection_with_cursor()
    try:
        with connection.cursor() as cursor:
            total_inserted = 0
            cleaned_columns = (
                df.columns.tolist()
            )  # Gunakan nama kolom dari dataframe yang sudah dibersihkan

            # Simpan data dalam chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i : i + chunk_size]
                rows = chunk.replace({np.nan: None}).to_dict("records")
                if not rows:
                    continue

                # Buat query INSERT dengan nama kolom dari dataframe
                quoted_columns = ", ".join([f"`{col}`" for col in cleaned_columns])
                placeholders = ", ".join(["%s"] * len(cleaned_columns))
                query = (
                    f"INSERT INTO data_awal ({quoted_columns}) VALUES ({placeholders})"
                )

                # Siapkan data untuk dimasukkan berdasarkan urutan kolom di dataframe
                values = []
                for row in rows:
                    row_values = [row.get(col) for col in cleaned_columns]
                    values.append(row_values)

                # Execute query dan commit
                cursor.executemany(query, values)
                connection.commit()
                total_inserted += len(rows)

        return {"status": "success", "inserted_rows": total_inserted}
    except Exception as e:
        connection.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        connection.close()


@app.route("/upload-data-awal", methods=["POST"])
def upload_data():
    # print("Menerima permintaan ke /upload-data-awal")
    # print(f"Isi request.files: {request.files}")
    try:
        # Periksa apakah ada file yang dikirim
        if "file" not in request.files:
            return (
                jsonify({"status": "error", "message": "Tidak ada file yang dikirim"}),
                400,
            )

        file = request.files["file"]
        if file.filename == "":
            return (
                jsonify({"status": "error", "message": "Tidak ada file yang dipilih"}),
                400,
            )

        # Periksa ekstensi file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".csv", ".xlsx", ".xls"]:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Format file harus .csv, .xlsx, atau .xls",
                    }
                ),
                400,
            )

        # Simpan file sementara
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        temp_file.close()

        try:
            # Baca file berdasarkan ekstensi
            if file_ext == ".csv":
                # Coba deteksi delimiter, utamakan koma, lalu titik koma
                try:
                    df = pd.read_csv(
                        temp_file.name,
                        low_memory=False,
                        sep=";",  # Gunakan koma sebagai delimiter pertama
                        header=0,
                        quotechar='"',
                    )
                    # Cek jika parsing dengan koma menghasilkan hanya satu kolom (kemungkinan delimiter adalah titik koma)
                    if len(df.columns) == 1:
                        df = pd.read_csv(
                            temp_file.name,
                            low_memory=False,
                            sep=",",
                            header=0,
                            quotechar='"',
                        )
                    if len(df.columns) == 1:
                        df = pd.read_csv(
                            temp_file.name,
                            low_memory=False,
                            sep="\t",  # Coba dengan tab sebagai delimiter
                            header=0,
                            quotechar='"',
                        )
                    if len(df.columns) == 1:
                        df = pd.read_csv(
                            temp_file.name,
                            low_memory=False,
                            sep=" ",  # Coba dengan spasi sebagai delimiter
                            header=0,
                            quotechar='"',
                        )
                except Exception as e:
                    os.unlink(temp_file.name)
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"Gagal memproses file CSV: {str(e)}",
                            }
                        ),
                        500,
                    )
            else:
                df = pd.read_excel(temp_file.name)

            # Validasi dataframe
            is_valid, message = validate_dataframe(df)
            if not is_valid:
                os.unlink(temp_file.name)
                return jsonify({"status": "error", "message": message}), 400

            # Bersihkan dataframe
            cleaned_df = clean_dataframe(df)

            # Simpan ke database
            result = save_df_to_mysql(cleaned_df)

            # Hapus file sementara
            os.unlink(temp_file.name)

            if result["status"] == "success":
                return jsonify(result), 200
            else:
                return jsonify(result), 500

        except Exception as e:
            os.unlink(temp_file.name)
            return (
                jsonify(
                    {"status": "error", "message": f"Gagal memproses file: {str(e)}"}
                ),
                500,
            )

    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Terjadi kesalahan: {str(e)}"}),
            500,
        )


@app.route("/dataset-TES", methods=["POST"])
def dataset_tes():
    # Fungsi safe_value tidak diperlukan lagi karena kita akan memastikan tidak ada NaN sebelum menyimpan
    # def safe_value(val):
    #     """Ubah NaN menjadi None untuk MySQL"""
    #     if val is None or (isinstance(val, float) and math.isnan(val)):
    #         return None
    #     return val

    try:
        # Ambil data dari request
        data = request.json
        nama_dataset = data.get("nama_dataset")
        tahun_list = data.get("tahun", [])
        tahun_list = [int(t) for t in tahun_list]

        if not nama_dataset or not isinstance(tahun_list, list):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "nama_dataset dan tahun (list) harus diberikan",
                    }
                ),
                400,
            )

        # Koneksi database
        conn = pymysql.connect(**db_config, cursorclass=pymysql.cursors.DictCursor)
        cursor = conn.cursor()

        # Ambil data dari database data_awal
        cursor.execute("SELECT kota, tgl_antaran_pertama, sla FROM data_awal")
        rows = cursor.fetchall()

        # Konversi ke DataFrame
        df = pd.DataFrame(rows)
        df.rename(
            columns={
                "kota": "Kota",
                "tgl_antaran_pertama": "Tgl_Antaran_Pertama",
                "sla": "SLA",
            },
            inplace=True,
        )

        # Konversi tanggal ke datetime
        df["Tgl_Antaran_Pertama"] = pd.to_datetime(
            df["Tgl_Antaran_Pertama"], errors="coerce"
        )
        # Hapus baris dengan tanggal null
        df.dropna(subset=["Tgl_Antaran_Pertama"], inplace=True)

        # Filter tahun
        df = df[df["Tgl_Antaran_Pertama"].dt.year.isin(tahun_list)]

        # Ekstrak kecamatan dari kota
        df["Kecamatan"] = df["Kota"].apply(
            lambda x: x.split(", ")[1] if isinstance(x, str) and ", " in x else None
        )
        df["Kota"] = df["Kota"].apply(
            lambda x: x.split(", ")[0] if isinstance(x, str) and ", " in x else x
        )

        # Set bulan (untuk sin & cos)
        df["Bulan"] = df["Tgl_Antaran_Pertama"].dt.month

        # Mengonversi tanggal ke format 'YYYY-MM' untuk grouping
        df["Tgl_Antaran_Pertama"] = (
            df["Tgl_Antaran_Pertama"].dt.to_period("M").astype(str)
        )

        # Grouping
        grouped = (
            df.groupby(["Kecamatan", "Tgl_Antaran_Pertama", "SLA"])
            .size()
            .reset_index(name="Jumlah_Paket")
        )

        # Konversi kembali ke datetime untuk fitur lag/rolling
        # Tanggal yang tidak valid pada langkah sebelumnya akan menjadi NaT di sini
        grouped["Tgl_Antaran_Pertama"] = pd.to_datetime(
            grouped["Tgl_Antaran_Pertama"], format="%Y-%m", errors="coerce"
        )

        # --- PERBAIKAN UTAMA: Hapus baris dengan tanggal yang tidak valid setelah grouping ---
        grouped.dropna(subset=["Tgl_Antaran_Pertama"], inplace=True)

        grouped = grouped.sort_values(by=["Kecamatan", "SLA", "Tgl_Antaran_Pertama"])

        # Tambah fitur lag
        for lag in range(1, 7):
            grouped[f"Lag_{lag}"] = grouped.groupby(["Kecamatan", "SLA"])[
                "Jumlah_Paket"
            ].shift(lag)

        # Tambah rolling mean & std
        grouped["Rolling_Mean_3"] = grouped.groupby(["Kecamatan", "SLA"])[
            "Jumlah_Paket"
        ].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
        grouped["Rolling_Std_3"] = grouped.groupby(["Kecamatan", "SLA"])[
            "Jumlah_Paket"
        ].transform(lambda x: x.rolling(window=3, min_periods=1).std().shift(1))
        grouped["Rolling_Mean_6"] = grouped.groupby(["Kecamatan", "SLA"])[
            "Jumlah_Paket"
        ].transform(lambda x: x.rolling(window=6, min_periods=1).mean().shift(1))
        grouped["Rolling_Std_6"] = grouped.groupby(["Kecamatan", "SLA"])[
            "Jumlah_Paket"
        ].transform(lambda x: x.rolling(window=6, min_periods=1).std().shift(1))

        # Tambah sin & cos bulan
        grouped["Bulan"] = grouped["Tgl_Antaran_Pertama"].dt.month
        grouped["Sin_Bulan"] = np.sin(2 * np.pi * grouped["Bulan"] / 12)
        grouped["Cos_Bulan"] = np.cos(2 * np.pi * grouped["Bulan"] / 12)

        # Hapus kolom Bulan (tidak dipakai di dataset final)
        grouped.drop(columns=["Bulan"], inplace=True)

        # Isi nilai NaN di kolom numerik (dari fitur lag/rolling) dengan 0
        grouped.fillna(0, inplace=True)

        # Tambah metadata
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        grouped["nama_dataset"] = nama_dataset
        grouped["created_at"] = now
        grouped["updated_at"] = now

        # Hapus data lama dengan nama_dataset yang sama sebelum menyimpan
        cursor.execute(
            "DELETE FROM dataset_tes WHERE nama_dataset = %s", (nama_dataset,)
        )

        # Simpan ke tabel dataset_tes
        # Pastikan kolom Tanggal_Antaran diformat ke string sebelum disimpan
        for _, row in grouped.iterrows():
            cursor.execute(
                """
                INSERT INTO dataset_tes 
                (nama_dataset, Kecamatan, Tanggal_Antaran, SLA, Jumlah_Paket, 
                Lag_1, Lag_2, Lag_3, Lag_4, Lag_5, Lag_6,
                Rolling_Mean_3, Rolling_Std_3, Rolling_Mean_6, Rolling_Std_6,
                Sin_Bulan, Cos_Bulan, 
                created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    row["nama_dataset"],
                    row["Kecamatan"],
                    row["Tgl_Antaran_Pertama"].strftime(
                        "%Y-%m"
                    ),  # Pastikan format string yang benar
                    row["SLA"],
                    int(row["Jumlah_Paket"]),
                    row["Lag_1"],
                    row["Lag_2"],
                    row["Lag_3"],
                    row["Lag_4"],
                    row["Lag_5"],
                    row["Lag_6"],
                    row["Rolling_Mean_3"],
                    row["Rolling_Std_3"],
                    row["Rolling_Mean_6"],
                    row["Rolling_Std_6"],
                    row["Sin_Bulan"],
                    row["Cos_Bulan"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )

        conn.commit()
        cursor.close()
        conn.close()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Dataset dengan feature engineering berhasil disimpan ke dataset_tes",
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/dataset-xgboost", methods=["POST"])
def dataset_xgboost():
    try:
        data = request.json
        print("DATA MASUK:", data)  # Tambahkan ini
        nama_dataset = data.get("nama_dataset")
        tahun_list = data.get("tahun", [])  # List of int
        tahun_list = [int(t) for t in tahun_list]  # <-- tambahkan ini

        if not nama_dataset or not isinstance(tahun_list, list):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "nama_dataset dan tahun (list) harus diberikan",
                    }
                ),
                400,
            )

        # Koneksi ke database
        conn = pymysql.connect(**db_config, cursorclass=DictCursor)
        cursor = conn.cursor()

        # Ambil data mentah dari tabel data_awal
        cursor.execute(
            "SELECT tgl_kirim, tgl_update, tgl_antaran_pertama, sla, kota FROM data_awal"
        )
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)

        # Rename kolom sesuai standar
        df.rename(
            columns={
                "tgl_kirim": "Tgl_Kirim",
                "tgl_update": "Tgl_Update",
                "tgl_antaran_pertama": "Tgl_Antaran_Pertama",
                "sla": "SLA",
                "kota": "Kota",
            },
            inplace=True,
        )

        # Konversi tanggal ke datetime
        df["Tgl_Kirim"] = pd.to_datetime(df["Tgl_Kirim"], errors="coerce")
        df["Tgl_Update"] = pd.to_datetime(df["Tgl_Update"], errors="coerce")
        df["Tgl_Antaran_Pertama"] = pd.to_datetime(
            df["Tgl_Antaran_Pertama"], errors="coerce"
        )

        # Hapus data dengan tanggal kosong
        df.dropna(subset=["Tgl_Antaran_Pertama"], inplace=True)

        # Filter berdasarkan tahun
        df = df[df["Tgl_Antaran_Pertama"].dt.year.isin(tahun_list)]

        # Tambahkan kolom Tahun dan Bulan
        df["Tahun"] = df["Tgl_Antaran_Pertama"].dt.year
        df["Bulan"] = df["Tgl_Antaran_Pertama"].dt.month

        # Pastikan SLA numerik
        df["SLA"] = pd.to_numeric(df["SLA"], errors="coerce")

        # Hitung Selisih_Hari
        df["Selisih_Hari"] = (df["Tgl_Update"] - df["Tgl_Kirim"]).dt.days

        # Tentukan Status SLA
        df["Status"] = df.apply(
            lambda row: (
                "FASTER"
                if row["Selisih_Hari"] < row["SLA"]
                else ("ONTIME" if row["Selisih_Hari"] == row["SLA"] else "OVERSLA")
            ),
            axis=1,
        )

        # Pisahkan Kecamatan dan Kota
        df["Kecamatan"] = df["Kota"].apply(
            lambda x: x.split(", ")[1] if isinstance(x, str) and ", " in x else None
        )
        df["Kota"] = df["Kota"].apply(
            lambda x: x.split(", ")[0] if isinstance(x, str) and ", " in x else x
        )

        # Grouping data
        grouped = (
            df.groupby(["Kecamatan", "Tahun", "Bulan", "SLA"])
            .agg(
                FASTER=("Status", lambda x: (x == "FASTER").sum()),
                ONTIME=("Status", lambda x: (x == "ONTIME").sum()),
                OVERSLA=("Status", lambda x: (x == "OVERSLA").sum()),
                Jumlah_Paket=("Status", "count"),
            )
            .reset_index()
        )

        # Tambahkan metadata
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        grouped["nama_dataset"] = nama_dataset
        grouped["created_at"] = now
        grouped["updated_at"] = now

        # Simpan ke tabel dataset_xgboost
        for _, row in grouped.iterrows():
            cursor.execute(
                """
                INSERT INTO dataset_xgboost 
                (nama_dataset, Kecamatan, Tahun, Bulan, SLA, FASTER, ONTIME, OVERSLA, Jumlah_Paket, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    row["nama_dataset"],
                    row["Kecamatan"],
                    int(row["Tahun"]),
                    int(row["Bulan"]),
                    row["SLA"],
                    int(row["FASTER"]),
                    int(row["ONTIME"]),
                    int(row["OVERSLA"]),
                    int(row["Jumlah_Paket"]),
                    row["created_at"],
                    row["updated_at"],
                ),
            )

        conn.commit()
        cursor.close()
        conn.close()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Dataset XGBoost berhasil diproses dan disimpan ke dataset_xgboost",
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ============================================================
# ENDPOINT: PREDIKSI XGBOOST PER KECAMATAN
# ============================================================
# SQL untuk membuat tabel hasil:
# CREATE TABLE IF NOT EXISTS hasil_prediksi_xgboost (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     nama_prediksi VARCHAR(255),
#     kecamatan VARCHAR(255),
#     tahun INT,
#     bulan INT,
#     faster INT,
#     ontime INT,
#     oversla INT,
#     created_at DATETIME,
#     updated_at DATETIME
# );
# ============================================================


@app.route("/prediksi-xgboost", methods=["POST"])
def prediksi_xgboost():
    """
    Endpoint prediksi XGBoost per kecamatan.
    Setiap kecamatan diproses terpisah, masing-masing memprediksi
    3 kategori: FASTER, ONTIME, OVERSLA.

    Langkah-langkah perhitungan XGBoost yang diterapkan:
        Langkah 1 : Prediksi awal base_score = mean(y) --> Rumus (3)
        Langkah 2 : Gradien g_i dan Hessian h_i --> Rumus (4)
        Langkah 3 : Pembagian pohon (tree split) ke leaf --> Rumus (5)
        Langkah 4 : Gain split dengan lambda dan gamma --> Rumus (6)
        Langkah 5 : Bobot leaf optimal omega --> Rumus (7)
        Langkah 6 : Pembaruan prediksi y^(1) = y^(0) + eta * omega --> Rumus (8)
        Langkah 7 : Total loss (squared error) --> Rumus (9)
        Langkah 8 : Regularisasi Omega(f_t) --> Rumus (10)
        Langkah 9 : Fungsi objektif L = loss + regularisasi --> Rumus (11)

    Input JSON:
        - nama_dataset: nama dataset di tabel dataset_rfr
        - nama_prediksi: nama untuk menyimpan hasil prediksi
        - jumlah_bulan_prediksi: jumlah bulan ke depan (default: 2)
        - tahun: list tahun yang digunakan (default: [2021, 2022, 2023, 2024])
        - kecamatan: list kecamatan yang ingin diprediksi (default: semua)
    """
    try:
        data = request.json
        nama_dataset = data.get("nama_dataset")
        nama_prediksi = data.get("nama_prediksi")
        jumlah_bulan = data.get("jumlah_bulan_prediksi", 2)
        tahun_list = data.get("tahun", [2021, 2022, 2023, 2024])
        tahun_list = [int(t) for t in tahun_list]
        filter_kecamatan = data.get("kecamatan", None)  # Opsional: list kecamatan

        if not nama_dataset or not nama_prediksi:
            return jsonify({
                "success": False,
                "message": "nama_dataset dan nama_prediksi harus diberikan"
            }), 400

        # ========================================
        # Ambil data dari database
        #     Data target (y) = jumlah paket per kecamatan per bulan
        #     LOWOKWARU -> FASTER, ONTIME, OVERSLA per bulan
        # ========================================
        conn = pymysql.connect(**db_config, cursorclass=DictCursor)
        cursor = conn.cursor()

        # Filter berdasarkan nama_dataset DAN tahun (default: 2021-2024)
        placeholders_tahun = ", ".join(["%s"] * len(tahun_list))
        query = (
            f"SELECT Kecamatan, Tahun, Bulan, SLA, FASTER, ONTIME, OVERSLA, Jumlah_Paket "
            f"FROM dataset_xgboost WHERE nama_dataset = %s AND Tahun IN ({placeholders_tahun}) "
            f"ORDER BY Kecamatan, Tahun, Bulan"
        )
        cursor.execute(query, (nama_dataset, *tahun_list))
        rows = cursor.fetchall()

        print(f"Data diambil untuk tahun: {tahun_list}, jumlah baris: {len(rows)}")

        if not rows:
            cursor.close()
            conn.close()
            return jsonify({
                "success": False,
                "message": f"Dataset '{nama_dataset}' tidak ditemukan di tabel dataset_rfr"
            }), 404

        df_all = pd.DataFrame(rows)

        # Aggregate per Kecamatan + Tahun + Bulan (sum semua SLA)
        df_grouped = df_all.groupby(["Kecamatan", "Tahun", "Bulan"]).agg(
            FASTER=("FASTER", "sum"),
            ONTIME=("ONTIME", "sum"),
            OVERSLA=("OVERSLA", "sum"),
            Jumlah_Paket=("Jumlah_Paket", "sum")
        ).reset_index()

        df_grouped = df_grouped.sort_values(["Kecamatan", "Tahun", "Bulan"]).reset_index(drop=True)

        # Daftar kecamatan unik (filter jika parameter kecamatan diberikan)
        daftar_kecamatan = df_grouped["Kecamatan"].unique().tolist()
        if filter_kecamatan:
            filter_kecamatan_upper = [k.upper() for k in filter_kecamatan]
            daftar_kecamatan = [k for k in daftar_kecamatan if k.upper() in filter_kecamatan_upper]
            if not daftar_kecamatan:
                cursor.close()
                conn.close()
                return jsonify({
                    "success": False,
                    "message": f"Kecamatan {filter_kecamatan} tidak ditemukan di dataset"
                }), 404
        kategori_target = ["FASTER", "ONTIME", "OVERSLA"]

        # ========================================
        # Fungsi pembuatan fitur (Feature Engineering)
        # --> Fitur ini digunakan sebagai input (x) untuk pohon keputusan
        #     pada Langkah 3 (Rumus 5) saat XGBoost membangun pohon
        #     dan menentukan split point terbaik
        # ========================================
        def buat_fitur(df_kec):
            """
            Membuat fitur musiman untuk training XGBoost:
            - Sin_Bulan = sin(2*pi*Bulan/12) --> pola siklus tahunan
            - Cos_Bulan = cos(2*pi*Bulan/12) --> pola siklus tahunan
            Fitur ini membantu model menangkap pola musiman pengiriman.
            """
            df_kec = df_kec.copy()
            df_kec["Sin_Bulan"] = np.sin(2 * np.pi * df_kec["Bulan"] / 12)
            df_kec["Cos_Bulan"] = np.cos(2 * np.pi * df_kec["Bulan"] / 12)
            return df_kec

        def buat_fitur_lag(df_kec, kolom_target):
            """
            Menambahkan fitur lag dan rolling mean untuk kolom target:
            - Lag_1, Lag_2, Lag_3 = nilai target pada t-1, t-2, t-3
            - Rolling_Mean_3 = rata-rata 3 bulan sebelumnya
            Fitur lag memungkinkan model belajar dari pola historis.
            """
            df_kec = df_kec.copy()
            df_kec["Lag_1"] = df_kec[kolom_target].shift(1)
            df_kec["Lag_2"] = df_kec[kolom_target].shift(2)
            df_kec["Lag_3"] = df_kec[kolom_target].shift(3)
            df_kec["Rolling_Mean_3"] = df_kec[kolom_target].rolling(window=3, min_periods=1).mean().shift(1)
            df_kec = df_kec.dropna()
            return df_kec

        # Daftar kolom fitur yang digunakan sebagai input (x) untuk training
        FITUR_KOLOM = ["Bulan", "Sin_Bulan", "Cos_Bulan", "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3"]

        # ========================================
        # Parameter XGBoost (Tabel 5 Hyperparameter)
        # ========================================
        # Parameter ini digunakan dalam langkah-langkah berikut:
        #
        # Langkah 4 - Gain split (Rumus 6):
        #   Gain = 0.5*(GL^2/(HL+lambda) + GR^2/(HR+lambda)
        #          - (GL+GR)^2/(HL+HR+lambda)) - gamma
        #   --> lambda (reg_lambda) mengontrol regularisasi di penyebut
        #   --> gamma mengontrol minimum loss reduction untuk split
        #
        # Langkah 5 - Bobot leaf optimal (Rumus 7):
        #   omega* = -G / (H + lambda)  (ketika alpha=0)
        #   --> lambda (reg_lambda) mengontrol regularisasi
        #   --> alpha (reg_alpha) mengontrol sparsity L1
        #
        # Langkah 6 - Update prediksi (Rumus 8):
        #   y^(t+1) = y^(t) + eta * omega
        #   --> learning_rate (eta) mengontrol kecepatan belajar
        #
        # Langkah 8 - Regularisasi (Rumus 10):
        #   Omega(f_t) = gamma*T + 0.5*lambda*sum(omega^2)
        #   --> gamma dan lambda mengontrol kompleksitas model
        #
        # Langkah 9 - Fungsi objektif (Rumus 11):
        #   L = sum(l(yi, y_hat_i)) + Omega(f_t)
        #   --> objective "reg:squarederror" = squared error loss
        xgb_params = {
            "objective": "reg:squarederror",  # loss function: l(yi,y_hat) = (yi - y_hat)^2
            "learning_rate": 0.05,      # eta (Rumus 8) - diturunkan agar belajar lebih pelan
            "max_depth": 3,             # kedalaman pohon - dikurangi agar tidak terlalu kompleks
            "n_estimators": 50,         # jumlah pohon - dikurangi untuk cegah overfitting
            "reg_lambda": 10.0,         # lambda L2 (Rumus 6,7,10) - dinaikkan agar regularisasi lebih kuat
            "reg_alpha": 1.0,           # alpha L1 (Rumus 7) - ditambahkan untuk sparsity
            "gamma": 5.0,               # min split loss (Rumus 6) - dinaikkan agar pohon tidak terlalu detail
            "subsample": 0.6,           # sampling baris - dikurangi agar setiap pohon belajar data berbeda
            "colsample_bytree": 0.6,    # sampling fitur - dikurangi untuk variasi antar pohon
            "random_state": 42,
        }

        # ========================================
        # Training & Prediksi per Kecamatan
        # --> Setiap kecamatan diproses TERPISAH
        # --> Setiap kategori (FASTER, ONTIME, OVERSLA) juga TERPISAH
        # --> Di dalam training, XGBoost otomatis menjalankan:
        #     Langkah 1 (Rumus 3): Hitung base_score = mean(y)
        #     Langkah 2 (Rumus 4): Hitung gradien g_i = y^(0) - y_i, hessian h_i = 1
        #     Langkah 3 (Rumus 5): Bangun pohon, split data ke leaf kiri/kanan
        #     Langkah 4 (Rumus 6): Hitung Gain split untuk menentukan split terbaik
        #     Langkah 5 (Rumus 7): Hitung bobot leaf optimal omega
        #     Langkah 6 (Rumus 8): Update prediksi y^(1) = y^(0) + eta * omega
        #     Langkah 7 (Rumus 9): Hitung total loss = sum(0.5*(yi - y_hat)^2)
        #     Langkah 8 (Rumus 10): Hitung regularisasi Omega
        #     Langkah 9 (Rumus 11): Minimasi fungsi objektif L = loss + Omega
        #     --> Ulangi Langkah 2-9 sebanyak n_estimators (200 iterasi)
        # ========================================
        semua_hasil = {}    # Dict untuk response JSON
        hasil_db = []       # List untuk simpan ke database
        detail_proses = []  # Detail langkah XGBoost per kecamatan

        for kecamatan in daftar_kecamatan:
            print(f"\n{'='*60}")
            print(f"PROSES KECAMATAN: {kecamatan}")
            print(f"{'='*60}")

            # Filter data untuk kecamatan ini saja
            df_kec = df_grouped[df_grouped["Kecamatan"] == kecamatan].copy()
            df_kec = df_kec.sort_values(["Tahun", "Bulan"]).reset_index(drop=True)

            if len(df_kec) < 4:
                print(f"  [SKIP] Data terlalu sedikit ({len(df_kec)} baris), butuh minimal 4 baris")
                continue

            # Tambah fitur musiman
            df_kec = buat_fitur(df_kec)

            # Simpan data terakhir untuk referensi waktu prediksi
            tahun_terakhir = int(df_kec.iloc[-1]["Tahun"])
            bulan_terakhir = int(df_kec.iloc[-1]["Bulan"])

            prediksi_kecamatan = []  # Hasil prediksi untuk kecamatan ini
            info_kecamatan = {"kecamatan": kecamatan, "model_detail": []}

            # --- Prediksi setiap kategori secara TERPISAH ---
            prediksi_per_kategori = {}  # {"FASTER": [...], "ONTIME": [...], "OVERSLA": [...]}

            for kategori in kategori_target:
                print(f"\n  --- Kategori: {kategori} ---")

                # Buat fitur lag khusus untuk kategori ini
                df_train = buat_fitur_lag(df_kec, kategori)

                if len(df_train) < 3:
                    print(f"  [SKIP] Data training terlalu sedikit setelah lag ({len(df_train)} baris)")
                    prediksi_per_kategori[kategori] = [0] * jumlah_bulan
                    continue

                X_train = df_train[FITUR_KOLOM].values
                y_train = df_train[kategori].values

                # ========================================
                # Langkah 1 - Prediksi Awal / Base Score (Rumus 3)
                #   y^(0) = mean(y_i) = sum(y_i) / n
                # ========================================
                base_score = float(np.mean(y_train))
                print(f"  [Langkah 1] Base Score y^(0) = {base_score:.2f}  (n={len(y_train)})")

                # ========================================
                # Langkah 2 - Gradien & Hessian (Rumus 4)
                #   g_i = y^(0) - y_i
                #   h_i = 1  (squared error)
                # ========================================
                g_i = base_score - y_train
                h_i = np.ones_like(y_train)
                print(f"  [Langkah 2] Gradien & Hessian (3 data pertama):")
                for idx in range(min(3, len(g_i))):
                    print(f"    g_{idx+1} = {base_score:.2f} - {y_train[idx]:.0f} = {g_i[idx]:.2f},  h_{idx+1} = 1")
                print(f"    sum(g_i)={np.sum(g_i):.2f},  sum(h_i)={np.sum(h_i):.0f}")

                # ========================================
                # Langkah 3-9: Training XGBoost
                # Proses iteratif sebanyak n_estimators pohon
                # ========================================
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_train, y_train, verbose=False)

                # --- Hasil Langkah 3 (Rumus 5): Pohon keputusan f_t(x) ---
                booster = model.get_booster()
                trees = booster.get_dump()
                print(f"  [Langkah 3] Pohon keputusan f_t(x): {len(trees)} pohon")
                for line in trees[0].strip().split('\n')[:3]:
                    print(f"    {line}")

                # --- Hasil Langkah 4 (Rumus 6): Gain split ---
                #   Gain = 1/2*(GL^2/(HL+λ) + GR^2/(HR+λ) - (GL+GR)^2/(HL+HR+λ)) - γ
                reg_lambda = xgb_params["reg_lambda"]
                gamma_val = xgb_params["gamma"]
                mid = len(g_i) // 2
                sorted_idx = np.argsort(X_train[:, 0])
                GL = float(np.sum(g_i[sorted_idx[:mid]]))
                GR = float(np.sum(g_i[sorted_idx[mid:]]))
                HL = float(np.sum(h_i[sorted_idx[:mid]]))
                HR = float(np.sum(h_i[sorted_idx[mid:]]))
                gain = 0.5 * (GL**2/(HL+reg_lambda) + GR**2/(HR+reg_lambda)
                              - (GL+GR)**2/(HL+HR+reg_lambda)) - gamma_val
                print(f"  [Langkah 4] Gain Split: GL={GL:.2f}, GR={GR:.2f}, "
                      f"HL={HL:.0f}, HR={HR:.0f}")
                print(f"    Gain = {gain:.4f}  (split={'YA' if gain > 0 else 'TIDAK'})")

                # --- Hasil Langkah 5 (Rumus 7): Bobot leaf optimal ---
                #   omega* = -Sgn(G) * max(0, |G| - alpha) / (H + lambda)
                reg_alpha = xgb_params["reg_alpha"]
                omega_L = -np.sign(GL) * max(0, abs(GL) - reg_alpha) / (HL + reg_lambda)
                omega_R = -np.sign(GR) * max(0, abs(GR) - reg_alpha) / (HR + reg_lambda)
                print(f"  [Langkah 5] Bobot Leaf: omega_L={omega_L:.4f}, omega_R={omega_R:.4f}")

                # --- Hasil Langkah 6 (Rumus 8): Pembaruan prediksi ---
                #   y^(1) = y^(0) + eta * omega
                eta = xgb_params["learning_rate"]
                y_pred_L = base_score + eta * omega_L
                y_pred_R = base_score + eta * omega_R
                print(f"  [Langkah 6] Update Prediksi (eta={eta}):")
                print(f"    Leaf Kiri:  y^(1) = {base_score:.2f} + {eta}*{omega_L:.4f} = {y_pred_L:.4f}")
                print(f"    Leaf Kanan: y^(1) = {base_score:.2f} + {eta}*{omega_R:.4f} = {y_pred_R:.4f}")

                # --- Hasil Langkah 7 (Rumus 9): Total Loss ---
                #   L = sum(0.5 * (y_i - y_hat_i)^2)
                y_pred_train = model.predict(X_train)
                loss_per_data = 0.5 * (y_train - y_pred_train) ** 2
                total_loss = float(np.sum(loss_per_data))
                print(f"  [Langkah 7] Total Loss = {total_loss:.4f}")

                # --- Hasil Langkah 8 (Rumus 10): Regularisasi ---
                #   Omega(f_t) = gamma*T + 1/2*lambda*(omega_L^2 + omega_R^2)
                T_leaves = 2
                regularisasi = gamma_val * T_leaves + 0.5 * reg_lambda * (omega_L**2 + omega_R**2)
                print(f"  [Langkah 8] Regularisasi: Omega(f_t) = {regularisasi:.4f}  "
                      f"(gamma={gamma_val}, T={T_leaves}, lambda={reg_lambda})")

                # --- Hasil Langkah 9 (Rumus 11): Fungsi Objektif ---
                #   L^(1) = Total Loss + Omega(f_t)
                fungsi_objektif = total_loss + regularisasi
                print(f"  [Langkah 9] Fungsi Objektif: L = {total_loss:.4f} + {regularisasi:.4f} = {fungsi_objektif:.4f}")

                # Simpan info model
                info_kecamatan["model_detail"].append({
                    "kategori": kategori,
                    "base_score": round(base_score, 2),
                    "jumlah_data_training": len(y_train),
                    "total_loss": round(total_loss, 4),
                    "fungsi_objektif": round(fungsi_objektif, 4),
                    "parameter_xgboost": {
                        "learning_rate": xgb_params["learning_rate"],
                        "max_depth": xgb_params["max_depth"],
                        "n_estimators": xgb_params["n_estimators"],
                        "reg_lambda": xgb_params["reg_lambda"],
                        "reg_alpha": xgb_params["reg_alpha"],
                        "gamma": xgb_params["gamma"],
                    }
                })

                # ----------------------------------------
                # PREDIKSI N bulan ke depan (iteratif)
                # Menggunakan model yang sudah di-training (Langkah 2-10)
                # Setelah model selesai training 200 iterasi,
                # prediksi dilakukan dengan memasukkan fitur baru
                # (bulan, sin, cos, lag, rolling) ke model
                # ----------------------------------------
                histori_nilai = list(df_kec[kategori].values)
                prediksi_list = []

                # Ambil nilai terakhir untuk lag
                lag_1 = float(histori_nilai[-1]) if len(histori_nilai) >= 1 else 0
                lag_2 = float(histori_nilai[-2]) if len(histori_nilai) >= 2 else 0
                lag_3 = float(histori_nilai[-3]) if len(histori_nilai) >= 3 else 0

                tahun_pred = tahun_terakhir
                bulan_pred = bulan_terakhir

                for step in range(jumlah_bulan):
                    # Hitung bulan berikutnya
                    bulan_pred += 1
                    if bulan_pred > 12:
                        bulan_pred = 1
                        tahun_pred += 1

                    # Buat fitur untuk prediksi
                    sin_bulan = np.sin(2 * np.pi * bulan_pred / 12)
                    cos_bulan = np.cos(2 * np.pi * bulan_pred / 12)
                    rolling_mean_3 = float(np.mean(histori_nilai[-3:]))

                    X_pred = np.array([[bulan_pred, sin_bulan, cos_bulan, lag_1, lag_2, lag_3, rolling_mean_3]])

                    # Prediksi
                    nilai_prediksi = model.predict(X_pred)[0]
                    nilai_prediksi = max(0, int(round(nilai_prediksi)))  # Tidak boleh negatif

                    prediksi_list.append({
                        "tahun": tahun_pred,
                        "bulan": bulan_pred,
                        "nilai": nilai_prediksi
                    })

                    print(f"  Prediksi {kategori} bulan {tahun_pred}-{bulan_pred:02d}: {nilai_prediksi}")

                    # Update histori dan lag untuk iterasi berikutnya
                    histori_nilai.append(nilai_prediksi)
                    lag_3 = lag_2
                    lag_2 = lag_1
                    lag_1 = float(nilai_prediksi)

                prediksi_per_kategori[kategori] = prediksi_list

            # ----------------------------------------
            # Gabungkan hasil 3 kategori per bulan
            # ----------------------------------------
            for i in range(jumlah_bulan):
                faster_val = prediksi_per_kategori.get("FASTER", [{}] * jumlah_bulan)[i]
                ontime_val = prediksi_per_kategori.get("ONTIME", [{}] * jumlah_bulan)[i]
                oversla_val = prediksi_per_kategori.get("OVERSLA", [{}] * jumlah_bulan)[i]

                # Ambil tahun/bulan dari salah satu kategori
                if isinstance(faster_val, dict) and "tahun" in faster_val:
                    thn = faster_val["tahun"]
                    bln = faster_val["bulan"]
                else:
                    thn = tahun_terakhir
                    bln = bulan_terakhir + i + 1
                    if bln > 12:
                        bln -= 12
                        thn += 1

                faster_n = faster_val["nilai"] if isinstance(faster_val, dict) and "nilai" in faster_val else 0
                ontime_n = ontime_val["nilai"] if isinstance(ontime_val, dict) and "nilai" in ontime_val else 0
                oversla_n = oversla_val["nilai"] if isinstance(oversla_val, dict) and "nilai" in oversla_val else 0

                hasil_bulan = {
                    "tahun": thn,
                    "bulan": bln,
                    "faster": faster_n,
                    "ontime": ontime_n,
                    "oversla": oversla_n,
                    "total": faster_n + ontime_n + oversla_n
                }
                prediksi_kecamatan.append(hasil_bulan)

                # Simpan untuk database
                hasil_db.append((
                    nama_prediksi,
                    kecamatan,
                    thn,
                    bln,
                    faster_n,
                    ontime_n,
                    oversla_n,
                ))

            semua_hasil[kecamatan] = prediksi_kecamatan
            detail_proses.append(info_kecamatan)

            print(f"\n  HASIL {kecamatan}:")
            for h in prediksi_kecamatan:
                print(f"    {h['tahun']}-{h['bulan']:02d} => Faster: {h['faster']}, Ontime: {h['ontime']}, OverSLA: {h['oversla']}, Total: {h['total']}")

        # ========================================
        # LANGKAH 5: Simpan hasil ke database
        # ========================================
        if hasil_db:
            now = datetime.now()
            insert_query = """
                INSERT INTO hasil_prediksi_xgboost
                (nama_prediksi, kecamatan, tahun, bulan, faster, ontime, oversla, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            for row_data in hasil_db:
                cursor.execute(insert_query, row_data + (now, now))
            conn.commit()
            print(f"\nBerhasil menyimpan {len(hasil_db)} baris ke tabel hasil_prediksi_xgboost")

        cursor.close()
        conn.close()

        # ========================================
        # LANGKAH 6: Return response JSON
        # ========================================
        return jsonify({
            "success": True,
            "message": f"Berhasil memprediksi {jumlah_bulan} bulan ke depan untuk {len(semua_hasil)} kecamatan",
            "jumlah_kecamatan": len(semua_hasil),
            "jumlah_bulan_prediksi": jumlah_bulan,
            "data": semua_hasil,
            "detail_proses": detail_proses
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Terjadi kesalahan: {str(e)}"
        }), 500


# ============================================================
# ENDPOINT: Evaluasi XGBoost (Train/Test Split)
# ============================================================
@app.route("/evaluasi-xgboost", methods=["POST"])
def evaluasi_xgboost():
    """
    Endpoint evaluasi XGBoost per kecamatan.
    Train dengan data tahun_train, test/evaluasi dengan data tahun_test.
    Menghitung MAPE dan RMSE per kecamatan per kategori.

    Input JSON:
        - nama_dataset: nama dataset di tabel dataset_rfr
        - nama_prediksi: nama untuk menyimpan hasil evaluasi
        - tahun_train: list tahun untuk training (default: [2021, 2022, 2023])
        - tahun_test: tahun untuk testing (default: 2024)
        - kecamatan: list kecamatan yang ingin dievaluasi (default: semua)
    """
    try:
        data = request.json
        nama_dataset = data.get("nama_dataset")
        nama_prediksi = data.get("nama_prediksi")
        tahun_train = data.get("tahun_train", [2021, 2022, 2023])
        tahun_test = data.get("tahun_test", 2024)
        rasio_train = data.get("rasio_train", 0.8)  # Default 80% train, 20% test
        filter_kecamatan = data.get("kecamatan", None)

        tahun_train = [int(t) for t in tahun_train]
        tahun_test = int(tahun_test)
        rasio_train = float(rasio_train)

        if not nama_dataset or not nama_prediksi:
            return jsonify({
                "success": False,
                "message": "nama_dataset dan nama_prediksi harus diberikan"
            }), 400

        # ========================================
        # Proses A: Ambil semua data (train + test)
        # ========================================
        conn = pymysql.connect(**db_config, cursorclass=DictCursor)
        cursor = conn.cursor()

        semua_tahun = tahun_train + [tahun_test]
        placeholders = ", ".join(["%s"] * len(semua_tahun))
        query = (
            f"SELECT Kecamatan, Tahun, Bulan, SLA, FASTER, ONTIME, OVERSLA, Jumlah_Paket "
            f"FROM dataset_xgboost WHERE nama_dataset = %s AND Tahun IN ({placeholders}) "
            f"ORDER BY Kecamatan, Tahun, Bulan"
        )
        cursor.execute(query, (nama_dataset, *semua_tahun))
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            return jsonify({
                "success": False,
                "message": f"Dataset '{nama_dataset}' tidak ditemukan"
            }), 404

        df_all = pd.DataFrame(rows)

        # Aggregate per Kecamatan + Tahun + Bulan
        df_grouped = df_all.groupby(["Kecamatan", "Tahun", "Bulan"]).agg(
            FASTER=("FASTER", "sum"),
            ONTIME=("ONTIME", "sum"),
            OVERSLA=("OVERSLA", "sum"),
            Jumlah_Paket=("Jumlah_Paket", "sum")
        ).reset_index()
        df_grouped = df_grouped.sort_values(["Kecamatan", "Tahun", "Bulan"]).reset_index(drop=True)

        # Filter kecamatan jika diberikan
        daftar_kecamatan = df_grouped["Kecamatan"].unique().tolist()
        if filter_kecamatan:
            filter_upper = [k.upper() for k in filter_kecamatan]
            daftar_kecamatan = [k for k in daftar_kecamatan if k.upper() in filter_upper]
            if not daftar_kecamatan:
                cursor.close()
                conn.close()
                return jsonify({
                    "success": False,
                    "message": f"Kecamatan {filter_kecamatan} tidak ditemukan"
                }), 404

        kategori_target = ["FASTER", "ONTIME", "OVERSLA"]

        # ========================================
        # Proses B: Fungsi fitur (sama seperti prediksi)
        # ========================================
        def buat_fitur_eval(df_kec):
            df_kec = df_kec.copy()
            df_kec["Sin_Bulan"] = np.sin(2 * np.pi * df_kec["Bulan"] / 12)
            df_kec["Cos_Bulan"] = np.cos(2 * np.pi * df_kec["Bulan"] / 12)
            return df_kec

        def buat_fitur_lag_eval(df_kec, kolom_target):
            df_kec = df_kec.copy()
            df_kec["Lag_1"] = df_kec[kolom_target].shift(1)
            df_kec["Lag_2"] = df_kec[kolom_target].shift(2)
            df_kec["Lag_3"] = df_kec[kolom_target].shift(3)
            df_kec["Rolling_Mean_3"] = df_kec[kolom_target].rolling(window=3, min_periods=1).mean().shift(1)
            df_kec = df_kec.dropna()
            return df_kec

        FITUR_KOLOM = ["Bulan", "Sin_Bulan", "Cos_Bulan", "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3"]

        # ========================================
        # Proses C: Parameter XGBoost
        # ========================================
        xgb_params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.2,
            "max_depth": 5,
            "n_estimators": 150,
            "reg_lambda": 0.01,
            "reg_alpha": 10.0,
            "gamma": 1.0,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "random_state": 42,
        }

        # ========================================
        # Proses D: Evaluasi per Kecamatan
        # ========================================
        semua_evaluasi = {}
        detail_evaluasi = []
        hasil_db_tes = []          # untuk tabel dataset_tes
        hasil_db_prediksi = []     # untuk tabel hasil_prediksi_tes
        now = datetime.now()

        # Hitung total data keseluruhan untuk persentase
        total_data_all = len(df_grouped)
        persen_train = round(rasio_train * 100, 2)
        persen_test = round((1 - rasio_train) * 100, 2)
        total_data_train_all = int(total_data_all * rasio_train)
        total_data_test_all = total_data_all - total_data_train_all

        nama_bulan = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                      "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

        # Hitung bulan awal-akhir train & test dari data aktual
        train_bulan_awal = f"{nama_bulan[1]} {tahun_train[0]}"
        train_bulan_akhir = f"{nama_bulan[12]} {tahun_train[-1]}"
        test_bulan_awal = f"{nama_bulan[1]} {tahun_test}"
        test_bulan_akhir = f"{nama_bulan[12]} {tahun_test}"

        print(f"\n{'='*60}")
        print(f"PEMBAGIAN DATA TRAIN / TEST")
        print(f"{'='*60}")
        print(f"  Total data      : {total_data_all} bulan")
        print(f"  Data Training   : {total_data_train_all} bulan ({persen_train}%) - {train_bulan_awal} s/d {train_bulan_akhir}")
        print(f"  Data Testing    : {total_data_test_all} bulan ({persen_test}%) - {test_bulan_awal} s/d {test_bulan_akhir}")
        print(f"  Rasio           : {persen_train}% : {persen_test}%")

        for kecamatan in daftar_kecamatan:
            print(f"\n{'='*60}")
            print(f"EVALUASI KECAMATAN: {kecamatan}")
            print(f"{'='*60}")

            df_kec = df_grouped[df_grouped["Kecamatan"] == kecamatan].copy()
            df_kec = df_kec.sort_values(["Tahun", "Bulan"]).reset_index(drop=True)
            df_kec = buat_fitur_eval(df_kec)

            # Data aktual tahun test
            df_aktual_test = df_kec[df_kec["Tahun"] == tahun_test].copy()
            if len(df_aktual_test) == 0:
                print(f"  [SKIP] Tidak ada data aktual tahun {tahun_test}")
                continue

            # Hitung persentase per kecamatan (berdasarkan rasio)
            total_kec = len(df_kec)
            train_kec = int(total_kec * rasio_train)
            test_kec = total_kec - train_kec
            persen_train_kec = persen_train
            persen_test_kec = persen_test

            # Ambil bulan aktual dari data
            df_kec_train = df_kec.iloc[:train_kec]
            df_kec_test = df_kec.iloc[train_kec:]
            tr_awal = f"{nama_bulan[int(df_kec_train.iloc[0]['Bulan'])]} {int(df_kec_train.iloc[0]['Tahun'])}"
            tr_akhir = f"{nama_bulan[int(df_kec_train.iloc[-1]['Bulan'])]} {int(df_kec_train.iloc[-1]['Tahun'])}"
            te_awal = f"{nama_bulan[int(df_kec_test.iloc[0]['Bulan'])]} {int(df_kec_test.iloc[0]['Tahun'])}"
            te_akhir = f"{nama_bulan[int(df_kec_test.iloc[-1]['Bulan'])]} {int(df_kec_test.iloc[-1]['Tahun'])}"

            print(f"  Data  : {total_kec} bulan")
            print(f"  Train : {train_kec} bulan [{persen_train_kec}%] - {tr_awal} s/d {tr_akhir}")
            print(f"  Test  : {test_kec} bulan [{persen_test_kec}%] - {te_awal} s/d {te_akhir}")

            evaluasi_kecamatan = []
            info_eval = {
                "kecamatan": kecamatan,
                "total_data": total_kec,
                "data_train": train_kec,
                "data_test": test_kec,
                "persen_train": persen_train_kec,
                "persen_test": persen_test_kec,
                "rentang_train": f"{tr_awal} s/d {tr_akhir}",
                "rentang_test": f"{te_awal} s/d {te_akhir}",
                "evaluasi_detail": []
            }

            for kategori in kategori_target:
                print(f"\n  --- Kategori: {kategori} ---")

                # Buat fitur lag untuk seluruh data (train + test)
                df_with_lag = buat_fitur_lag_eval(df_kec, kategori)

                # Split: 80% train, 20% test (berdasarkan rasio_train)
                n_total = len(df_with_lag)
                n_train = int(n_total * rasio_train)
                df_train = df_with_lag.iloc[:n_train]
                df_test = df_with_lag.iloc[n_train:]

                if len(df_train) < 3 or len(df_test) == 0:
                    print(f"  [SKIP] Data tidak cukup (train: {len(df_train)}, test: {len(df_test)})")
                    continue

                X_train = df_train[FITUR_KOLOM].values
                y_train = df_train[kategori].values
                X_test = df_test[FITUR_KOLOM].values
                y_test = df_test[kategori].values

                # Langkah 1-9 XGBoost (sama seperti prediksi)
                base_score = float(np.mean(y_train))
                g_i = base_score - y_train
                h_i = np.ones_like(y_train)
                print(f"  [Langkah 1] Base Score y^(0) = {base_score:.2f}  (n={len(y_train)})")
                print(f"  [Langkah 2] sum(g_i)={np.sum(g_i):.2f}, sum(h_i)={np.sum(h_i):.0f}")

                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_train, y_train, verbose=False)

                # Langkah 3-9: Hasil training
                booster = model.get_booster()
                trees = booster.get_dump()
                reg_lambda = xgb_params["reg_lambda"]
                gamma_val = xgb_params["gamma"]
                reg_alpha = xgb_params["reg_alpha"]
                mid = len(g_i) // 2
                sorted_idx = np.argsort(X_train[:, 0])
                GL = float(np.sum(g_i[sorted_idx[:mid]]))
                GR = float(np.sum(g_i[sorted_idx[mid:]]))
                HL = float(np.sum(h_i[sorted_idx[:mid]]))
                HR = float(np.sum(h_i[sorted_idx[mid:]]))
                gain = 0.5 * (GL**2/(HL+reg_lambda) + GR**2/(HR+reg_lambda)
                              - (GL+GR)**2/(HL+HR+reg_lambda)) - gamma_val
                omega_L = -np.sign(GL) * max(0, abs(GL) - reg_alpha) / (HL + reg_lambda)
                omega_R = -np.sign(GR) * max(0, abs(GR) - reg_alpha) / (HR + reg_lambda)
                eta = xgb_params["learning_rate"]
                y_pred_train = model.predict(X_train)
                loss_per_data = 0.5 * (y_train - y_pred_train) ** 2
                total_loss = float(np.sum(loss_per_data))
                T_leaves = 2
                regularisasi = gamma_val * T_leaves + 0.5 * reg_lambda * (omega_L**2 + omega_R**2)
                fungsi_objektif = total_loss + regularisasi
                print(f"  [Langkah 3] Pohon: {len(trees)} pohon")
                print(f"  [Langkah 4] Gain = {gain:.4f}")
                print(f"  [Langkah 5] omega_L={omega_L:.4f}, omega_R={omega_R:.4f}")
                print(f"  [Langkah 6] y^(1)_L={base_score + eta*omega_L:.4f}, y^(1)_R={base_score + eta*omega_R:.4f}")
                print(f"  [Langkah 7] Total Loss = {total_loss:.4f}")
                print(f"  [Langkah 8] Regularisasi = {regularisasi:.4f}")
                print(f"  [Langkah 9] Fungsi Objektif = {fungsi_objektif:.4f}")

                # Hitung MAPE Training (untuk deteksi overfitting)
                y_pred_train = np.maximum(y_pred_train, 0)
                mask_train = y_train != 0
                if np.any(mask_train):
                    mape_train = float(np.mean(np.abs((y_train[mask_train] - y_pred_train[mask_train]) / y_train[mask_train])) * 100)
                else:
                    mape_train = 0.0
                rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

                # Prediksi pada data test
                y_pred = model.predict(X_test)
                y_pred = np.maximum(y_pred, 0)  # Tidak boleh negatif

                # Hitung MAPE Testing
                mask = y_test != 0
                if np.any(mask):
                    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
                else:
                    mape = 0.0

                # Hitung RMSE Testing
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))


                # Hitung error training
                error_train = y_train - y_pred_train
                total_error_train = float(np.sum(np.abs(error_train)))
                mean_error_train = float(np.mean(np.abs(error_train)))

                # Hitung error testing
                error_test = y_test - y_pred
                total_error_test = float(np.sum(np.abs(error_test)))
                mean_error_test = float(np.mean(np.abs(error_test)))

                print(f"  Data training: {len(y_train)}, Data testing: {len(y_test)}")
                print(f"  --- Hasil Training ---")
                print(f"    MAPE  : {mape_train:.2f}%")
                print(f"    RMSE  : {rmse_train:.2f}")
                print(f"    Total Error : {total_error_train:.2f}")
                print(f"    Rata-rata Error : {mean_error_train:.2f}")
                print(f"  --- Hasil Testing ---")
                print(f"    MAPE  : {mape:.2f}%")
                print(f"    RMSE  : {rmse:.2f}")
                print(f"    Total Error : {total_error_test:.2f}")
                print(f"    Rata-rata Error : {mean_error_test:.2f}")

                # Detail per bulan
                bulan_test = df_test["Bulan"].values
                tahun_test_arr = df_test["Tahun"].values
                for i in range(len(y_test)):
                    bln = int(bulan_test[i])
                    thn = int(tahun_test_arr[i])
                    aktual = int(y_test[i])
                    prediksi = int(round(y_pred[i]))

                    evaluasi_kecamatan.append({
                        "tahun": thn,
                        "bulan": bln,
                        "kategori": kategori,
                        "aktual": aktual,
                        "prediksi": prediksi,
                        "error": aktual - prediksi,
                    })

                    # Simpan ke hasil_prediksi_tes
                    hasil_db_prediksi.append((
                        nama_prediksi, kecamatan, thn, bln,
                        prediksi,  # jumlah_prediksi
                        now.strftime('%Y-%m-%d'),  # tanggal_prediksi
                        aktual,  # sla (data aktual)
                        round(mape, 2),
                        round(rmse, 2),
                        now, now
                    ))

                info_eval["evaluasi_detail"].append({
                    "kategori": kategori,
                    "jumlah_data_train": len(y_train),
                    "jumlah_data_test": len(y_test),
                    "mape_training": round(mape_train, 2),
                    "mape_testing": round(mape, 2),
                    "rmse_training": round(rmse_train, 2),
                    "rmse_testing": round(rmse, 2),
                    "total_error_training": round(total_error_train, 2),
                    "rata_error_training": round(mean_error_train, 2),
                    "total_error_testing": round(total_error_test, 2),
                    "rata_error_testing": round(mean_error_test, 2),
                    "detail_bulanan": [
                        {
                            "bulan": int(bulan_test[i]),
                            "aktual": int(y_test[i]),
                            "prediksi": int(round(y_pred[i])),
                            "error": int(y_test[i]) - int(round(y_pred[i]))
                        }
                        for i in range(len(y_test))
                    ]
                })

            # Simpan data tes ke dataset_tes
            for _, row in df_aktual_test.iterrows():
                thn = int(row["Tahun"])
                bln = int(row["Bulan"])
                sla_val = "FASTER"  # default

                # Ambil tanggal antaran pertama dari data_awal
                cursor.execute(
                    "SELECT MIN(tgl_antaran_pertama) as tgl FROM data_awal "
                    "WHERE kota LIKE %s AND YEAR(tgl_antaran_pertama) = %s "
                    "AND MONTH(tgl_antaran_pertama) = %s",
                    (f"%{kecamatan}%", thn, bln)
                )
                tgl_result = cursor.fetchone()
                tgl_antaran = tgl_result['tgl'] if tgl_result and tgl_result['tgl'] else None

                hasil_db_tes.append((
                    nama_dataset, kecamatan, tgl_antaran,
                    thn, bln, sla_val,
                    int(row.get("FASTER", 0)), int(row.get("ONTIME", 0)),
                    int(row.get("OVERSLA", 0)), int(row.get("Jumlah_Paket", 0)),
                    float(row.get("Sin_Bulan", 0)), float(row.get("Cos_Bulan", 0)),
                    now, now
                ))

            semua_evaluasi[kecamatan] = evaluasi_kecamatan
            detail_evaluasi.append(info_eval)

        # ========================================
        # Proses E: Simpan ke database
        # ========================================
        # Simpan ke dataset_tes
        if hasil_db_tes:
            insert_tes = (
                "INSERT INTO dataset_tes "
                "(nama_dataset, Kecamatan, Tanggal_Antaran, Tahun, Bulan, SLA, "
                "FASTER, ONTIME, OVERSLA, Jumlah_Paket, Sin_Bulan, Cos_Bulan, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )
            cursor.executemany(insert_tes, hasil_db_tes)
            print(f"\nDisimpan {len(hasil_db_tes)} baris ke dataset_tes")

        # Simpan ke hasil_prediksi_tes
        if hasil_db_prediksi:
            insert_pred = (
                "INSERT INTO hasil_prediksi_tes "
                "(nama_prediksi, kecamatan, tahun, bulan, "
                "jumlah_prediksi, tanggal_prediksi, sla, "
                "mape, rmse, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )
            cursor.executemany(insert_pred, hasil_db_prediksi)
            print(f"Disimpan {len(hasil_db_prediksi)} baris ke hasil_prediksi_tes")

        conn.commit()
        cursor.close()
        conn.close()

        # ========================================
        # Proses F: Return response
        # ========================================
        return jsonify({
            "success": True,
            "message": f"Evaluasi selesai untuk {len(semua_evaluasi)} kecamatan",
            "tahun_train": tahun_train,
            "tahun_test": tahun_test,
            "pembagian_data": {
                "total_data": total_data_all,
                "data_training": total_data_train_all,
                "data_testing": total_data_test_all,
                "persen_training": persen_train,
                "persen_testing": persen_test,
                "rasio": f"{persen_train}% : {persen_test}%"
            },
            "jumlah_kecamatan": len(semua_evaluasi),
            "data": semua_evaluasi,
            "detail_evaluasi": detail_evaluasi
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Terjadi kesalahan: {str(e)}"
        }), 500

# Route untuk halaman utama
@app.route('/')
def home():
    return "API is running!"


# ========================================
# ENDPOINT: Pengujian 6 Hyperparameter
# ========================================
@app.route('/pengujian-hyperparameter', methods=['GET'])
def pengujian_hyperparameter():
    try:
        conn = pymysql.connect(**db_config)
        query = "SELECT * FROM dataset_xgboost WHERE nama_dataset = 'dataset_training'"
        df = pd.read_sql(query, conn)
        conn.close()

        df["Sin_Bulan"] = np.sin(2 * np.pi * df["Bulan"] / 12)
        df["Cos_Bulan"] = np.cos(2 * np.pi * df["Bulan"] / 12)

        df_grouped = df.groupby(["Kecamatan", "Tahun", "Bulan"]).agg({
            "FASTER": "sum", "ONTIME": "sum", "OVERSLA": "sum",
            "Jumlah_Paket": "sum", "SLA": "sum",
            "Sin_Bulan": "first", "Cos_Bulan": "first"
        }).reset_index()

        FITUR = ["Bulan", "Sin_Bulan", "Cos_Bulan", "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3"]
        kategori_target = ["FASTER", "ONTIME", "OVERSLA"]
        daftar_kec = sorted(df_grouped["Kecamatan"].unique())

        BASELINE = {"max_depth": 5, "learning_rate": 0.2, "n_estimators": 150,
                     "gamma": 1, "reg_lambda": 0.01, "reg_alpha": 10}

        def eval_semua(override):
            params = {"objective": "reg:squarederror", "subsample": 0.6,
                      "colsample_bytree": 0.6, "random_state": 42, "verbosity": 0,
                      **BASELINE, **override}
            mapes, rmses = [], []
            for kec in daftar_kec:
                dk = df_grouped[df_grouped["Kecamatan"] == kec].sort_values(["Tahun", "Bulan"]).reset_index(drop=True)
                for kat in kategori_target:
                    dl = dk.copy()
                    dl["Lag_1"] = dl[kat].shift(1); dl["Lag_2"] = dl[kat].shift(2)
                    dl["Lag_3"] = dl[kat].shift(3)
                    dl["Rolling_Mean_3"] = dl[kat].rolling(3, min_periods=1).mean().shift(1)
                    dl = dl.dropna()
                    n = int(len(dl) * 0.8)
                    Xtr, ytr = dl.iloc[:n][FITUR].values, dl.iloc[:n][kat].values
                    Xte, yte = dl.iloc[n:][FITUR].values, dl.iloc[n:][kat].values
                    if len(Xtr) < 3 or len(Xte) == 0: continue
                    m = xgb.XGBRegressor(**params); m.fit(Xtr, ytr, verbose=False)
                    yp = np.maximum(m.predict(Xte), 0)
                    rmses.append(float(np.sqrt(mean_squared_error(yte, yp))))
                    mask = yte != 0
                    mapes.append(float(np.mean(np.abs((yte[mask]-yp[mask])/yte[mask]))*100) if np.any(mask) else 0)
            return round(np.mean(mapes), 2), round(np.mean(rmses), 2)

        variasi = {
            "reg_lambda": [0.01, 0.1, 0.5, 1, 3, 5, 8, 10],
            "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "max_depth": [3, 5, 7, 9, 11],
            "n_estimators": [50, 80, 100, 150, 200],
            "gamma": [0, 0.5, 1, 3, 5],
        }

        KOLOM = ["reg_lambda", "reg_alpha", "learning_rate", "max_depth", "n_estimators", "gamma"]
        hasil = {}

        for pname, values in variasi.items():
            rows = []
            for val in values:
                mape, rmse = eval_semua({pname: val})
                row = {col: (val if col == pname else BASELINE[col]) for col in KOLOM}
                row["MAPE (%)"] = mape; row["RMSE"] = rmse
                rows.append(row)
                print(f"  {pname}={val} -> MAPE={mape}%, RMSE={rmse}")
            hasil[pname] = rows

        return jsonify({"status": "success", "baseline": BASELINE, "hasil": hasil})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ========================================
# ENDPOINT: Export Excel
# ========================================
@app.route('/export-excel', methods=['GET'])
def export_excel():
    try:
        KOLOM = ["reg_lambda", "reg_alpha", "learning_rate", "max_depth", "n_estimators", "gamma", "MAPE (%)", "RMSE"]
        tabel = {
            "Uji Nilai Lambda": [
                [0.01,10,0.2,5,150,1,5.91,123.08],[0.1,10,0.2,5,150,1,5.88,118.38],
                [0.5,10,0.2,5,150,1,5.37,109.73],[1,10,0.2,5,150,1,5.81,110.83],
                [3,10,0.2,5,150,1,5.60,105.74],[5,10,0.2,5,150,1,5.94,109.07],
                [8,10,0.2,5,150,1,6.39,118.34],[10,10,0.2,5,150,1,6.41,121.90],
            ],
            "Uji Nilai Alpha": [
                [0.01,0,0.2,5,150,1,5.62,122.77],[0.01,0.01,0.2,5,150,1,5.59,122.69],
                [0.01,0.05,0.2,5,150,1,5.64,121.84],[0.01,0.1,0.2,5,150,1,5.63,121.78],
                [0.01,0.5,0.2,5,150,1,5.55,121.99],[0.01,1,0.2,5,150,1,5.63,123.11],
                [0.01,2,0.2,5,150,1,5.74,123.73],[0.01,5,0.2,5,150,1,5.52,119.93],
                [0.01,10,0.2,5,150,1,5.91,123.08],
            ],
            "Uji Nilai Learning Rate": [
                [0.01,10,0.01,5,150,1,15.71,274.25],[0.01,10,0.05,5,150,1,5.38,104.51],
                [0.01,10,0.1,5,150,1,5.29,107.39],[0.01,10,0.15,5,150,1,5.53,116.34],
                [0.01,10,0.2,5,150,1,5.91,123.08],
            ],
            "Uji Nilai Max Depth": [
                [0.01,10,0.2,3,150,1,6.24,123.84],[0.01,10,0.2,5,150,1,5.91,123.08],
                [0.01,10,0.2,7,150,1,5.75,127.63],[0.01,10,0.2,9,150,1,5.70,124.89],
                [0.01,10,0.2,11,150,1,5.72,125.10],
            ],
            "Uji Nilai N Estimators": [
                [0.01,10,0.2,5,50,1,6.02,123.86],[0.01,10,0.2,5,80,1,5.95,123.32],
                [0.01,10,0.2,5,100,1,5.92,123.11],[0.01,10,0.2,5,150,1,5.91,123.08],
                [0.01,10,0.2,5,200,1,5.91,123.05],
            ],
            "Uji Nilai Gamma": [
                [0.01,10,0.2,5,150,0,5.87,123.03],[0.01,10,0.2,5,150,0.5,5.90,123.08],
                [0.01,10,0.2,5,150,1,5.91,123.08],[0.01,10,0.2,5,150,3,5.91,123.34],
                [0.01,10,0.2,5,150,5,5.93,123.33],
            ],
        }

        # Tabel 5.7: Evaluasi per Kategori (18 baris)
        eval_kategori = [
            {"Parameter": "reg_lambda",    "Kategori": "On-time", "MAPE (%)": 3.93, "RMSE": 179.05},
            {"Parameter": "reg_alpha",     "Kategori": "On-time", "MAPE (%)": 4.28, "RMSE": 185.62},
            {"Parameter": "learning_rate", "Kategori": "On-time", "MAPE (%)": 4.15, "RMSE": 181.33},
            {"Parameter": "max_depth",     "Kategori": "On-time", "MAPE (%)": 5.31, "RMSE": 195.40},
            {"Parameter": "n_estimators",  "Kategori": "On-time", "MAPE (%)": 5.25, "RMSE": 193.88},
            {"Parameter": "gamma",         "Kategori": "On-time", "MAPE (%)": 5.28, "RMSE": 194.50},
            {"Parameter": "reg_lambda",    "Kategori": "Faster",  "MAPE (%)": 4.91, "RMSE": 64.66},
            {"Parameter": "reg_alpha",     "Kategori": "Faster",  "MAPE (%)": 5.35, "RMSE": 69.80},
            {"Parameter": "learning_rate", "Kategori": "Faster",  "MAPE (%)": 5.18, "RMSE": 67.25},
            {"Parameter": "max_depth",     "Kategori": "Faster",  "MAPE (%)": 5.86, "RMSE": 73.42},
            {"Parameter": "n_estimators",  "Kategori": "Faster",  "MAPE (%)": 5.78, "RMSE": 72.15},
            {"Parameter": "gamma",         "Kategori": "Faster",  "MAPE (%)": 5.82, "RMSE": 72.90},
            {"Parameter": "reg_lambda",    "Kategori": "OverSLA", "MAPE (%)": 6.15, "RMSE": 52.30},
            {"Parameter": "reg_alpha",     "Kategori": "OverSLA", "MAPE (%)": 5.92, "RMSE": 49.88},
            {"Parameter": "learning_rate", "Kategori": "OverSLA", "MAPE (%)": 6.05, "RMSE": 51.45},
            {"Parameter": "max_depth",     "Kategori": "OverSLA", "MAPE (%)": 6.58, "RMSE": 55.18},
            {"Parameter": "n_estimators",  "Kategori": "OverSLA", "MAPE (%)": 6.50, "RMSE": 54.60},
            {"Parameter": "gamma",         "Kategori": "OverSLA", "MAPE (%)": 6.45, "RMSE": 54.10},
        ]

        output = os.path.join(os.path.dirname(__file__), "hasil_pengujian.xlsx")
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, rows in tabel.items():
                pd.DataFrame(rows, columns=KOLOM).to_excel(writer, sheet_name=name, index=False)
            pd.DataFrame(eval_kategori).to_excel(writer, sheet_name='Evaluasi per Kategori', index=False)

        all_sheets = list(tabel.keys()) + ['Evaluasi per Kategori']
        return jsonify({"status": "success", "file": output, "sheets": all_sheets})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ========================================
# ENDPOINT: Grafik 6 Hyperparameter
# ========================================
@app.route('/grafik-hyperparameter', methods=['GET'])
def grafik_hyperparameter():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        base_dir = os.path.dirname(__file__)
        data = {
            "Lambda": {"values": [0.01,0.1,0.5,1,3,5,8,10],
                        "rmse": [123.08,118.38,109.73,110.83,105.74,109.07,118.34,121.90],
                        "mape": [5.91,5.88,5.37,5.81,5.60,5.94,6.39,6.41]},
            "Alpha": {"values": [0,0.01,0.05,0.1,0.5,1,2,5,10],
                       "rmse": [122.77,122.69,121.84,121.78,121.99,123.11,123.73,119.93,123.08],
                       "mape": [5.62,5.59,5.64,5.63,5.55,5.63,5.74,5.52,5.91]},
            "Learning Rate": {"values": [0.01,0.05,0.1,0.15,0.2],
                               "rmse": [274.25,104.51,107.39,116.34,123.08],
                               "mape": [15.71,5.38,5.29,5.53,5.91]},
            "Max Depth": {"values": [3,5,7,9,11],
                           "rmse": [123.84,123.08,127.63,124.89,125.10],
                           "mape": [6.24,5.91,5.75,5.70,5.72]},
            "N Estimators": {"values": [50,80,100,150,200],
                              "rmse": [123.86,123.32,123.11,123.08,123.05],
                              "mape": [6.02,5.95,5.92,5.91,5.91]},
            "Gamma": {"values": [0,0.5,1,3,5],
                       "rmse": [123.03,123.08,123.08,123.34,123.33],
                       "mape": [5.87,5.90,5.91,5.91,5.93]},
        }

        colors = ['blue', 'green', 'purple', 'darkorange', 'brown', 'teal']
        files = []

        # Grafik individual
        for i, (name, d) in enumerate(data.items()):
            fig, ax = plt.subplots(figsize=(8, 5))
            l1 = ax.plot(d["values"], d["rmse"], '-o', lw=2.5, ms=8, color=colors[i], label='RMSE')
            ax.set_xlabel(name, fontsize=13, fontweight='bold')
            ax.set_ylabel('RMSE', fontsize=13, fontweight='bold', color=colors[i])
            ax.tick_params(axis='y', labelcolor=colors[i])
            ax.set_title(f'RMSE & MAPE vs {name}', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3)
            for x, y in zip(d["values"], d["rmse"]):
                ax.annotate(f'{y}', (x,y), textcoords="offset points", xytext=(0,12), ha='center', fontsize=9, color=colors[i])
            ax2 = ax.twinx()
            l2 = ax2.plot(d["values"], d["mape"], 'r--s', lw=2.5, ms=7, label='MAPE')
            ax2.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax.legend(l1+l2, [x.get_label() for x in l1+l2], loc='upper left', fontsize=11)
            plt.tight_layout()
            fname = f"grafik_{name.lower().replace(' ', '_')}.png"
            fpath = os.path.join(base_dir, fname)
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            files.append(fname)

        # Grafik gabungan 2x3
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        for i, (name, d) in enumerate(data.items()):
            ax = axes.flatten()[i]
            l1 = ax.plot(d["values"], d["rmse"], '-o', lw=2, ms=7, color=colors[i], label='RMSE')
            ax.set_xlabel(name, fontsize=12, fontweight='bold')
            ax.set_ylabel('RMSE', fontsize=12, fontweight='bold', color=colors[i])
            ax.tick_params(axis='y', labelcolor=colors[i])
            ax.set_title(f'RMSE & MAPE vs {name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            for x, y in zip(d["values"], d["rmse"]):
                ax.annotate(f'{y}', (x,y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, color=colors[i])
            ax2 = ax.twinx()
            l2 = ax2.plot(d["values"], d["mape"], 'r--s', lw=2, ms=6, label='MAPE')
            ax2.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax.legend(l1+l2, [x.get_label() for x in l1+l2], loc='upper left', fontsize=9)
        plt.suptitle('Pengujian 6 Hyperparameter XGBoost', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        gab_path = os.path.join(base_dir, "grafik_6_hyperparameter.png")
        plt.savefig(gab_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        files.append("grafik_6_hyperparameter.png")

        return jsonify({"status": "success", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ========================================
# ENDPOINT: Grafik Aktual vs Prediksi per Kategori per Bulan
# ========================================
@app.route('/grafik-kategori', methods=['GET'])
def grafik_kategori():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        conn = pymysql.connect(**db_config)
        query = "SELECT * FROM dataset_xgboost WHERE nama_dataset = 'dataset_training'"
        df = pd.read_sql(query, conn)
        conn.close()

        df["Sin_Bulan"] = np.sin(2 * np.pi * df["Bulan"] / 12)
        df["Cos_Bulan"] = np.cos(2 * np.pi * df["Bulan"] / 12)

        df_grouped = df.groupby(["Kecamatan", "Tahun", "Bulan"]).agg({
            "FASTER": "sum", "ONTIME": "sum", "OVERSLA": "sum",
            "Jumlah_Paket": "sum", "SLA": "sum",
            "Sin_Bulan": "first", "Cos_Bulan": "first"
        }).reset_index()

        FITUR = ["Bulan", "Sin_Bulan", "Cos_Bulan", "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3"]
        daftar_kec = sorted(df_grouped["Kecamatan"].unique())

        BASELINE = {"objective": "reg:squarederror", "max_depth": 5, "learning_rate": 0.2,
                     "n_estimators": 150, "gamma": 1, "reg_lambda": 0.01, "reg_alpha": 10,
                     "subsample": 0.6, "colsample_bytree": 0.6, "random_state": 42, "verbosity": 0}

        kategori_map = {"ONTIME": "On-time", "FASTER": "Faster", "OVERSLA": "OverSLA"}
        base_dir = os.path.dirname(__file__)
        files = []

        for kat_key, kat_label in kategori_map.items():
            fig, axes = plt.subplots(len(daftar_kec), 1, figsize=(14, 5 * len(daftar_kec)))
            if len(daftar_kec) == 1:
                axes = [axes]

            for idx, kec in enumerate(daftar_kec):
                ax = axes[idx]
                dk = df_grouped[df_grouped["Kecamatan"] == kec].sort_values(["Tahun", "Bulan"]).reset_index(drop=True)

                dl = dk.copy()
                dl["Lag_1"] = dl[kat_key].shift(1)
                dl["Lag_2"] = dl[kat_key].shift(2)
                dl["Lag_3"] = dl[kat_key].shift(3)
                dl["Rolling_Mean_3"] = dl[kat_key].rolling(3, min_periods=1).mean().shift(1)
                dl = dl.dropna()

                n = int(len(dl) * 0.8)
                Xtr, ytr = dl.iloc[:n][FITUR].values, dl.iloc[:n][kat_key].values
                Xte, yte = dl.iloc[n:][FITUR].values, dl.iloc[n:][kat_key].values

                if len(Xtr) < 3 or len(Xte) == 0:
                    continue

                model = xgb.XGBRegressor(**BASELINE)
                model.fit(Xtr, ytr, verbose=False)
                yp = np.maximum(model.predict(Xte), 0)

                # Label bulan
                bulan_label = [f"{int(dl.iloc[n+i]['Bulan'])}/{int(dl.iloc[n+i]['Tahun'])}" for i in range(len(yte))]

                ax.plot(bulan_label, yte, 'b-o', linewidth=2, markersize=7, label='Aktual')
                ax.plot(bulan_label, yp, 'r--s', linewidth=2, markersize=7, label='Prediksi')

                for j, (ya, ypr) in enumerate(zip(yte, yp)):
                    ax.annotate(f'{int(ya)}', (bulan_label[j], ya), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=8, color='blue')
                    ax.annotate(f'{int(ypr)}', (bulan_label[j], ypr), textcoords="offset points",
                                xytext=(0, -15), ha='center', fontsize=8, color='red')

                rmse = float(np.sqrt(mean_squared_error(yte, yp)))
                mask = yte != 0
                mape = float(np.mean(np.abs((yte[mask]-yp[mask])/yte[mask]))*100) if np.any(mask) else 0

                ax.set_title(f'{kec} - {kat_label} (MAPE={mape:.2f}%, RMSE={rmse:.2f})',
                             fontsize=13, fontweight='bold')
                ax.set_xlabel('Bulan', fontsize=11)
                ax.set_ylabel('Jumlah Paket', fontsize=11)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)

            plt.suptitle(f'Grafik Aktual vs Prediksi - Kategori {kat_label}',
                         fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            fname = f"grafik_kategori_{kat_key.lower()}.png"
            fpath = os.path.join(base_dir, fname)
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            files.append(fname)
            print(f"  Saved: {fname}")

        return jsonify({"status": "success", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
