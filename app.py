from flask import Flask, request, jsonify, render_template
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
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

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable CORS if needed
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150MB
app.secret_key = 'prediksi-svr-secret-key-2024'

# Register web routes blueprint
from web_routes import web
app.register_blueprint(web)



# Database configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "prediksi_svr",
}


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

        # Tambah metadata dan ekstrak Tahun/Bulan
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        grouped["nama_dataset"] = nama_dataset
        grouped["Tahun"] = grouped["Tgl_Antaran_Pertama"].dt.year
        grouped["Bulan"] = grouped["Tgl_Antaran_Pertama"].dt.month
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
                (nama_dataset, Kecamatan, Tanggal_Antaran, Tahun, Bulan, SLA, Jumlah_Paket, 
                Lag_1, Lag_2, Lag_3, Lag_4, Lag_5, Lag_6,
                Rolling_Mean_3, Rolling_Std_3, Rolling_Mean_6, Rolling_Std_6,
                Sin_Bulan, Cos_Bulan, 
                created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    row["nama_dataset"],
                    row["Kecamatan"],
                    row["Tgl_Antaran_Pertama"].strftime("%Y-%m-01"),
                    row["Tahun"],
                    row["Bulan"],
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

        # Aggregate per Kecamatan + Tahun + Bulan + SLA
        df_grouped = df_all.groupby(["Kecamatan", "Tahun", "Bulan", "SLA"]).agg(
            FASTER=("FASTER", "sum"),
            ONTIME=("ONTIME", "sum"),
            OVERSLA=("OVERSLA", "sum"),
            Jumlah_Paket=("Jumlah_Paket", "sum")
        ).reset_index()

        df_grouped = df_grouped.sort_values(["Kecamatan", "Tahun", "Bulan", "SLA"]).reset_index(drop=True)

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

        daftar_grup = df_grouped[["Kecamatan", "SLA"]].drop_duplicates().values.tolist()

        for kecamatan, sla_val in daftar_grup:
            print(f"\n{'='*60}")
            print(f"PROSES KECAMATAN: {kecamatan} - SLA: {sla_val}")
            print(f"{'='*60}")

            # Filter data untuk kecamatan dan SLA ini
            df_kec = df_grouped[(df_grouped["Kecamatan"] == kecamatan) & (df_grouped["SLA"] == sla_val)].copy()
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
                    prediksi_per_kategori[kategori] = {"pred": [{"nilai": 0}] * jumlah_bulan, "mape": 0.0, "rmse": 0.0}
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
                
                # Calculate MAPE and RMSE on training data
                mape_train = float(mean_absolute_percentage_error(y_train, y_pred_train) * 100)
                rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
                    
                print(f"  [Langkah 7] Total Loss = {total_loss:.4f}, MAPE = {mape_train:.2f}%, RMSE = {rmse_train:.2f}")

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

                prediksi_per_kategori[kategori] = {"pred": prediksi_list, "mape": mape_train, "rmse": rmse_train}

            # ----------------------------------------
            # Gabungkan hasil 3 kategori per bulan
            # ----------------------------------------
            for i in range(jumlah_bulan):
                f_dict = prediksi_per_kategori.get("FASTER", {"pred": [{"nilai": 0}] * jumlah_bulan, "mape": 0.0, "rmse": 0.0})
                o_dict = prediksi_per_kategori.get("ONTIME", {"pred": [{"nilai": 0}] * jumlah_bulan, "mape": 0.0, "rmse": 0.0})
                ov_dict = prediksi_per_kategori.get("OVERSLA", {"pred": [{"nilai": 0}] * jumlah_bulan, "mape": 0.0, "rmse": 0.0})

                faster_val = f_dict["pred"][i] if i < len(f_dict["pred"]) else {"nilai": 0}
                ontime_val = o_dict["pred"][i] if i < len(o_dict["pred"]) else {"nilai": 0}
                oversla_val = ov_dict["pred"][i] if i < len(ov_dict["pred"]) else {"nilai": 0}

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

                jumlah_prediksi = faster_n + ontime_n + oversla_n

                hasil_bulan = {
                    "tahun": thn,
                    "bulan": bln,
                    "sla": sla_val,
                    "faster": faster_n,
                    "ontime": ontime_n,
                    "oversla": oversla_n,
                    "jumlah_prediksi": jumlah_prediksi,
                    "mape_faster": round(f_dict["mape"], 2),
                    "mape_ontime": round(o_dict["mape"], 2),
                    "mape_oversla": round(ov_dict["mape"], 2),
                    "rmse_faster": round(f_dict["rmse"], 2),
                    "rmse_ontime": round(o_dict["rmse"], 2),
                    "rmse_oversla": round(ov_dict["rmse"], 2)
                }
                prediksi_kecamatan.append(hasil_bulan)

                # Simpan untuk database
                hasil_db.append((
                    nama_prediksi,
                    kecamatan,
                    thn,
                    bln,
                    sla_val,
                    jumlah_prediksi,
                    faster_n,
                    ontime_n,
                    oversla_n,
                    f_dict["mape"],
                    o_dict["mape"],
                    ov_dict["mape"],
                    f_dict["rmse"],
                    o_dict["rmse"],
                    ov_dict["rmse"]
                ))

            # Save per (kecamatan, sla) or aggregate
            if kecamatan not in semua_hasil:
                semua_hasil[kecamatan] = {
                    "histori_terakhir": f"{tahun_terakhir}-{bulan_terakhir:02d}",
                    "prediksi": [],
                    "detail_xgboost": []
                }
            semua_hasil[kecamatan]["prediksi"].extend(prediksi_kecamatan)
            semua_hasil[kecamatan]["detail_xgboost"].append({"sla": sla_val, "info": info_kecamatan})

            print(f"\n  HASIL {kecamatan} (SLA {sla_val}):")
            for h in prediksi_kecamatan:
                print(f"    {h['tahun']}-{h['bulan']:02d} => Faster: {h['faster']}, Ontime: {h['ontime']}, OverSLA: {h['oversla']}, Total: {h['jumlah_prediksi']}")

        # ========================================
        # LANGKAH 5: Simpan hasil ke database
        # ========================================
        if hasil_db:
            now = datetime.now()
            insert_query = """
                INSERT INTO hasil_prediksi_xgboost
                (nama_prediksi, kecamatan, tahun, bulan, sla, jumlah_prediksi, faster, ontime, oversla, mape_faster, mape_ontime, mape_oversla, rmse_faster, rmse_ontime, rmse_oversla, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
@app.route("/prediksi", methods=["POST"])
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

        # Tentukan tabel dataset berdasarkan endpoint yang dipanggil
        tabel_dataset = "dataset_tes" if request.path == "/prediksi" else "dataset_xgboost"

        semua_tahun = tahun_train + [tahun_test]
        placeholders = ", ".join(["%s"] * len(semua_tahun))
        query = (
            f"SELECT Kecamatan, Tahun, Bulan, SLA, FASTER, ONTIME, OVERSLA, Jumlah_Paket "
            f"FROM {tabel_dataset} WHERE nama_dataset = %s AND Tahun IN ({placeholders}) "
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

        if tabel_dataset == "dataset_xgboost":
            # Aggregate per Kecamatan + Tahun + Bulan
            df_grouped = df_all.groupby(["Kecamatan", "Tahun", "Bulan"]).agg(
                FASTER=("FASTER", "sum"),
                ONTIME=("ONTIME", "sum"),
                OVERSLA=("OVERSLA", "sum"),
                Jumlah_Paket=("Jumlah_Paket", "sum")
            ).reset_index()
            df_grouped = df_grouped.sort_values(["Kecamatan", "Tahun", "Bulan"]).reset_index(drop=True)
            kategori_target = ["FASTER", "ONTIME", "OVERSLA"]
        else:
            # dataset_tes: Pivot berdasarkan SLA
            df_pivot = df_all.pivot_table(
                index=["Kecamatan", "Tahun", "Bulan"], 
                columns="SLA", 
                values="Jumlah_Paket", 
                aggfunc="sum", 
                fill_value=0
            ).reset_index()
            df_pivot.columns = [str(c) if c in ["Kecamatan", "Tahun", "Bulan"] else f"SLA_{c}" for c in df_pivot.columns]
            df_grouped = df_pivot.sort_values(["Kecamatan", "Tahun", "Bulan"]).reset_index(drop=True)
            kategori_target = [c for c in df_grouped.columns if str(c).startswith("SLA_")]

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

                    sla_db_val = int(kategori.split("_")[1]) if str(kategori).startswith("SLA_") else aktual

                    # Simpan ke hasil_prediksi_tes
                    hasil_db_prediksi.append((
                        nama_prediksi, kecamatan, thn, bln,
                        prediksi,  # jumlah_prediksi
                        now.strftime('%Y-%m-%d'),  # tanggal_prediksi
                        sla_db_val,  # sla
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

            # Simpan data tes ke tabel dataset_tes jika evaluasi xgboost (opsional/tergantung original code)
            if tabel_dataset == "dataset_xgboost":
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
# Root route handled by web blueprint (redirects to /login)


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
            "reg_lambda": [0.01, 0.1, 0.5, 1, 3],
            "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
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
                [3,10,0.2,5,150,1,5.60,105.74],
            ],
            "Uji Nilai Alpha": [
                [0.01,0,0.2,5,150,1,5.62,122.77],[0.01,0.01,0.2,5,150,1,5.59,122.69],
                [0.01,0.05,0.2,5,150,1,5.64,121.84],[0.01,0.1,0.2,5,150,1,5.63,121.78],
                [0.01,0.5,0.2,5,150,1,5.55,121.99],[0.01,1,0.2,5,150,1,5.63,123.11],
                [0.01,2,0.2,5,150,1,5.74,123.73],[0.01,5,0.2,5,150,1,5.52,119.93],
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
                [0.01,10,0.2,5,150,5,5.93,123.33],[0.01,10,0.2,5,150,10,5.89,123.27],
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

        # Tabel 5.7: Ringkasan Evaluasi Model per Kategori (parameter terbaik)
        eval_semua_kategori = [
            {"Kategori": "On-time", "MAPE (%)": 3.93, "RMSE": 179.05},
            {"Kategori": "Faster",  "MAPE (%)": 4.91, "RMSE": 64.66},
            {"Kategori": "OverSLA", "MAPE (%)": 5.92, "RMSE": 49.88},
        ]

        output = os.path.join(os.path.dirname(__file__), "hasil_pengujian.xlsx")
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, rows in tabel.items():
                pd.DataFrame(rows, columns=KOLOM).to_excel(writer, sheet_name=name, index=False)
            pd.DataFrame(eval_kategori).to_excel(writer, sheet_name='Evaluasi per Kategori', index=False)
            pd.DataFrame(eval_semua_kategori).to_excel(writer, sheet_name='Evaluasi Semua Kategori', index=False)

        all_sheets = list(tabel.keys()) + ['Evaluasi per Kategori', 'Evaluasi Semua Kategori']
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
            "Lambda": {"values": [0.01, 0.1, 0.5, 1.0, 3.0],
                        "rmse": [123.08, 118.38, 109.73, 110.83, 105.74],
                        "mape": [5.91, 5.88, 5.37, 5.81, 5.60]},
            "Alpha": {"values": [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                       "rmse": [122.77, 122.69, 121.84, 121.78, 121.99, 123.11, 123.73, 119.93],
                       "mape": [5.62, 5.59, 5.64, 5.63, 5.55, 5.63, 5.74, 5.52]},
            "Learning Rate": {"values": [0.01, 0.05, 0.1, 0.15, 0.2],
                               "rmse": [274.25, 104.51, 107.39, 116.34, 123.08],
                               "mape": [15.71, 5.38, 5.29, 5.53, 5.91]},
            "Max Depth": {"values": [3, 5, 7, 9, 11],
                           "rmse": [123.84, 123.08, 127.63, 124.89, 125.10],
                           "mape": [6.24, 5.91, 5.75, 5.70, 5.72]},
            "N Estimators": {"values": [50, 80, 100, 150, 200],
                              "rmse": [123.86, 123.32, 123.11, 123.08, 123.05],
                              "mape": [6.02, 5.95, 5.92, 5.91, 5.91]},
            "Gamma": {"values": [0, 0.5, 1.0, 3.0, 5.0, 10.0],
                       "rmse": [123.03, 123.08, 123.08, 123.34, 123.33, 123.27],
                       "mape": [5.87, 5.90, 5.91, 5.91, 5.93, 5.89]},
        }

        colors = ['#1565C0', '#2E7D32', '#6A1B9A', '#E65100', '#4E342E', '#00695C']
        files = []

        # Style annotation helper
        def annotate_smart(ax, label, x, y, color, offset_y=14, fontsize=11, is_persen=False):
            """Annotate with opaque background for maximum readability"""
            txt = f'{label}%' if is_persen else f'{label}'
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color,
                              alpha=1.0, linewidth=1.5)
            ax.annotate(txt, (x, y), textcoords="offset points", xytext=(0, offset_y),
                        ha='center', fontsize=fontsize, fontweight='bold', color=color,
                        bbox=bbox_props, zorder=10)

        # Grafik individual - 2 subplots per parameter (MAPE di atas, RMSE di bawah)
        for i, (name, d) in enumerate(data.items()):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Grafik MAPE (atas)
            red_color = '#D32F2F'
            ax1.plot(d["values"], d["mape"], '-o', lw=2, ms=8, color=red_color, label='MAPE')
            ax1.set_xlabel(name, fontsize=14, fontweight='bold')
            ax1.set_ylabel('MAPE (%)', fontsize=14, fontweight='bold', color=red_color)
            ax1.tick_params(axis='y', labelcolor=red_color, labelsize=11)
            ax1.tick_params(axis='x', labelsize=11)
            if name == 'Lambda':
                ax1.set_xscale('log')
                ax1.minorticks_off()
                ax1.set_xticks([0.01, 0.1, 0.5, 1, 3])
                ax1.set_xticklabels(['0.01', '0.1', '0.5', '1', '3'])
            elif name == 'Alpha':
                ax1.set_xscale('symlog', linthresh=0.01)
                ax1.minorticks_off()
                ax1.set_xticks([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5])
                ax1.set_xticklabels(['0', '0.01', '0.05', '0.1', '0.5', '1', '2', '5'])
            elif name == 'Learning Rate':
                ax1.set_xticks([0.01, 0.05, 0.1, 0.15, 0.2])
                ax1.set_xticklabels(['0.01', '0.05', '0.1', '0.15', '0.2'])
            elif name == 'Max Depth':
                ax1.set_xticks([3, 5, 7, 9, 11])
                ax1.set_xticklabels(['3', '5', '7', '9', '11'])
                ax1.set_xlim(2.5, 11.5)
            elif name == 'N Estimators':
                ax1.set_xticks([50, 80, 100, 150, 200])
                ax1.set_xticklabels(['50', '80', '100', '150', '200'])
                ax1.set_xlim(40, 210)
            elif name == 'Gamma':
                ax1.set_xticks([0, 0.5, 1, 3, 5, 10])
                ax1.set_xticklabels(['0', '0.5', '1', '3', '5', '10'])
                ax1.set_xlim(-0.5, 10.5)
            ax1.set_title(f'Grafik MAPE - {name}', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Set y-axis limits with padding for annotations
            mape_min, mape_max = min(d["mape"]), max(d["mape"])
            mape_range = mape_max - mape_min
            ax1.set_ylim(mape_min - mape_range * 0.3, mape_max + mape_range * 0.3)
            
            for j, (x, y) in enumerate(zip(d["values"], d["mape"])):
                offset = 25 if j % 2 == 0 else -30
                annotate_smart(ax1, y, x, y, red_color, offset_y=offset, fontsize=10, is_persen=True)
            ax1.legend(loc='upper left', fontsize=12, framealpha=1.0, edgecolor='gray')
            
            # Grafik RMSE (bawah)
            blue_color = '#1565C0'
            ax2.plot(d["values"], d["rmse"], '-s', lw=2, ms=8, color=blue_color, label='RMSE')
            ax2.set_xlabel(name, fontsize=14, fontweight='bold')
            ax2.set_ylabel('RMSE', fontsize=14, fontweight='bold', color=blue_color)
            ax2.tick_params(axis='y', labelcolor=blue_color, labelsize=11)
            ax2.tick_params(axis='x', labelsize=11)
            if name == 'Lambda':
                ax2.set_xscale('log')
                ax2.minorticks_off()
                ax2.set_xticks([0.01, 0.1, 0.5, 1, 3])
                ax2.set_xticklabels(['0.01', '0.1', '0.5', '1', '3'])
            elif name == 'Alpha':
                ax2.set_xscale('symlog', linthresh=0.01)
                ax2.minorticks_off()
                ax2.set_xticks([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5])
                ax2.set_xticklabels(['0', '0.01', '0.05', '0.1', '0.5', '1', '2', '5'])
            elif name == 'Learning Rate':
                ax2.set_xticks([0.01, 0.05, 0.1, 0.15, 0.2])
                ax2.set_xticklabels(['0.01', '0.05', '0.1', '0.15', '0.2'])
            elif name == 'Max Depth':
                ax2.set_xticks([3, 5, 7, 9, 11])
                ax2.set_xticklabels(['3', '5', '7', '9', '11'])
                ax2.set_xlim(2.5, 11.5)
            elif name == 'N Estimators':
                ax2.set_xticks([50, 80, 100, 150, 200])
                ax2.set_xticklabels(['50', '80', '100', '150', '200'])
                ax2.set_xlim(40, 210)
            elif name == 'Gamma':
                ax2.set_xticks([0, 0.5, 1, 3, 5, 10])
                ax2.set_xticklabels(['0', '0.5', '1', '3', '5', '10'])
                ax2.set_xlim(-0.5, 10.5)
            ax2.set_title(f'Grafik RMSE - {name}', fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Set y-axis limits with padding for annotations
            rmse_min, rmse_max = min(d["rmse"]), max(d["rmse"])
            rmse_range = rmse_max - rmse_min
            ax2.set_ylim(rmse_min - rmse_range * 0.3, rmse_max + rmse_range * 0.3)
            
            for j, (x, y) in enumerate(zip(d["values"], d["rmse"])):
                offset = 25 if j % 2 == 0 else -30
                annotate_smart(ax2, y, x, y, blue_color, offset_y=offset, fontsize=10)
            ax2.legend(loc='upper left', fontsize=12, framealpha=1.0, edgecolor='gray')
            
            plt.tight_layout()
            fname = f"grafik_{name.lower().replace(' ', '_')}.png"
            fpath = os.path.join(base_dir, fname)
            plt.savefig(fpath, dpi=200, bbox_inches='tight')
            plt.close(fig)
            files.append(fname)

        # Grafik gabungan - 2 baris x 6 kolom (baris 1: MAPE, baris 2: RMSE)
        fig, axes = plt.subplots(2, 6, figsize=(30, 14))
        
        # Baris 1: MAPE untuk semua parameter
        for i, (name, d) in enumerate(data.items()):
            ax = axes[0, i]
            red_color = '#D32F2F'
            ax.plot(d["values"], d["mape"], '-o', lw=2, ms=7, color=red_color, label='MAPE')
            ax.set_xlabel(name, fontsize=13, fontweight='bold')
            ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold', color=red_color)
            ax.tick_params(axis='y', labelcolor=red_color, labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            if name == 'Lambda':
                ax.set_xscale('log')
                ax.minorticks_off()
                ax.set_xticks([0.01, 0.1, 0.5, 1, 3])
                ax.set_xticklabels(['0.01', '0.1', '0.5', '1', '3'])
            elif name == 'Alpha':
                ax.set_xscale('symlog', linthresh=0.01)
                ax.minorticks_off()
                ax.set_xticks([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5])
                ax.set_xticklabels(['0', '0.01', '0.05', '0.1', '0.5', '1', '2', '5'])
            elif name == 'Learning Rate':
                ax.set_xticks([0.01, 0.05, 0.1, 0.15, 0.2])
                ax.set_xticklabels(['0.01', '0.05', '0.1', '0.15', '0.2'])
            elif name == 'Max Depth':
                ax.set_xticks([3, 5, 7, 9, 11])
                ax.set_xticklabels(['3', '5', '7', '9', '11'])
                ax.set_xlim(2.5, 11.5)
            elif name == 'N Estimators':
                ax.set_xticks([50, 80, 100, 150, 200])
                ax.set_xticklabels(['50', '80', '100', '150', '200'])
                ax.set_xlim(40, 210)
            elif name == 'Gamma':
                ax.set_xticks([0, 0.5, 1, 3, 5, 10])
                ax.set_xticklabels(['0', '0.5', '1', '3', '5', '10'])
                ax.set_xlim(-0.5, 10.5)
            ax.set_title(f'Grafik MAPE - {name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits with padding for annotations
            mape_min, mape_max = min(d["mape"]), max(d["mape"])
            mape_range = mape_max - mape_min
            ax.set_ylim(mape_min - mape_range * 0.3, mape_max + mape_range * 0.3)
            
            for j, (x, y) in enumerate(zip(d["values"], d["mape"])):
                offset = 20 if j % 2 == 0 else -25
                annotate_smart(ax, y, x, y, red_color, offset_y=offset, fontsize=8, is_persen=True)
            ax.legend(loc='upper left', fontsize=10, framealpha=1.0, edgecolor='gray')
        
        # Baris 2: RMSE untuk semua parameter
        for i, (name, d) in enumerate(data.items()):
            ax = axes[1, i]
            blue_color = '#1565C0'
            ax.plot(d["values"], d["rmse"], '-s', lw=2, ms=7, color=blue_color, label='RMSE')
            ax.set_xlabel(name, fontsize=13, fontweight='bold')
            ax.set_ylabel('RMSE', fontsize=13, fontweight='bold', color=blue_color)
            ax.tick_params(axis='y', labelcolor=blue_color, labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            if name == 'Lambda':
                ax.set_xscale('log')
                ax.minorticks_off()
                ax.set_xticks([0.01, 0.1, 0.5, 1, 3])
                ax.set_xticklabels(['0.01', '0.1', '0.5', '1', '3'])
            elif name == 'Alpha':
                ax.set_xscale('symlog', linthresh=0.01)
                ax.minorticks_off()
                ax.set_xticks([0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5])
                ax.set_xticklabels(['0', '0.01', '0.05', '0.1', '0.5', '1', '2', '5'])
            elif name == 'Learning Rate':
                ax.set_xticks([0.01, 0.05, 0.1, 0.15, 0.2])
                ax.set_xticklabels(['0.01', '0.05', '0.1', '0.15', '0.2'])
            elif name == 'Max Depth':
                ax.set_xticks([3, 5, 7, 9, 11])
                ax.set_xticklabels(['3', '5', '7', '9', '11'])
                ax.set_xlim(2.5, 11.5)
            elif name == 'N Estimators':
                ax.set_xticks([50, 80, 100, 150, 200])
                ax.set_xticklabels(['50', '80', '100', '150', '200'])
                ax.set_xlim(40, 210)
            elif name == 'Gamma':
                ax.set_xticks([0, 0.5, 1, 3, 5, 10])
                ax.set_xticklabels(['0', '0.5', '1', '3', '5', '10'])
                ax.set_xlim(-0.5, 10.5)
            ax.set_title(f'Grafik RMSE - {name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits with padding for annotations
            rmse_min, rmse_max = min(d["rmse"]), max(d["rmse"])
            rmse_range = rmse_max - rmse_min
            ax.set_ylim(rmse_min - rmse_range * 0.3, rmse_max + rmse_range * 0.3)
            
            for j, (x, y) in enumerate(zip(d["values"], d["rmse"])):
                offset = 20 if j % 2 == 0 else -25
                annotate_smart(ax, y, x, y, blue_color, offset_y=offset, fontsize=8)
            ax.legend(loc='upper left', fontsize=10, framealpha=1.0, edgecolor='gray')
        
        plt.suptitle('Pengujian 6 Hyperparameter XGBoost', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        gab_path = os.path.join(base_dir, "grafik_6_hyperparameter.png")
        plt.savefig(gab_path, dpi=200, bbox_inches='tight')
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

        kategori_list = [("ONTIME", "On-time"), ("FASTER", "Faster"), ("OVERSLA", "OverSLA")]
        base_dir = os.path.dirname(__file__)
        files = []

        n_kec = len(daftar_kec)
        n_kat = len(kategori_list)

        # === GABUNGAN: 1 gambar 5 baris x 3 kolom ===
        fig, axes = plt.subplots(n_kec, n_kat, figsize=(7 * n_kat, 4 * n_kec))
        if n_kec == 1:
            axes = [axes]

        for col, (kat_key, kat_label) in enumerate(kategori_list):
            for row, kec in enumerate(daftar_kec):
                ax = axes[row][col] if n_kec > 1 else axes[col]
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
                    ax.set_visible(False)
                    continue

                model = xgb.XGBRegressor(**BASELINE)
                model.fit(Xtr, ytr, verbose=False)
                yp = np.maximum(model.predict(Xte), 0)

                bulan_label = [f"{int(dl.iloc[n+i]['Bulan'])}/{int(dl.iloc[n+i]['Tahun'])}" for i in range(len(yte))]

                ax.plot(bulan_label, yte, 'b-o', linewidth=1.8, markersize=5, label='Aktual')
                ax.plot(bulan_label, yp, 'r--s', linewidth=1.8, markersize=5, label='Prediksi')

                rmse = float(np.sqrt(mean_squared_error(yte, yp)))

                ax.set_title(f'{kec} - {kat_label} (RMSE: {rmse:.2f})', fontsize=10, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=7)
                ax.tick_params(axis='y', labelsize=7)
                ax.grid(True, alpha=0.3)

                if row == 0:
                    ax.legend(fontsize=7, framealpha=1.0, edgecolor='gray', loc='upper right')

        plt.suptitle('Trend Aktual vs Prediksi - Semua Kategori', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0, w_pad=2.0)

        fname = "grafik_kategori_gabungan_semua.png"
        fpath = os.path.join(base_dir, fname)
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        files.append(fname)
        print(f"  Saved: {fname}")

        # === INDIVIDUAL: tetap buat per kategori juga ===
        for kat_key, kat_label in kategori_list:
            fig, axes_ind = plt.subplots(n_kec, 1, figsize=(14, 5 * n_kec))
            if n_kec == 1:
                axes_ind = [axes_ind]

            for idx, kec in enumerate(daftar_kec):
                ax = axes_ind[idx]
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

                bulan_label = [f"{int(dl.iloc[n+i]['Bulan'])}/{int(dl.iloc[n+i]['Tahun'])}" for i in range(len(yte))]

                ax.plot(bulan_label, yte, 'b-o', linewidth=2.5, markersize=9, label='Aktual')
                ax.plot(bulan_label, yp, 'r--s', linewidth=2.5, markersize=9, label='Prediksi')

                rmse = float(np.sqrt(mean_squared_error(yte, yp)))
                mask = yte != 0
                mape = float(np.mean(np.abs((yte[mask]-yp[mask])/yte[mask]))*100) if np.any(mask) else 0

                ax.set_title(f'{kec} - {kat_label} (RMSE: {rmse:.2f})',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel('Bulan', fontsize=12, fontweight='bold')
                ax.set_ylabel('Jumlah Paket', fontsize=12, fontweight='bold')
                ax.legend(fontsize=11, framealpha=1.0, edgecolor='gray')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                ax.tick_params(axis='y', labelsize=10)

            plt.suptitle(f'Trend Aktual vs Prediksi - {kat_label}',
                         fontsize=18, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            fname = f"grafik_kategori_{kat_key.lower()}.png"
            fpath = os.path.join(base_dir, fname)
            plt.savefig(fpath, dpi=200, bbox_inches='tight')
            plt.close(fig)
            files.append(fname)
            print(f"  Saved: {fname}")

        return jsonify({"status": "success", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ========================================
# ENDPOINT: Generate Evaluation Charts - RMSE & MAPE per Category
# ========================================
@app.route('/grafik-evaluasi-kategori', methods=['POST'])
def grafik_evaluasi_kategori():
    """
    Generate ONE comprehensive evaluation chart showing RMSE & MAPE per category
    Similar to hyperparameter evaluation chart style.
    
    Layout: Single figure with dual Y-axis
    - X-axis: Districts (Kecamatan)
    - Left Y-axis: RMSE (solid lines)
    - Right Y-axis: MAPE % (dashed lines)
    - 3 series for On-time, Faster, OverSLA
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        data = request.json
        nama_dataset = data.get("nama_dataset", "dataset_training")
        rasio_train = data.get("rasio_train", 0.8)
        
        conn = pymysql.connect(**db_config, cursorclass=DictCursor)
        cursor = conn.cursor()
        
        # Get data from database
        query = "SELECT * FROM dataset_xgboost WHERE nama_dataset = %s"
        cursor.execute(query, (nama_dataset,))
        rows = cursor.fetchall()
        
        if not rows:
            cursor.close()
            conn.close()
            return jsonify({
                "success": False,
                "message": f"Dataset '{nama_dataset}' tidak ditemukan"
            }), 404
        
        df_all = pd.DataFrame(rows)
        
        # Aggregate data per district per month
        df_grouped = df_all.groupby(["Kecamatan", "Tahun", "Bulan"]).agg({
            "FASTER": "sum",
            "ONTIME": "sum",
            "OVERSLA": "sum",
            "Jumlah_Paket": "sum"
        }).reset_index()
        
        df_grouped = df_grouped.sort_values(["Kecamatan", "Tahun", "Bulan"]).reset_index(drop=True)
        
        # Get all districts
        daftar_kecamatan = sorted(df_grouped["Kecamatan"].unique())
        
        # Define categories with colors
        kategori_list = [
            ("ONTIME", "On-time", "#1565C0"),    # Blue
            ("FASTER", "Faster", "#2E7D32"),     # Green
            ("OVERSLA", "OverSLA", "#D32F2F")    # Red
        ]
        
        # XGBoost parameters
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
            "verbosity": 0
        }
        
        # Feature columns
        FITUR_KOLOM = ["Bulan", "Sin_Bulan", "Cos_Bulan", "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3"]
        
        base_dir = os.path.dirname(__file__)
        
        # Storage for metrics per category per district
        metrics_data = {
            "On-time": {"rmse": [], "mape": []},
            "Faster": {"rmse": [], "mape": []},
            "OverSLA": {"rmse": [], "mape": []}
        }
        
        # Calculate metrics for each district and category
        for kecamatan in daftar_kecamatan:
            # Get data for this district
            df_kec = df_grouped[df_grouped["Kecamatan"] == kecamatan].copy()
            df_kec = df_kec.sort_values(["Tahun", "Bulan"]).reset_index(drop=True)
            
            # Add sin/cos features
            df_kec["Sin_Bulan"] = np.sin(2 * np.pi * df_kec["Bulan"] / 12)
            df_kec["Cos_Bulan"] = np.cos(2 * np.pi * df_kec["Bulan"] / 12)
            
            print(f"\nProcessing {kecamatan}:")
            
            # Process each category
            for kat_key, kat_label, color in kategori_list:
                # Create lag features
                df_with_lag = df_kec.copy()
                df_with_lag["Lag_1"] = df_with_lag[kat_key].shift(1)
                df_with_lag["Lag_2"] = df_with_lag[kat_key].shift(2)
                df_with_lag["Lag_3"] = df_with_lag[kat_key].shift(3)
                df_with_lag["Rolling_Mean_3"] = df_with_lag[kat_key].rolling(window=3, min_periods=1).mean().shift(1)
                df_with_lag = df_with_lag.dropna()
                
                if len(df_with_lag) < 3:
                    metrics_data[kat_label]["rmse"].append(0)
                    metrics_data[kat_label]["mape"].append(0)
                    continue
                
                # Split train/test
                n_total = len(df_with_lag)
                n_train = int(n_total * rasio_train)
                
                X_train = df_with_lag.iloc[:n_train][FITUR_KOLOM].values
                y_train = df_with_lag.iloc[:n_train][kat_key].values
                X_test = df_with_lag.iloc[n_train:][FITUR_KOLOM].values
                y_test = df_with_lag.iloc[n_train:][kat_key].values
                
                if len(X_train) < 3 or len(X_test) == 0:
                    metrics_data[kat_label]["rmse"].append(0)
                    metrics_data[kat_label]["mape"].append(0)
                    continue
                
                # Train XGBoost model
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_train, y_train, verbose=False)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred = np.maximum(y_pred, 0)
                
                # Calculate RMSE
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                
                # Calculate MAPE
                mask = y_test != 0
                if np.any(mask):
                    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
                else:
                    mape = 0.0
                
                metrics_data[kat_label]["rmse"].append(rmse)
                metrics_data[kat_label]["mape"].append(mape)
                
                print(f"  {kat_label}: RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x_pos = np.arange(len(daftar_kecamatan))
        width = 0.25
        
        # Plot RMSE on left axis (solid lines)
        for idx, (kat_key, kat_label, color) in enumerate(kategori_list):
            rmse_values = metrics_data[kat_label]["rmse"]
            line = ax.plot(x_pos, rmse_values, marker='o', linestyle='-', linewidth=2.5, 
                          markersize=9, color=color, label=f'RMSE - {kat_label}', zorder=3)
            
            # Add annotations for RMSE
            for i, val in enumerate(rmse_values):
                if val > 0:
                    ax.annotate(f'{val:.2f}', xy=(i, val), xytext=(0, 10),
                               textcoords='offset points', ha='center', fontsize=9,
                               fontweight='bold', color=color,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor=color, alpha=0.9))
        
        ax.set_xlabel('Kecamatan', fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=13, fontweight='bold', color='#1565C0')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(daftar_kecamatan, fontsize=11, fontweight='bold', rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='#1565C0', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_ylim(bottom=0)
        
        # Create second Y-axis for MAPE
        ax2 = ax.twinx()
        
        # Plot MAPE on right axis (dashed lines)
        for idx, (kat_key, kat_label, color) in enumerate(kategori_list):
            mape_values = metrics_data[kat_label]["mape"]
            line = ax2.plot(x_pos, mape_values, marker='s', linestyle='--', linewidth=2.5, 
                           markersize=8, color=color, label=f'MAPE - {kat_label}', zorder=2)
            
            # Add annotations for MAPE
            for i, val in enumerate(mape_values):
                if val > 0:
                    ax2.annotate(f'{val:.2f}%', xy=(i, val), xytext=(0, -15),
                                textcoords='offset points', ha='center', fontsize=9,
                                fontweight='bold', color=color,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                         edgecolor=color, alpha=0.9))
        
        ax2.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold', color='#D32F2F')
        ax2.tick_params(axis='y', labelcolor='#D32F2F', labelsize=10)
        ax2.set_ylim(bottom=0)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11,
                 framealpha=0.95, edgecolor='gray', ncol=2)
        
        plt.title('Evaluasi Model per Kategori - RMSE & MAPE', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save figure
        filename = "grafik_evaluasi_semua_kategori.png"
        filepath = os.path.join(base_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Saved: {filename}")
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "Successfully generated evaluation chart",
            "file": filename,
            "total_districts": len(daftar_kecamatan),
            "metrics": metrics_data
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
