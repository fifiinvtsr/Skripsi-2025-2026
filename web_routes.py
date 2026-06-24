"""Web routes for the Flask application - replaces Laravel frontend."""
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from pymysql.cursors import DictCursor
import pymysql
import bcrypt
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from web_helpers import login_required, admin_required, BULAN_NAMES

web = Blueprint('web', __name__)

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "prediksi_svr",
}

def get_conn():
    return pymysql.connect(**db_config, cursorclass=DictCursor)

# ===================== AUTH =====================
@web.route('/login', methods=['GET', 'POST'])
def web_login():
    if 'user_id' in session:
        return redirect(url_for('web.admin_dashboard') if session.get('role') == 'admin' else url_for('web.manager_dashboard'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cur.fetchone()
            if user:
                stored_hash = user['password'].replace('$2y$', '$2b$', 1)
                pw_ok = bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
                if pw_ok:
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    session['name'] = user.get('name', user['username'])
                    session['role'] = user['role']
                    if user['role'] == 'admin':
                        return redirect(url_for('web.admin_dashboard'))
                    else:
                        return redirect(url_for('web.manager_dashboard'))
                else:
                    error = 'Username atau password salah.'
            else:
                error = 'Username atau password salah.'
        finally:
            conn.close()
    return render_template('login.html', error=error)

@web.route('/logout', methods=['POST', 'GET'])
def web_logout():
    session.clear()
    return redirect(url_for('web.web_login'))

@web.route('/')
def index():
    return redirect(url_for('web.web_login'))

# ===================== DASHBOARD =====================
@web.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT nama_prediksi FROM hasil_prediksi_tes ORDER BY nama_prediksi")
            tes_list = [r['nama_prediksi'] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT nama_prediksi FROM hasil_prediksi_xgboost ORDER BY nama_prediksi")
            xgb_list = [r['nama_prediksi'] for r in cur.fetchall()]
        sel_tes = request.args.get('prediksi_tes', tes_list[-1] if tes_list else '')
        sel_xgb = request.args.get('prediksi_xgb', xgb_list[-1] if xgb_list else '')
        f_tahun = request.args.get('tahun', '')
        f_bulan = request.args.get('bulan', '')
        f_kec = request.args.get('kecamatan', '')
        f_sla = request.args.get('sla', '')
        tahun_list, bulan_list, kecamatan_list, sla_list = [], [], [], []
        tabel_detail, chart_payload, kpi_stats = [], {}, None
        # Hasil Pengujian Hyperparameter (data statis dari pengujian) - selalu ditampilkan
        pengujian_payload = {
            'hyperparameter': {
                'Lambda': {
                    'values': [0.01, 0.1, 0.5, 1.0, 3.0],
                    'mape': [5.91, 5.88, 5.37, 5.81, 5.60],
                    'rmse': [123.08, 118.38, 109.73, 110.83, 105.74],
                    'best_idx': 2, 'best_val': 0.5,
                },
                'Alpha': {
                    'values': [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                    'mape': [5.62, 5.59, 5.64, 5.63, 5.55, 5.63, 5.74, 5.52],
                    'rmse': [122.77, 122.69, 121.84, 121.78, 121.99, 123.11, 123.73, 119.93],
                    'best_idx': 7, 'best_val': 5.0,
                },
                'Learning Rate': {
                    'values': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'mape': [15.71, 5.38, 5.29, 5.53, 5.91],
                    'rmse': [274.25, 104.51, 107.39, 116.34, 123.08],
                    'best_idx': 2, 'best_val': 0.1,
                },
                'Max Depth': {
                    'values': [3, 5, 7, 9, 11],
                    'mape': [6.24, 5.91, 5.75, 5.70, 5.72],
                    'rmse': [123.84, 123.08, 127.63, 124.89, 125.10],
                    'best_idx': 3, 'best_val': 9,
                },
                'N Estimators': {
                    'values': [50, 80, 100, 150, 200],
                    'mape': [6.02, 5.95, 5.92, 5.91, 5.91],
                    'rmse': [123.86, 123.32, 123.11, 123.08, 123.05],
                    'best_idx': 4, 'best_val': 200,
                },
                'Gamma': {
                    'values': [0, 0.5, 1.0, 3.0, 5.0, 10.0],
                    'mape': [5.87, 5.90, 5.91, 5.91, 5.93, 5.89],
                    'rmse': [123.03, 123.08, 123.08, 123.34, 123.33, 123.27],
                    'best_idx': 0, 'best_val': 0,
                },
            },
            'eval_kategori': [
                {'kategori': 'On-time', 'mape': 3.93, 'rmse': 179.05},
                {'kategori': 'Faster', 'mape': 4.91, 'rmse': 64.66},
                {'kategori': 'OverSLA', 'mape': 5.92, 'rmse': 49.88},
            ],
            'eval_detail': [
                {'parameter': 'reg_lambda', 'kategori': 'On-time', 'mape': 3.93, 'rmse': 179.05},
                {'parameter': 'reg_alpha', 'kategori': 'On-time', 'mape': 4.28, 'rmse': 185.62},
                {'parameter': 'learning_rate', 'kategori': 'On-time', 'mape': 4.15, 'rmse': 181.33},
                {'parameter': 'max_depth', 'kategori': 'On-time', 'mape': 5.31, 'rmse': 195.40},
                {'parameter': 'n_estimators', 'kategori': 'On-time', 'mape': 5.25, 'rmse': 193.88},
                {'parameter': 'gamma', 'kategori': 'On-time', 'mape': 5.28, 'rmse': 194.50},
                {'parameter': 'reg_lambda', 'kategori': 'Faster', 'mape': 4.91, 'rmse': 64.66},
                {'parameter': 'reg_alpha', 'kategori': 'Faster', 'mape': 5.35, 'rmse': 69.80},
                {'parameter': 'learning_rate', 'kategori': 'Faster', 'mape': 5.18, 'rmse': 67.25},
                {'parameter': 'max_depth', 'kategori': 'Faster', 'mape': 5.86, 'rmse': 73.42},
                {'parameter': 'n_estimators', 'kategori': 'Faster', 'mape': 5.78, 'rmse': 72.15},
                {'parameter': 'gamma', 'kategori': 'Faster', 'mape': 5.82, 'rmse': 72.90},
                {'parameter': 'reg_lambda', 'kategori': 'OverSLA', 'mape': 6.15, 'rmse': 52.30},
                {'parameter': 'reg_alpha', 'kategori': 'OverSLA', 'mape': 5.92, 'rmse': 49.88},
                {'parameter': 'learning_rate', 'kategori': 'OverSLA', 'mape': 6.05, 'rmse': 51.45},
                {'parameter': 'max_depth', 'kategori': 'OverSLA', 'mape': 6.58, 'rmse': 55.18},
                {'parameter': 'n_estimators', 'kategori': 'OverSLA', 'mape': 6.50, 'rmse': 54.60},
                {'parameter': 'gamma', 'kategori': 'OverSLA', 'mape': 6.45, 'rmse': 54.10},
            ],
            'eval_kecamatan': {
                'labels': ['BLIMBING', 'KEDUNGKANDANG', 'KLOJEN', 'LOWOKWARU', 'SUKUN'],
                'ontime':  {'mape': [2.72, 7.06, 6.80, 4.04, 4.50], 'rmse': [115.0, 156.6, 497.1, 214.4, 185.8]},
                'faster':  {'mape': [3.53, 6.55, 11.36, 2.50, 5.74], 'rmse': [39.8, 69.4, 173.3, 44.9, 46.2]},
                'oversla': {'mape': [7.83, 9.74, 9.51, 8.29, 12.05], 'rmse': [50.3, 42.5, 143.3, 62.5, 191.9]},
            },
        }
        if sel_tes and sel_xgb:
            with conn.cursor() as cur:
                tahun_list = [2021, 2022, 2023, 2024]
                bulan_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                cur.execute("SELECT DISTINCT kecamatan FROM hasil_prediksi_xgboost WHERE nama_prediksi=%s ORDER BY kecamatan", (sel_xgb,))
                kecamatan_list = [r['kecamatan'] for r in cur.fetchall()]
                sla_list = [2, 3, 4, 5, 6]
                
                q = """
                SELECT xgb.tahun, xgb.bulan, xgb.kecamatan, 
                       (xgb.faster + xgb.ontime + xgb.oversla) as jumlah, 
                       xgb.ontime, xgb.oversla, xgb.faster,
                       tes.jumlah_prediksi as prediksi_tes,
                       tes.sla as aktual_tes
                FROM hasil_prediksi_xgboost xgb 
                LEFT JOIN hasil_prediksi_tes tes 
                  ON xgb.tahun = tes.tahun AND xgb.bulan = tes.bulan 
                  AND xgb.kecamatan COLLATE utf8mb4_unicode_ci = tes.kecamatan COLLATE utf8mb4_unicode_ci 
                  AND tes.nama_prediksi COLLATE utf8mb4_unicode_ci = %s
                WHERE xgb.nama_prediksi COLLATE utf8mb4_unicode_ci = %s
                """
                params = [sel_tes, sel_xgb]
                if f_tahun: q += " AND xgb.tahun=%s"; params.append(f_tahun)
                if f_bulan: q += " AND xgb.bulan=%s"; params.append(f_bulan)
                if f_kec: q += " AND xgb.kecamatan=%s"; params.append(f_kec)
                cur.execute(q, params)
                tabel_detail = []
                for r in cur.fetchall():
                    r['sla'] = f_sla if f_sla else "2, 3, 4, 5, 6"
                    tabel_detail.append(r)
                # Charts - stacked bar
                q2 = "SELECT kecamatan, SUM(oversla) as oversla, SUM(ontime) as ontime, SUM(faster) as faster FROM hasil_prediksi_xgboost WHERE nama_prediksi=%s"
                p2 = [sel_xgb]
                if f_tahun: q2 += " AND tahun=%s"; p2.append(f_tahun)
                if f_bulan: q2 += " AND bulan=%s"; p2.append(f_bulan)
                if f_kec: q2 += " AND kecamatan=%s"; p2.append(f_kec)
                q2 += " GROUP BY kecamatan"
                cur.execute(q2, p2)
                chart_rows = cur.fetchall()
                labels = [r['kecamatan'] for r in chart_rows]
                stacked = {'labels': labels, 'datasets': [
                    {'label': 'oversla', 'backgroundColor': '#EF4444', 'data': [float(r['oversla'] or 0) for r in chart_rows]},
                    {'label': 'ontime', 'backgroundColor': '#10B981', 'data': [float(r['ontime'] or 0) for r in chart_rows]},
                    {'label': 'faster', 'backgroundColor': '#3B82F6', 'data': [float(r['faster'] or 0) for r in chart_rows]},
                ]}
                # Grouped bar
                q3 = "SELECT kecamatan, SUM(faster + ontime + oversla) as total, SUM(oversla) as oversla, SUM(ontime) as ontime, SUM(faster) as faster FROM hasil_prediksi_xgboost WHERE nama_prediksi=%s"
                p3 = [sel_xgb]
                if f_tahun: q3 += " AND tahun=%s"; p3.append(f_tahun)
                if f_bulan: q3 += " AND bulan=%s"; p3.append(f_bulan)
                if f_kec: q3 += " AND kecamatan=%s"; p3.append(f_kec)
                q3 += " GROUP BY kecamatan"
                cur.execute(q3, p3)
                g_rows = cur.fetchall()
                g_labels = [r['kecamatan'] for r in g_rows]
                grouped = {'labels': g_labels, 'datasets': [
                    {'label': 'Jumlah Paket', 'backgroundColor': '#FACC15', 'data': [int(r['total'] or 0) for r in g_rows], 'yAxisID': 'y'},
                    {'label': 'Oversla (%)', 'backgroundColor': '#EF4444', 'data': [round(float(r['oversla'] or 0)/max(float(r['total'] or 1),1)*100,2) for r in g_rows], 'yAxisID': 'y1'},
                    {'label': 'Ontime (%)', 'backgroundColor': '#10B981', 'data': [round(float(r['ontime'] or 0)/max(float(r['total'] or 1),1)*100,2) for r in g_rows], 'yAxisID': 'y1'},
                    {'label': 'Faster (%)', 'backgroundColor': '#3B82F6', 'data': [round(float(r['faster'] or 0)/max(float(r['total'] or 1),1)*100,2) for r in g_rows], 'yAxisID': 'y1'},
                ]}
                chart_payload = {'stackedBar': stacked, 'groupedBar': grouped}
                # KPI
                total_pred = sum(float(r.get('jumlah', 0) or 0) for r in tabel_detail)
                total_ontime = sum(float(r.get('ontime', 0) or 0) for r in tabel_detail)
                ontime_rate = (total_ontime / total_pred * 100) if total_pred > 0 else 0
                kec_stats = {}
                for r in tabel_detail:
                    k = r['kecamatan']
                    if k not in kec_stats: kec_stats[k] = {'total': 0, 'ontime': 0, 'oversla': 0}
                    kec_stats[k]['total'] += float(r.get('jumlah', 0) or 0)
                    kec_stats[k]['ontime'] += float(r.get('ontime', 0) or 0)
                    kec_stats[k]['oversla'] += float(r.get('oversla', 0) or 0)
                best = max(kec_stats.items(), key=lambda x: x[1]['ontime']/max(x[1]['total'],1), default=('-', {}))
                worst = max(kec_stats.items(), key=lambda x: x[1]['oversla']/max(x[1]['total'],1), default=('-', {}))
                kpi_stats = {
                    'total_predictions': "{:,}".format(int(total_pred)),
                    'overall_ontime': "{:.2f}%".format(ontime_rate),
                    'best_kecamatan': best[0] if best else '-',
                    'worst_kecamatan': worst[0] if worst else '-',
                }
    finally:
        conn.close()
    return render_template('dashboard_admin.html', active_page='admin.dashboard',
        nama_prediksi_tes_list=tes_list, selected_prediksi_tes=sel_tes,
        nama_prediksi_xgb_list=xgb_list, selected_prediksi_xgb=sel_xgb,
        tahun_list=tahun_list, bulan_list=bulan_list, kecamatan_list=kecamatan_list, sla_list=sla_list,
        filter_tahun=f_tahun, filter_bulan=f_bulan, filter_kecamatan=f_kec, filter_sla=f_sla,
        tabel_detail=tabel_detail, chart_payload=chart_payload, kpi_stats=kpi_stats,
        pengujian_payload=pengujian_payload, bulan_names=BULAN_NAMES)

@web.route('/manager/dashboard')
@login_required
def manager_dashboard():
    return render_template('dashboard_manager.html', active_page='manager.dashboard')

# ===================== UPLOAD DATA AWAL =====================
@web.route('/upload-data-awal', methods=['GET', 'POST'])
@admin_required
def web_upload_data_awal():
    conn = get_conn()
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename:
            import requests as req
            try:
                resp = req.post('http://127.0.0.1:5000/upload-data-awal', files={'file': (file.filename, file.stream, file.content_type)})
                data = resp.json()
                flash(data.get('message', 'Berhasil!') if data.get('status') == 'success' else data.get('message', 'Gagal'), 'success' if data.get('status') == 'success' else 'error')
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('web.web_upload_data_awal'))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT YEAR(tgl_antaran_pertama) as y FROM data_awal ORDER BY y")
            available_years = [r['y'] for r in cur.fetchall() if r['y']]
            cur.execute("SELECT MAX(tgl_antaran_pertama) as d FROM data_awal")
            latest = cur.fetchone()
            latest_date = str(latest['d']) if latest and latest['d'] else '-'
            cur.execute("SELECT COUNT(*) as c FROM data_awal")
            total_records = cur.fetchone()['c']
            page = int(request.args.get('page', 1))
            per_page = 10
            offset = (page - 1) * per_page
            cur.execute("SELECT * FROM data_awal ORDER BY tgl_antaran_pertama DESC LIMIT %s OFFSET %s", (per_page, offset))
            data_terbaru = cur.fetchall()
            total_pages = (min(total_records, 100) + per_page - 1) // per_page
    finally:
        conn.close()
    pagination = {'page': page, 'pages': total_pages} if total_pages > 1 else None
    return render_template('upload_data_awal.html', active_page='upload.data.awal',
        available_years=available_years, latest_date=latest_date, total_records=total_records,
        missing_count=0, data_terbaru=data_terbaru, pagination=pagination)

@web.route('/delete-data-awal', methods=['POST'])
@admin_required
def web_delete_data_awal():
    opt = request.form.get('delete_option', '')
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if opt == 'all':
                cur.execute("DELETE FROM data_awal")
            elif opt == 'date':
                cur.execute("DELETE FROM data_awal WHERE DATE(tgl_antaran_pertama) = %s", (request.form.get('filter_date'),))
            elif opt == 'monthYear':
                cur.execute("DELETE FROM data_awal WHERE MONTH(tgl_antaran_pertama)=%s AND YEAR(tgl_antaran_pertama)=%s",
                    (request.form.get('filter_month'), request.form.get('filter_year')))
            elif opt == 'year':
                cur.execute("DELETE FROM data_awal WHERE YEAR(tgl_antaran_pertama)=%s", (request.form.get('filter_year_only'),))
            conn.commit()
        flash('Data berhasil dihapus.', 'success')
    except Exception as e:
        flash(f'Gagal: {str(e)}', 'error')
    finally:
        conn.close()
    return redirect(url_for('web.web_upload_data_awal'))

# ===================== UPLOAD DATASET TES =====================
@web.route('/upload-dataset-TES', methods=['GET', 'POST'])
@admin_required
def web_upload_dataset_tes():
    conn = get_conn()
    if request.method == 'POST':
        nama = request.form.get('nama_dataset', '')
        tahun = request.form.getlist('tahun')
        if nama and tahun:
            import requests as req
            try:
                resp = req.post('http://127.0.0.1:5000/dataset-TES', json={'nama_dataset': nama, 'tahun': [int(t) for t in tahun]})
                data = resp.json()
                flash(data.get('message', 'Berhasil!'), 'success' if data.get('status') == 'success' else 'error')
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('web.web_upload_dataset_tes'))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT YEAR(tgl_antaran_pertama) as y FROM data_awal WHERE tgl_antaran_pertama IS NOT NULL ORDER BY y")
            years = [r['y'] for r in cur.fetchall() if r['y']]
            cur.execute("""
                SELECT * FROM (
                    SELECT nama_dataset, Kecamatan, Tahun, Bulan, SLA, Jumlah_Paket, created_at,
                           ROW_NUMBER() OVER(PARTITION BY Kecamatan ORDER BY Tahun DESC, Bulan DESC, CAST(SLA AS UNSIGNED) ASC) as rn
                    FROM dataset_tes
                ) t WHERE rn <= 10
            """)
            rows = cur.fetchall()
            data_terbaru_grouped = {}
            for r in rows:
                k = r['Kecamatan']
                if k not in data_terbaru_grouped:
                    data_terbaru_grouped[k] = []
                data_terbaru_grouped[k].append(r)
            cur.execute("SELECT DISTINCT nama_dataset FROM dataset_tes")
            datasets_list = [r['nama_dataset'] for r in cur.fetchall()]
    finally:
        conn.close()
    return render_template('upload_dataset.html', active_page='upload.dataset.tes',
        page_title='Dataset Paket (TES)', page_subtitle='Buat dataset untuk prediksi volume paket',
        form_action=url_for('web.web_upload_dataset_tes'), available_years=years, 
        data_terbaru_grouped=data_terbaru_grouped, datasets_list=datasets_list, table_name='dataset_tes')

# ===================== UPLOAD DATASET MLP =====================
@web.route('/upload-dataset-MLP', methods=['GET', 'POST'])
@admin_required
def web_upload_dataset_mlp():
    conn = get_conn()
    if request.method == 'POST':
        nama = request.form.get('nama_dataset', '')
        tahun = request.form.getlist('tahun')
        if nama and tahun:
            import requests as req
            try:
                resp = req.post('http://127.0.0.1:5000/dataset-xgboost', json={'nama_dataset': nama, 'tahun': [int(t) for t in tahun]})
                data = resp.json()
                flash(data.get('message', 'Berhasil!'), 'success' if data.get('status') == 'success' else 'error')
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('web.web_upload_dataset_mlp'))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT YEAR(tgl_antaran_pertama) as y FROM data_awal WHERE tgl_antaran_pertama IS NOT NULL ORDER BY y")
            years = [r['y'] for r in cur.fetchall() if r['y']]
            cur.execute("""
                SELECT * FROM (
                    SELECT nama_dataset, Kecamatan, Tahun, Bulan, SLA, FASTER, ONTIME, OVERSLA, Jumlah_Paket, created_at,
                           ROW_NUMBER() OVER(PARTITION BY Kecamatan ORDER BY Tahun DESC, Bulan DESC, CAST(SLA AS UNSIGNED) ASC) as rn
                    FROM dataset_xgboost
                ) t WHERE rn <= 10
            """)
            rows = cur.fetchall()
            data_terbaru_grouped = {}
            for r in rows:
                k = r['Kecamatan']
                if k not in data_terbaru_grouped:
                    data_terbaru_grouped[k] = []
                data_terbaru_grouped[k].append(r)
            cur.execute("SELECT DISTINCT nama_dataset FROM dataset_xgboost")
            datasets_list = [r['nama_dataset'] for r in cur.fetchall()]
    finally:
        conn.close()
    return render_template('upload_dataset.html', active_page='upload.dataset.mlp',
        page_title='Dataset XGBoost regression', page_subtitle='Buat dataset untuk prediksi dengan XGBoost regression',
        form_action=url_for('web.web_upload_dataset_mlp'), available_years=years, 
        data_terbaru_grouped=data_terbaru_grouped, datasets_list=datasets_list, table_name='dataset_xgboost')

@web.route('/delete-dataset', methods=['POST'])
@admin_required
def web_delete_dataset():
    table_name = request.form.get('table_name')
    nama_dataset = request.form.get('nama_dataset')
    
    if table_name not in ['dataset_tes', 'dataset_xgboost']:
        flash('Invalid table', 'error')
        return redirect(request.referrer or url_for('web.admin_dashboard'))
        
    if not nama_dataset:
        flash('Nama dataset tidak valid', 'error')
        return redirect(request.referrer or url_for('web.admin_dashboard'))
        
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if nama_dataset == 'ALL':
                cur.execute(f"DELETE FROM {table_name}")
            else:
                cur.execute(f"DELETE FROM {table_name} WHERE nama_dataset = %s", (nama_dataset,))
            conn.commit()
        flash('Dataset berhasil dihapus.', 'success')
    except Exception as e:
        flash(f'Gagal: {str(e)}', 'error')
    finally:
        conn.close()
        
    return redirect(request.referrer or url_for('web.admin_dashboard'))


# ===================== PREDIKSI TES =====================
@web.route('/prediksi/TES', methods=['GET', 'POST'])
@admin_required
def web_prediksi_tes():
    conn = get_conn()
    prediction_results = []
    last_name = None
    if request.method == 'POST':
        nama_ds = request.form.get('nama_dataset', '')
        nama_pred = request.form.get('nama_prediksi', '')
        if nama_ds and nama_pred:
            import requests as req
            try:
                resp = req.post('http://127.0.0.1:5000/prediksi', json={'nama_dataset': nama_ds, 'nama_prediksi': nama_pred})
                data = resp.json()
                if data.get('success') or data.get('status') == 'success':
                    flash('Prediksi berhasil!', 'success')
                    last_name = nama_pred
                else:
                    flash(data.get('message', data.get('error', 'Gagal')), 'error')
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT nama_dataset FROM dataset_tes")
            avail = [r['nama_dataset'] for r in cur.fetchall()]
            if last_name:
                cur.execute("SELECT * FROM hasil_prediksi_tes WHERE nama_prediksi=%s ORDER BY tahun, bulan", (last_name,))
                prediction_results = cur.fetchall()
    finally:
        conn.close()
    return render_template('prediksi.html', active_page='prediksi.tes',
        page_title='Prediksi Paket (TES)', page_subtitle='Estimasi volume paket menggunakan model TES',
        page_icon='ti-bar-chart-alt', form_action=url_for('web.web_prediksi_tes'),
        show_training_testing=False, available_datasets=avail,
        show_sla_columns=False, prediction_results=prediction_results,
        last_prediction_name=last_name, bulan_names=BULAN_NAMES)

# ===================== PREDIKSI MLP =====================
@web.route('/prediksi/MLP', methods=['GET', 'POST'])
@admin_required
def web_prediksi_mlp():
    conn = get_conn()
    prediction_results = []
    last_name = None
    if request.method == 'POST':
        nama_ds = request.form.get('nama_dataset_training', '')
        nama = request.form.get('nama_prediksi', '')
        if nama_ds and nama:
            import requests as req
            try:
                resp = req.post('http://127.0.0.1:5000/prediksi-xgboost', json={
                    'nama_dataset': nama_ds, 'nama_prediksi': nama})
                data = resp.json()
                if data.get('success') or data.get('status') == 'success':
                    flash('Prediksi berhasil!', 'success')
                    last_name = nama
                else:
                    flash(data.get('message', data.get('error', 'Gagal')), 'error')
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT nama_dataset FROM dataset_xgboost")
            training = [r['nama_dataset'] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT nama_prediksi FROM hasil_prediksi_tes")
            testing = [r['nama_prediksi'] for r in cur.fetchall()]
            if last_name:
                cur.execute("SELECT * FROM hasil_prediksi_xgboost WHERE nama_prediksi=%s ORDER BY tahun, bulan", (last_name,))
                prediction_results = cur.fetchall()
    finally:
        conn.close()
    return render_template('prediksi.html', active_page='prediksi.mlp',
        page_title='Permintaan Prediksi SLA (XGBoost Regression)', page_subtitle='Estimasi performa SLA menggunakan XGBoost',
        page_icon='ti-stats-up', form_action=url_for('web.web_prediksi_mlp'),
        show_training_testing=True, list_dataset_training=training, list_dataset_testing=testing,
        show_sla_columns=True, prediction_results=prediction_results,
        last_prediction_name=last_name, bulan_names=BULAN_NAMES)

# ===================== PREDIKSI XGBOOST =====================
@web.route('/prediksi/XGBoost', methods=['GET', 'POST'])
@admin_required
def web_prediksi_xgboost():
    conn = get_conn()
    prediction_results = []
    last_name = None
    if request.method == 'POST':
        nama_ds = request.form.get('nama_dataset_training', '')
        nama = request.form.get('nama_prediksi', '')
        if nama_ds and nama:
            import requests as req
            try:
                resp = req.post('http://127.0.0.1:5000/prediksi-xgboost', json={
                    'nama_dataset': nama_ds, 'nama_prediksi': nama})
                data = resp.json()
                if data.get('success') or data.get('status') == 'success':
                    flash('Prediksi berhasil!', 'success')
                    last_name = nama
                else:
                    flash(data.get('message', data.get('error', 'Gagal')), 'error')
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT nama_dataset FROM dataset_xgboost")
            training = [r['nama_dataset'] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT nama_prediksi FROM hasil_prediksi_tes")
            testing = [r['nama_prediksi'] for r in cur.fetchall()]
            if last_name:
                cur.execute("SELECT * FROM hasil_prediksi_xgboost WHERE nama_prediksi=%s ORDER BY tahun, bulan", (last_name,))
                prediction_results = cur.fetchall()
    finally:
        conn.close()
    return render_template('prediksi.html', active_page='prediksi.xgboost',
        page_title='Prediksi SLA (XGBoost)', page_subtitle='Estimasi performa SLA menggunakan XGBoost',
        page_icon='ti-stats-up', form_action=url_for('web.web_prediksi_xgboost'),
        show_training_testing=True, list_dataset_training=training, list_dataset_testing=testing,
        show_sla_columns=True, prediction_results=prediction_results,
        last_prediction_name=last_name, bulan_names=BULAN_NAMES)

# ===================== HASIL PREDIKSI =====================
def _hasil_page(table, active, title, subtitle, icon, show_sla, delete_action):
    conn = get_conn()
    try:
        page = int(request.args.get('page', 1))
        per_page = 20
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) as c FROM {table}")
            total = cur.fetchone()['c']
            cur.execute(f"SELECT * FROM {table} ORDER BY tahun, bulan LIMIT %s OFFSET %s", (per_page, (page-1)*per_page))
            results = cur.fetchall()
        total_pages = (total + per_page - 1) // per_page
    finally:
        conn.close()
    pagination = {'page': page, 'pages': total_pages} if total_pages > 1 else None
    return render_template('hasil_prediksi.html', active_page=active,
        page_title=title, page_subtitle=subtitle, page_icon=icon,
        results=results, total_results=total, show_sla_columns=show_sla,
        pagination=pagination, delete_action=delete_action, bulan_names=BULAN_NAMES)

@web.route('/Hasil/TES')
@admin_required
def web_hasil_prediksi_tes():
    return _hasil_page('hasil_prediksi_tes', 'hasil.prediksi.tes',
        'Hasil Prediksi Kepadatan Paket (TES)', 'Laporan detail volume paket', 'ti-bar-chart-alt',
        False, url_for('web.web_delete_hasil', table='hasil_prediksi_tes'))

@web.route('/Hasil/MLP')
@admin_required
def web_hasil_prediksi_mlp():
    return _hasil_page('hasil_prediksi_xgboost', 'hasil.prediksi.mlp',
        'Hasil Prediksi Status SLA (XGBoost Regression)', 'Laporan detail kinerja SLA', 'ti-stats-up',
        True, url_for('web.web_delete_hasil', table='hasil_prediksi_xgboost'))

@web.route('/Hasil/XGBoost')
@admin_required
def web_hasil_prediksi_xgboost():
    return _hasil_page('hasil_prediksi_xgboost', 'hasil.prediksi.xgboost',
        'Hasil Prediksi Status SLA (XGBoost)', 'Laporan detail kinerja SLA XGBoost', 'ti-stats-up',
        True, url_for('web.web_delete_hasil', table='hasil_prediksi_xgboost'))

@web.route('/delete-hasil/<table>', methods=['POST'])
@admin_required
def web_delete_hasil(table):
    allowed = ['hasil_prediksi_tes', 'hasil_prediksi_rfr', 'hasil_prediksi_xgboost']
    if table not in allowed:
        flash('Invalid table', 'error')
        return redirect(url_for('web.admin_dashboard'))
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table}")
            conn.commit()
        flash('Semua data berhasil dihapus.', 'success')
    finally:
        conn.close()
    route_map = {'hasil_prediksi_tes': 'web.web_hasil_prediksi_tes', 'hasil_prediksi_rfr': 'web.web_hasil_prediksi_mlp', 'hasil_prediksi_xgboost': 'web.web_hasil_prediksi_xgboost'}
    return redirect(url_for(route_map[table]))

# ===================== USER MANAGEMENT =====================
@web.route('/users')
@admin_required
def user_management():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, name, role FROM users ORDER BY id")
            users = cur.fetchall()
    finally:
        conn.close()
    return render_template('user_management.html', active_page='users.index', users=users)

@web.route('/users/add', methods=['POST'])
@admin_required
def web_add_user():
    u = request.form.get('username', '')
    n = request.form.get('name', '')
    p = request.form.get('password', '')
    r = request.form.get('role', 'admin')
    if u and n and p:
        hashed = bcrypt.hashpw(p.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users (username, name, password, role, created_at, updated_at) VALUES (%s,%s,%s,%s,NOW(),NOW())", (u, n, hashed, r))
                conn.commit()
            flash('User berhasil ditambahkan.', 'success')
        except Exception as e:
            flash(f'Gagal: {str(e)}', 'error')
        finally:
            conn.close()
    return redirect(url_for('web.user_management'))

@web.route('/users/delete/<int:user_id>', methods=['POST'])
@admin_required
def web_delete_user(user_id):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
            conn.commit()
        flash('User berhasil dihapus.', 'success')
    finally:
        conn.close()
    return redirect(url_for('web.user_management'))
