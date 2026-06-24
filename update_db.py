import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='prediksi_svr')
try:
    with conn.cursor() as cur:
        # Delete unused models to only leave the two we want
        # The user has: eval_lowokwaru, uji_evaluasi_1, uji_overfit_1, uji_overfit_2
        # We will delete eval_lowokwaru and uji_evaluasi_1
        cur.execute("DELETE FROM hasil_prediksi_tes WHERE nama_prediksi IN ('eval_lowokwaru', 'uji_evaluasi_1')")
        cur.execute("DELETE FROM hasil_prediksi_rfr WHERE nama_prediksi IN ('eval_lowokwaru', 'uji_evaluasi_1')")
        
        # Rename uji_overfit_1 to eval 1
        cur.execute("UPDATE hasil_prediksi_tes SET nama_prediksi='eval 1' WHERE nama_prediksi='uji_overfit_1'")
        cur.execute("UPDATE hasil_prediksi_rfr SET nama_prediksi='eval 1' WHERE nama_prediksi='uji_overfit_1'")
        
        # Rename uji_overfit_2 to eval 2
        cur.execute("UPDATE hasil_prediksi_tes SET nama_prediksi='eval 2' WHERE nama_prediksi='uji_overfit_2'")
        cur.execute("UPDATE hasil_prediksi_rfr SET nama_prediksi='eval 2' WHERE nama_prediksi='uji_overfit_2'")
        
    conn.commit()
    print("Database updated successfully.")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
