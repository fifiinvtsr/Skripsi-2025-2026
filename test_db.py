import pymysql
conn=pymysql.connect(host='localhost',user='root',password='',db='prediksi_svr',cursorclass=pymysql.cursors.DictCursor)
cur=conn.cursor()
cur.execute("SELECT * FROM hasil_prediksi_xgboost WHERE nama_prediksi='TEST_MAPE_RMSE'")
print(cur.fetchall())
cur.execute("SELECT * FROM hasil_prediksi_tes WHERE nama_prediksi='Prediksi_Paket_W2_2025'")
print(cur.fetchall())
