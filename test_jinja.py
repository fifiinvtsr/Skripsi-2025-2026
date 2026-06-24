from jinja2 import Template
import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='prediksi_svr', cursorclass=pymysql.cursors.DictCursor)
c = conn.cursor()
c.execute('SELECT * FROM hasil_prediksi_xgboost ORDER BY id DESC LIMIT 1')
p = c.fetchone()
print("p:", p)
t = Template('{{ "{:.2f}%".format(p.mape_faster) if p.mape_faster is not none else "-" }}')
print("rendered:", t.render(p=p))
