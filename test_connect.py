import pymysql
connection = pymysql.connect(
    host='sh-cynosdbmysql-grp-d2ld76ic.sql.tencentcdb.com',
    port=21240,
    user='deviceCloud',
    password='L-Y47!hB!z9.zGA',
    db='deviceCloud',
    charset='utf8'
)
print("OK")

connection.close()
