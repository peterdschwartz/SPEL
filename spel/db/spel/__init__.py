# import mariadb
# import sys
# global conn, cur
# try:
#     conn = mariadb.connect(
#         user="root",
#         password="root",
#         host="127.0.0.1",
#         port=3306,
#         database="spel",
#         # unix_socket='/var/run/mysqld/mysqld.sock'  # Use the correct path
#     )
# except mariadb.Error as e:
#     print("error")
#     sys.exit(1)
# cur = conn.cursor()