import mariadb
import sys

def connect():
    try:
        conn = mariadb.connect(
            user="root",
            password="root",
            host="127.0.0.1",
            port=3306,
            database="spel",
        )
    except mariadb.Error as e:
        print("error")
        sys.exit(1)
    cur = conn.cursor()
    return cur, conn