from flask import Flask
import psycopg2
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Connect to the PostgreSQL database
def get_db_connection():
    conn = psycopg2.connect(app.config['DATABASE_URI'])
    print(conn)
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products') 
    records = cursor.fetchall()
    conn.close()
    
    return f"Database records: {records}"

if __name__ == "__main__":
    app.run(debug=True)
