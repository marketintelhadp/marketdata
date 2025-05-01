import sqlite3

conn = sqlite3.connect('market_data.db')  # Update this with your actual DB path
cursor = conn.cursor()

cursor.execute("ALTER TABLE market_data ADD COLUMN User TEXT;")

conn.commit()
conn.close()

print("Column 'user' added successfully.")
