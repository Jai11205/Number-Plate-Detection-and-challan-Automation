import sqlite3
conn = sqlite3.connect('/content/traffic_violations.db')

def init_database():

    conn.execute("PRAGMA foreign_keys = ON;") # CRITICAL: Enable FK constraints
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS vehicles
               (number_plate TEXT PRIMARY KEY, owner_name TEXT, violations INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS challan (
    challan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    number_plate TEXT,
    violation_type TEXT,
    fine_amount REAL,
    violations INT,
    evidence_path TEXT, 
    -- Store the path to the image/clip here
    date_issued TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(number_plate) REFERENCES vehicles(number_plate)
)''')
    conn.commit()
    return conn