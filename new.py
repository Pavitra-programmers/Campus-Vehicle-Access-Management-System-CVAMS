import sqlite3

# Connect to the database
conn = sqlite3.connect('vehicle_data.db')
c = conn.cursor()

# Delete all entries from the vehicle_data table
c.execute("DELETE FROM vehicle_data")

# Commit the changes and close the connection
conn.commit()
conn.close()
print("All entries deleted successfully.")
