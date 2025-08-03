import matplotlib.pyplot as plt
import csv
import ast

# Ganti path ini dengan lokasi file log kamu
log_file = "groundtruth_log.csv"

# Penampung nilai x dan y
x_vals, y_vals = [], []

# Baca dan parse file
with open(log_file, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # lewati header
    for row in reader:
        try:
            parsed = ast.literal_eval(row[0])  # konversi string ke list Python
            x_vals.append(parsed[1])
            y_vals.append(parsed[2])
        except Exception as e:
            print(f"Baris dilewati karena error: {e}")

# Visualisasi
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, linestyle='-', color='blue')
plt.title("Trajectory X vs Y")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
