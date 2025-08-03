import matplotlib.pyplot as plt
import ast

x_vals, y_vals = [], []

with open("vo_trajectory_log.csv", "r") as f:
    lines = f.readlines()
#with open("groundtruth_log.csv", "r") as f:
#    lines = f.readlines()    

# Lewati header (2 baris pertama)
for row in lines[2:]:
    try:
        parsed = ast.literal_eval(row.strip().strip('"'))
        frame, x, y = parsed[0], parsed[1], parsed[2]
        x_vals.append(x)
        y_vals.append(y)
    except Exception as e:
        print(f"Error parsing row: {row} -> {e}")

# Visualisasi
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, linestyle='-', color='blue')
plt.title('Trajectory (X vs Y)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.axis('equal')
plt.show()
