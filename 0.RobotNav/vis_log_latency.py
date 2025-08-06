import matplotlib.pyplot as plt
import csv
import ast

# --- Fungsi Baca Log ---
def read_timing_log(path):
    data = {
        "frame_idx": [],
        "t_capture_ms": [],
        #"t_yolo_ms": [],
        "t_vo_ms": [],
        "t_gt_ms": [],
        "t_control_show_ms": [],
        "t_total_ms": [],
        "fps": []
    }
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            parsed = ast.literal_eval(row[0])
            data["frame_idx"].append(parsed[0])
            data["t_capture_ms"].append(parsed[1])
            #data["t_yolo_ms"].append(parsed[2])
            data["t_vo_ms"].append(parsed[2])
            data["t_gt_ms"].append(parsed[3])
            data["t_control_show_ms"].append(parsed[4])
            data["t_total_ms"].append(parsed[5])
            data["fps"].append(parsed[6])
    return data

# --- Ganti path sesuai file log ---
log_path = "latency_fps_log_without YOLO.csv"
data = read_timing_log(log_path)

# --- Hitung Rata-rata ---
avg_capture = sum(data["t_capture_ms"]) / len(data["t_capture_ms"])
#avg_yolo = sum(data["t_yolo_ms"]) / len(data["t_yolo_ms"])
avg_vo = sum(data["t_vo_ms"]) / len(data["t_vo_ms"])
avg_gt = sum(data["t_gt_ms"]) / len(data["t_gt_ms"])
avg_control = sum(data["t_control_show_ms"]) / len(data["t_control_show_ms"])
avg_total = sum(data["t_total_ms"]) / len(data["t_total_ms"])
avg_fps = sum(data["fps"]) / len(data["fps"])

# --- Buat Summary String ---
summary_text = (
    f"=== Average Latency & FPS ===\n"
    #f"Capture: {avg_capture:.2f} ms\n"
    #f"YOLO: {avg_yolo:.2f} ms\n"
    f"VO: {avg_vo:.2f} ms\n"
    f"GroundTruth: {avg_gt:.2f} ms\n"
    f"Control+Display: {avg_control:.2f} ms\n"
    f"Total: {avg_total:.2f} ms\n"
    f"FPS: {avg_fps:.2f}"
)

# --- Plot Waktu Eksekusi ---
plt.figure(figsize=(12, 7))
#plt.plot(data["frame_idx"], data["t_yolo_ms"], label="YOLO (ms)")
plt.plot(data["frame_idx"], data["t_vo_ms"], label="VO (ms)")
plt.plot(data["frame_idx"], data["t_gt_ms"], label="Groundtruth (ms)")
plt.plot(data["frame_idx"], data["t_control_show_ms"], label="Robot Strategies (ms)")
plt.plot(data["frame_idx"], data["t_total_ms"], label="Total (ms)", linewidth=2)

plt.legend(loc='upper right')
plt.grid(True)
plt.xlabel("Frame Index")
plt.ylabel("Time (ms)")
plt.title("Execution Time per Component")

# Summary di pojok kiri atas
plt.text(
    0.02, 0.98, summary_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# --- Plot FPS ---
plt.figure(figsize=(10, 4))
plt.plot(data["frame_idx"], data["fps"], color='blue')
plt.xlabel("Frame Index")
plt.ylabel("FPS")
plt.title("Frames Per Second")
plt.grid(True)

plt.show()
