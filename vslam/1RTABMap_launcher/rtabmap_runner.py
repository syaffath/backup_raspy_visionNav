# rtabmap_runner.py
import subprocess
import time
import os

def run_rtabmap_loop(dataset_dir="dataset", interval=30):
    print(f"🧭 RTAB-Map monitoring folder '{dataset_dir}' every {interval}s")

    while True:
        # Pastikan folder dataset ada dan punya isi
        if os.path.isdir(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
            print("📡 Running RTAB-Map...")
            subprocess.run(["rtabmap-console", dataset_dir])
        else:
            print("⏳ Waiting for dataset to be populated...")

        time.sleep(interval)

if __name__ == "__main__":
    try:
        run_rtabmap_loop()
    except KeyboardInterrupt:
        print("🛑 RTAB-Map loop stopped by user.")
