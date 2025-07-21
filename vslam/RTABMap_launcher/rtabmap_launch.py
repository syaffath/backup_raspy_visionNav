# launcher.py
import os
import subprocess
import time
from yuv_streamer import YUVStreamer

def start_libcamera(pipe_path="live.yuv", width=640, height=480):
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
        print(f"🛠️ Created named pipe: {pipe_path}")
    
    cmd = [
        "libcamera-vid", "-t", "0",
        "--width", str(width), "--height", str(height),
        "--codec", "yuv420", "--inline", "--nopreview",
        "-o", pipe_path
    ]
    print(f"🚀 Starting libcamera-vid...")
    return subprocess.Popen(cmd)

def start_rtabmap_runner():
    print("🎯 Starting RTAB-Map runner in background...")
    return subprocess.Popen(["python3", "rtabmap_runner.py"])

def main():
    pipe_path = "live.yuv"
    width, height = 640, 480

    cam_proc = start_libcamera(pipe_path, width, height)
    rtabmap_proc = start_rtabmap_runner()

    try:
        streamer = YUVStreamer(pipe_path=pipe_path, width=width, height=height)
        streamer.run()
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    finally:
        print("🧹 Cleaning up...")
        cam_proc.terminate()
        rtabmap_proc.terminate()

if __name__ == "__main__":
    main()
