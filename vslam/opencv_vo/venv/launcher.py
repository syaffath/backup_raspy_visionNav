# launcher.py

import os
import subprocess
from yuv_streamer import YUVStreamer

def start_libcamera(pipe_path="/tmp/live.yuv", width=640, height=480):
    if os.path.exists(pipe_path):
        os.remove(pipe_path)
    os.mkfifo(pipe_path)
    print(f"🛠️ Created named pipe: {pipe_path}")
    
    cmd = [
        "libcamera-vid",
        "-t", "0",
        "--width", str(width),
        "--height", str(height),
        "--codec", "yuv420",
        "--inline",
        "--nopreview",
        "-o", pipe_path
    ]
    print(f"🚀 Starting libcamera-vid: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def main():
    pipe_path = "/tmp/live.yuv"
    width, height = 640, 480

    cam_proc = start_libcamera(pipe_path, width, height)
    
    try:
        streamer = YUVStreamer(pipe_path=pipe_path, width=width, height=height)
        streamer.run()
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    finally:
        cam_proc.terminate()
        print("🧹 libcamera-vid terminated.")

if __name__ == "__main__":
    main()
