import os
import subprocess
import threading
import time
import numpy as np
import cv2
from pathlib import Path

class YUVStreamer:
    def __init__(self, pipe_path="live.yuv", output_dir="dataset", width=640, height=480):
        self.pipe_path = pipe_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.frame_size = width * height * 3 // 2  # YUV420
        self.frame_id = 0
        self._stop = False

    def yuv420_to_bgr(self, yuv_bytes):
        yuv = np.frombuffer(yuv_bytes, dtype=np.uint8)
        yuv = yuv.reshape((self.height * 3 // 2, self.width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return bgr

    def run(self):
        print(f"📂 Output dir: {self.output_dir}")
        try:
            with open(self.pipe_path, "rb") as pipe:
                while not self._stop:
                    yuv_bytes = pipe.read(self.frame_size)
                    if len(yuv_bytes) < self.frame_size:
                        print("⚠️ Incomplete frame received.")
                        continue
                    frame = self.yuv420_to_bgr(yuv_bytes)
                    self.frame_id += 1
                    fname = self.output_dir / f"frame_{self.frame_id:05d}.png"
                    cv2.imwrite(str(fname), frame)
                    print(f"[{self.frame_id}] Saved: {fname}")
        except FileNotFoundError:
            print(f"❌ Pipe '{self.pipe_path}' not found.")

    def stop(self):
        self._stop = True


def start_libcamera(pipe_path, width, height):
    if not os.path.exists(pipe_path):
        print(f"🔧 Creating named pipe: {pipe_path}")
        os.mkfifo(pipe_path)

    cmd = [
        "libcamera-vid",
        "--width", str(width),
        "--height", str(height),
        "--codec", "yuv420",
        "--inline",
        "--nopreview",
        "-t", "0",
        "-o", pipe_path
    ]
    print(f"🚀 Starting libcamera-vid: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def main():
    pipe_path = "live.yuv"
    output_dir = "dataset"
    width, height = 640, 480

    cam_proc = start_libcamera(pipe_path, width, height)
    time.sleep(1)  # wait for libcamera-vid to start

    streamer = YUVStreamer(pipe_path=pipe_path, output_dir=output_dir, width=width, height=height)
    try:
        streamer.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Stopping processes...")
    finally:
        streamer.stop()
        cam_proc.terminate()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
