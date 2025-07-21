import os
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

    def yuv420_to_bgr(self, yuv_bytes):
        yuv = np.frombuffer(yuv_bytes, dtype=np.uint8)
        yuv = yuv.reshape((self.height * 3 // 2, self.width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return bgr

    def run(self):
        print(f"📂 Output dir: {self.output_dir}")
        with open(self.pipe_path, "rb") as pipe:
            while True:
                yuv_bytes = pipe.read(self.frame_size)
                if len(yuv_bytes) < self.frame_size:
                    print("⚠️ Incomplete frame received.")
                    continue
                frame = self.yuv420_to_bgr(yuv_bytes)
                self.frame_id += 1
                fname = self.output_dir / f"frame_{self.frame_id:05d}.png"
                cv2.imwrite(str(fname), frame)
                print(f"[{self.frame_id}] Saved: {fname}")

if __name__ == "__main__":
    streamer = YUVStreamer()
    streamer.run()
