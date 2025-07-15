from hailo_platform.pyhailort import pyhailort
import numpy as np
import time
from picamera2 import Picamera2
import cv2

hef_path = "/home/raspy/Documents/syafiq/object_detection/yolo-picamera2/yolov5n.hef"

hef = pyhailort.HEF(hef_path)
device = pyhailort.Device()

activated_net = pyhailort.ActivatedNetwork(device, hef)
print("Network loaded!")

input_names = [info.name for info in hef.get_input_vstream_infos()]
output_names = [info.name for info in hef.get_output_vstream_infos()]
print("Input names:", input_names)
print("Output names:", output_names)

input_info = hef.get_input_vstream_infos()[0]
h, w, c = input_info.shape
shape = (1, c, h, w)
dummy_input = np.zeros(shape, dtype=np.float32)

input_params = []
for info in hef.get_input_vstream_infos():
    p = pyhailort.InputVStreamParams()
    p.name = info.name
    input_params.append(p)

output_params = []
for info in hef.get_output_vstream_infos():
    p = pyhailort.OutputVStreamParams()
    p.name = info.name
    output_params.append(p)

input_vstreams = pyhailort.InputVStreams(activated_net, input_params)
output_vstreams = pyhailort.OutputVStreams(activated_net, output_params)

input_vstreams[0].send(dummy_input)
output = output_vstreams[0].recv()
print("Output shape:", output.shape)

del input_vstreams
del output_vstreams
del activated_net
del device
