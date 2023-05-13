import torch
from models.experimental import attempt_load
from models.yolo import Model
model = Model(cfg = 'models/yolov5s.yaml', ch=3, nc=80).to('cpu')
x = torch.randn(1,3,640,640)
export_onnx_file = "./yolov5s.onnx"
model.model[-1].export = True
model = attempt_load('./weights/yolov5s.pt', map_location='cpu')  # load FP32 model
torch.onnx._export(model,x,export_onnx_file,verbose=True)
print('OK')
