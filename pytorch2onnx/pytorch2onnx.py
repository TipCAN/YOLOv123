import torch.onnx
import sys
sys.path.append("..")
from models.resnet_YOLO import resnet50

# An instance of your model
model = resnet50()
model_path = '/home/yjh/yolos/yolo_pytorch_v1/save_person1/epoch_500.pth'
model.load_state_dict(torch.load(model_path).state_dict())
print(model)
# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 448, 448)

# Export the model
torch_out = torch.onnx._export(model, x, "./yolov12.onnx", export_params=True)