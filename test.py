import torch
import torchvision

print(torch.cuda.is_available())
print(torchvision.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())