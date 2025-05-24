import torch

print("CUDA is available", torch.cuda.is_available())

print("GPU Number", torch.cuda.device_count())

print("Available GPU", torch.cuda.current_device())

print("GPU name", torch.cuda.get_device_name())

print("GPU Skills", torch.cuda.get_device_capability(torch.cuda.current_device()))