import torch

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(torch.cuda.current_device()))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
    #time.sleep(5)
