import torch
import time
import os

torch.cuda.empty_cache()

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


device = 'cuda:0'


dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1024/1024 = 360.0M


memory_allocated = torch.cuda.memory_allocated(device)/1024/1024

memory_reserved = torch.cuda.memory_reserved(device)/1024/1024


print("第一阶段：")
print("变量类型：", dummy_tensor_4.dtype)
print("变量实际占用内存空间：", 120*3*512*512*4/1024/1024, "M")
print("GPU实际分配给的可用内存", memory_allocated, "M")
print("GPU实际分配给的缓存", memory_reserved, "M")

torch.cuda.empty_cache()

time.sleep(15)

memory_allocated = torch.cuda.memory_allocated(device)/1024/1024

memory_reserved = torch.cuda.memory_reserved(device)/1024/1024

print("第二阶段：")
print("释放缓存后:", "."*100)
print("变量实际占用内存空间：", 120*3*512*512*4/1024/1024, "M")
print("GPU实际分配给的可用内存", memory_allocated, "M")
print("GPU实际分配给的缓存", memory_reserved, "M")


del dummy_tensor_4

torch.cuda.empty_cache()

time.sleep(15)

memory_allocated = torch.cuda.memory_allocated(device)/1024/1024

memory_reserved = torch.cuda.memory_reserved(device)/1024/1024


print("第三阶段：")
print("删除变量后释放缓存后:", "."*100)
print("变量实际占用内存空间：", 0, "M")
print("GPU实际分配给的可用内存", memory_allocated, "M")
print("GPU实际分配给的缓存", memory_reserved, "M")
