import torch
import time
from niuload import balanced_load
from transformers import AutoModelForCausalLM
import gc
tested_models = [
    # "/mnt/rangehow/models/Qwen2.5-7B-Instruct",
    "/mnt/rangehow/models/gemma-2-9b-it",
    # "/mnt/rangehow/models/Meta-Llama-3.1-8B-Instruct",
    # "/mnt/rangehow/models/gemma-2b"
]

def print_model_device(model):
    for name, module in model.named_modules():
        # 获取该模块的所有参数的设备
        if any(p.device for p in module.parameters()):
            device = next(module.parameters()).device
        else:
            # 如果没有参数，可以检查缓冲区的设备
            if any(b.device for b in module.buffers()):
                device = next(module.buffers()).device
            else:
                # 如果模块既没有参数也没有缓冲区，则设备为None（可能是容器模块，如Sequential）
                device = 'No parameters or buffers'
        
        print(f"Module: {name}, Device: {device}")


def load_method1(model_name):
    return balanced_load(model_name, show_hf_device=False)

def load_method2(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

def generate_random_input(batch_size, seq_length, vocab_size):
    return torch.randint(0, vocab_size, (batch_size, seq_length))

def test_forward_pass(model, input_ids, num_warmup=1, num_actual=1):
    # 预热运行
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_ids)
    
    # 确保GPU操作完成
    torch.cuda.synchronize()
    
    # 使用CUDA事件进行计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 实际测量
    total_time = 0
    for _ in range(num_actual):
        start_event.record()
        with torch.no_grad():
            _ = model(input_ids)
            # print(result[0])
        end_event.record()
        
        # 等待GPU操作完成
        torch.cuda.synchronize()
        
        total_time += start_event.elapsed_time(end_event)
    
    del start_event, end_event
    return total_time / num_actual  # 返回平均时间（毫秒）

# 测试参数
batch_size = 1
seq_length = 4096
vocab_size = 32000
num_runs = 8

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("CUDA is not available. Running on CPU.")

for model_name in tested_models:
    print(f"Testing model: {model_name}")
    
    # 生成随机输入，并确保在两个模型中一致
    input_ids = generate_random_input(batch_size, seq_length, vocab_size)
    
   
    
    # 测试方法2
    model2 = load_method2(model_name)
    print("Method 2 (AutoModelForCausalLM):")
    run_times_sum = 0
    for _ in range(num_runs):
        # 将相同的输入移动到GPU（如果可用的话）
        if cuda_available:
            input_ids_cuda = input_ids.to(model2.device)
        else:
            input_ids_cuda = input_ids
        
        time_taken = test_forward_pass(model2, input_ids_cuda)
        run_times_sum += time_taken
    
    print(f"  Total run time: {run_times_sum/1000:.2f} seconds")
    
    
    del model2
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
     # 测试方法1
    model1 = load_method1(model_name)
    run_times_sum = 0
    print("Method 1 (balanced_load):")
    for _ in range(num_runs):
        # 将输入移动到GPU（如果可用的话）
        if cuda_available:
            input_ids_cuda = input_ids.to(model1.device)
        else:
            input_ids_cuda = input_ids
        
        time_taken = test_forward_pass(model1, input_ids_cuda)
        run_times_sum += time_taken
    
    print(f"  Total run time: {run_times_sum/1000:.2f} seconds")
    
    
    del model1
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()
 
    
    print("\n")