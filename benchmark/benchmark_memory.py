import torch
from niuload import balanced_load
from transformers import AutoModelForCausalLM
import gc
import subprocess
import time
import json
import matplotlib.pyplot as plt
import numpy as np

tested_models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2b"
]

def load_method1(model_name):
    return balanced_load(model_name, show_hf_device=False)

def load_method2(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def get_gpu_memory_usage():
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used,memory.total',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    gpu_memory = [list(map(int, x.split(','))) for x in result.strip().split('\n')]
    return {i: {'used': mem[0], 'total': mem[1]} for i, mem in enumerate(gpu_memory)}

def test_max_memory_usage(load_function, model_name):
    clear_gpu_memory()
    before_load = get_gpu_memory_usage()
    model = load_function(model_name)

    after_load = get_gpu_memory_usage()
    memory_usage = {idx: {
        'used': after_load[idx]['used'] - before_load[idx]['used'],
        'total': after_load[idx]['total']
    } for idx in after_load}
    del model
    clear_gpu_memory()
    return memory_usage

if not torch.cuda.is_available():
    print("CUDA is not available. This script requires a GPU.")
    exit()

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

results = {}

for model_name in tested_models:
    print(f"\nTesting model: {model_name}")
    memory_used1 = test_max_memory_usage(load_method1, model_name)
    print("Method 1 (balanced_load) memory usage per GPU:")
    for idx, mem in memory_used1.items():
        print(f"  GPU {idx}: {mem['used']} MB / {mem['total']} MB ({mem['used']/mem['total']*100:.2f}%)")
    
    memory_used2 = test_max_memory_usage(load_method2, model_name)
    print("Method 2 (AutoModelForCausalLM) memory usage per GPU:")
    for idx, mem in memory_used2.items():
        print(f"  GPU {idx}: {mem['used']} MB / {mem['total']} MB ({mem['used']/mem['total']*100:.2f}%)")
    
    results[model_name] = {"Method 1": memory_used1, "Method 2": memory_used2}
    


# 保存结果到 JSON 文件
with open('gpu_memory_usage_data.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Data saved to 'gpu_memory_usage_data.json'")

# 绘图代码
num_models = len(tested_models)
bar_width = 0.35
index = np.arange(num_gpus)

fig, axs = plt.subplots(num_models, 1, figsize=(15, 8*num_models), sharex=True)
fig.suptitle("GPU Memory Usage per Model and Loading Method", fontsize=16, y=1.02)

for i, (model_name, data) in enumerate(results.items()):
    method1_data = [data["Method 1"].get(j, {'used': 0, 'total': 1})['used'] / data["Method 1"].get(j, {'used': 0, 'total': 1})['total'] * 100 for j in range(num_gpus)]
    method2_data = [data["Method 2"].get(j, {'used': 0, 'total': 1})['used'] / data["Method 2"].get(j, {'used': 0, 'total': 1})['total'] * 100 for j in range(num_gpus)]
    
    axs[i].bar(index, method1_data, bar_width, label='Method 1 (balanced_load)')
    axs[i].bar(index + bar_width, method2_data, bar_width, label='Method 2 (AutoModelForCausalLM)')
    
    axs[i].set_ylabel('Memory Usage (%)')
    axs[i].set_title(f'Model: {model_name.split("/")[-1]}')
    axs[i].set_xticks(index + bar_width / 2)
    axs[i].set_xticklabels([f'GPU {j}' for j in range(num_gpus)])
    axs[i].legend()
    axs[i].set_ylim(0, 25)

    for j, v1, v2 in zip(index, method1_data, method2_data):
        axs[i].text(j, v1, f'{v1:.1f}%', ha='center', va='bottom')
        axs[i].text(j + bar_width, v2, f'{v2:.1f}%', ha='center', va='bottom')

    axs[i].axhline(y=100, color='r', linestyle='--', linewidth=1)

plt.tight_layout(rect=[0, 0, 1, 0.99])  # 为标题留出空间
plt.savefig('gpu_memory_usage_percentage.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'gpu_memory_usage_percentage.png'")