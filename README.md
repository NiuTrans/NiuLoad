# NiuLoad

niuload is a Python module for balanced loading large language models across multiple GPUs(pipeline parallel).
This package can be used during training and inference.

## Report (Chinese)
More details can be found at:
https://zhuanlan.zhihu.com/p/792303768 

## Usage

niuload offers the following advanced features:

1. Customizable load distribution across each GPU. If not specified, the model will be evenly distributed across all GPUs by default.

```python
from niuload import balanced_load
model = balanced_load("openai-community/gpt2", ratio=["0.5,1,1"], num_devices=4)
```
This example allows you to split the model across four GPUs, with GPU 0 only having half the load compared to the others.

2. Specify which GPUs to use for model splitting.

```python
from niuload import balanced_load
model = balanced_load("openai-community/gpt2", device_idx=[1,2,5,7], num_devices=4)
```
3. Support for training. Models automatically split by Hugging Face's `device_map='auto'` do not always support training due to a current bug in accelerate. Specifically, when model.device is not on device 0, certain models can encounter errors. We currently force embeddings to be on GPU 0 to avoid this issue.

4. Support for splitting models that are not supported by Hugging Face, but this requires some additional adaptation. We are working on reorganizing the code structure soon and welcome community contributions to the project.


## Benchmark
For reproducing our results in report, you can checkout scripts under /benchmark.

## FQA
1. Q: Why do I see the model being loaded twice?

    A: To better analyze the model's parameter structure, we first load the model onto a meta device. This process is very fast because the initialized tensors are random and meaningless. After obtaining the device_map, we then actually load the model.
