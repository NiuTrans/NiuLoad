import accelerate
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
import math
from torch.cuda import device_count


model_type2lm_head_modules = {
    "fsmt": lambda model: model.model.decoder.output_projection,
    "llama": lambda model: model.lm_head,
    "gemma": lambda model: model.lm_head,
    "qwen2": lambda model: model.lm_head,
    "mistral": lambda model: model.lm_head,
    "bart": lambda model: model.lm_head,
}

model_type2lm_head_names = {
    "fsmt": "model.decoder.output_projection",
    "llama": "lm_head",
    "gemma": "lm_head",
    "qwen2": "lm_head",
    "mistral": "lm_head",
    "bart": "lm_head",
}

def balanced_partition(items, k):
    """
    使用贪心策略将字典 items 的 item 尽可能均分成 k 份
    :param items: dict，key 为 item 名称，value 为 item 对应的值
    :param k: int，目标分成的份数
    :return: list，包含 k 个列表，每个列表是一个分配的组，元素是分配的量
    """
    # 获取所有 item 的值
    values = list(items.values())
    
    # 目标是分成 k 份
    groups = [[] for _ in range(k)]
    group_sums = [0] * k
    
    # 按照 item 值大小从大到小排序，尽量先分配较大的数
    sorted_values = sorted(values, reverse=True)
    
    for value in sorted_values:
        # 找到当前总和最小的组，将当前 value 分配到该组
        min_group_index = group_sums.index(min(group_sums))
        groups[min_group_index].append(value)
        group_sums[min_group_index] += value
    
    return group_sums[::-1]



def balanced_load(
    model_dir,
    num_devices=device_count(),
    is_distillation=False,
    ratio=None,
    devices_idx=None,
    encoder_decoder=False,
    encoder_only=False,
):
    """_summary_

    Args:
        model_dir (str): can be local path or hf identifier.
        num_devices (int, optional): the num of devices you want to use for loading model . Defaults to all cuda device.
        is_distillation (bool, optional): _description_. Defaults to False.
        ratio (list[float], optional): each device load how much of the model, like [0.8,1] means the ratio between two device.
        devices_idx (list[int], optional): _description_. the device you want to use, like [2,4] means "cuda:2" and "cuda:4".
        encoder_decoder (bool, optional): the model you want to load is encoder-decoder model or not? . Defaults to False.

    Returns:
        _type_: _description_
    """

    if ratio is not None:
        assert len(ratio) == num_devices, "len(ratio) should equal to num_devices"
        if sum(ratio) != 1:
            ratio = [d / sum(ratio) for d in ratio]

    from collections import OrderedDict
    import math
    from transformers import AutoModelForCausalLM
    import accelerate

    with accelerate.init_empty_weights():
        if encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        elif encoder_only:
            model = AutoModel.from_pretrained(
                model_dir,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype="auto",
                trust_remote_code=True,
            )
    print(model)
    
    # model._no_split_modules  model._tied_weights_keys 不用前者是因为有些模型没有这个属性，不用后者是因为有些模型这个属性只写了个lm head，不如我自己根据config去tie
    devices_idx = list(range(num_devices)) if devices_idx is None else devices_idx
    assert (
        len(devices_idx) == num_devices
    ), "len(index of allocated device) should equal to num_devices"

    def create_manual_device_map(
        model,
        num_devices,
        is_distillation=False,
        ratio=ratio,
        devices_idx=devices_idx,
        encoder_decoder=encoder_decoder,
    ):
        device_map = {}

        params_cnt = {}  # 用于存放除了layers以外，其他模块的参数量总和

        
        
        # 计算每个模块的参数量
        if not encoder_only:
            lm_head = model_type2lm_head_modules[model.config.model_type](model)

            lm_head_params = sum(p.numel() for p in lm_head.parameters())
            if not model.config.tie_word_embeddings:
                if encoder_decoder:
                    lm_head_params *= 3  # ed模型有三个embedding
                else:
                    lm_head_params *= 2
            params_cnt[model_type2lm_head_names[model.config.model_type]] = (
                lm_head_params
            )
        else:
            lm_head = model.embeddings
            lm_head_params = sum(p.numel() for p in lm_head.parameters())
            params_cnt["embeddings"] = lm_head_params

        if encoder_decoder:
            params_cnt["model.encoder.embed_positions"] = sum(
                p.numel() for p in model.model.encoder.embed_positions.parameters()
            )
            params_cnt["model.decoder.embed_positions"] = sum(
                p.numel() for p in model.model.encoder.embed_positions.parameters()
            )
            if hasattr(model, "final_logits_bias"):
                final_logits_bias_params = model.final_logits_bias.numel()

                params_cnt["final_logits_bias"] = final_logits_bias_params

        if hasattr(model, "model"):  # encoder only模型我从AutoModel导入，所以少一节。
            if hasattr(model.model, "norm"):
                norm_params = sum(p.numel() for p in model.model.norm.parameters())
                params_cnt["model.norm"] = norm_params

            if hasattr(model.model, "rotary_emb"):
                rotary_emb_params = sum(
                    p.numel() for p in model.model.rotary_emb.parameters()
                )
                params_cnt["model.rotary_emb"] = rotary_emb_params

            if hasattr(model.model, "shared"):
                shared_emb_params = sum(
                    p.numel() for p in model.model.shared.parameters()
                )
                params_cnt["model.shared"] = 0
            if encoder_decoder:
                if hasattr(model.model.encoder, "layernorm_embedding"):
                    layernorm_embedding = sum(
                        p.numel()
                        for p in model.model.encoder.layernorm_embedding.parameters()
                    )
                    params_cnt["model.encoder.layernorm_embedding"] = (
                        layernorm_embedding
                    )
                if hasattr(model.model.decoder, "layernorm_embedding"):
                    layernorm_embedding = sum(
                        p.numel()
                        for p in model.model.decoder.layernorm_embedding.parameters()
                    )
                    params_cnt["model.decoder.layernorm_embedding"] = (
                        layernorm_embedding
                    )
        if encoder_decoder:
            encoder_layer_params = sum(
                p.numel() for p in model.model.encoder.layers[0].parameters()
            )
            # 没办法，decoder还有个ca呢,而且有时候ffn_dim也不同，比如宽窄模型
            decoder_layer_params = sum(
                p.numel() for p in model.model.decoder.layers[0].parameters()
            )

            # NOTE 我认为细粒度一点的可能会让层分配的更均匀，因为我下面设的是四舍五入
            layer_params = min(encoder_layer_params, decoder_layer_params)
        elif encoder_only:
            layer_params = sum(p.numel() for p in model.encoder.layer[0].parameters())
        else:
            layer_params = sum(p.numel() for p in model.model.layers[0].parameters())

        # 计算每个模块等效的层数
        params_ratio = {}
        for i, item in enumerate(params_cnt.items()):
            key, value = item
            params_ratio[key] = round(value / layer_params)

        total_layers = 0
        if encoder_decoder:
            if encoder_layer_params < decoder_layer_params:
                total_layers += model.config.encoder_layers
                total_layers += (
                    round(encoder_layer_params / decoder_layer_params)
                    * model.config.decoder_layers
                )
            else:
                total_layers += model.config.decoder_layers
                total_layers += (
                    round(encoder_layer_params / decoder_layer_params)
                    * model.config.encoder_layers
                )

        else:  # both encoder/decoder only
            total_layers += model.config.num_hidden_layers

        total_layers += sum(d for d in params_ratio.values())

        if ratio is not None:
            layers_per_device = [round(r * total_layers) for r in ratio]
        else:
            layers_per_device = [
                total_layers // num_devices for _ in range(num_devices)
            ]

        remainder = total_layers - sum(layers_per_device)

        # 从后面开始分配剩余层
        start_index = num_devices - 1
        while remainder != 0:
            layers_per_device[start_index] += 1
            remainder -= 1
            start_index = start_index - 1 % num_devices
        for i in range(remainder - 1, -1, -1):
            layers_per_device[i] += 1

        
        
        layers = {}
        if encoder_decoder:

            for i in range(model.config.encoder_layers):
                params_ratio[f"model.encoder.layers.{i}"] = (
                    1
                    if encoder_layer_params <= decoder_layer_params
                    else round(encoder_layer_params / decoder_layer_params)
                )

            for i in range(model.config.encoder_layers):
                params_ratio[f"model.decoder.layers.{i}"] = (
                    1
                    if encoder_layer_params >= decoder_layer_params
                    else round(decoder_layer_params / encoder_layer_params)
                )
        elif encoder_only:
            for i in range(model.config.num_hidden_layers):
                params_ratio[f"encoder.layer.{i}"] = 1
        else:

            for i in range(model.config.num_hidden_layers):
                params_ratio[f"model.layers.{i}"] = 1

        
        
        # encoder only模型把word embedding和position embedding、type embedding打包在一起了
        # 位置编码和输入层必须放在一起，不然会马上报错.
        must_allocate_burden = 0
        must_allocate_keys = []
        for layer_name, layer_burden in params_ratio.items():
            if "emb" in layer_name or "shared" in layer_name:
                must_allocate_burden += layer_burden
                must_allocate_keys.append(layer_name)

        # 上面的代码用来处理纯粹的位置编码，然后下面这里用来处理lm head
        if not encoder_only:
            must_allocate_keys.append(model_type2lm_head_names[model.config.model_type])
            must_allocate_burden += params_ratio[
                model_type2lm_head_names[model.config.model_type]
            ]
        # temp_max_burden = 0
        # temp_max_idx = len(layers_per_device) - 1
        # for idx, burden in enumerate(layers_per_device):
        #     if temp_max_burden <= burden:  # 等于可以尽可能分配到后面去
        #         temp_max_burden = burden
        #         temp_max_idx = idx

        for keys in must_allocate_keys:
            # device_map[keys] = devices_idx[temp_max_idx]

            del params_ratio[keys]

        params_ratio["must_together"]=must_allocate_burden
        
        layers_per_device=balanced_partition(params_ratio,num_devices)
        print(layers_per_device)


        # BUG 下面的两种分配过程存在一种corner case，就是任何一层都放不下了，会导致current device找不到合法值越界。
        # 但这个现象非常难以出现。因为大模型里最大的层通常就是embedding层。而embedding层在上面已经被我们分配完了。
        # 目前看最大的可能是encoder layer和decoder layer差距很多容易导致这个。

        # 开始分配特殊层
        for layer_name, layer_burden in params_ratio.items():

            current_device = 0
            while (
                current_device < num_devices
                and layers_per_device[current_device] - layer_burden < 0
            ):
                current_device += 1

            device_map[layer_name] = devices_idx[current_device]
            layers_per_device[current_device] -= layer_burden

        

        # 先分配普通层，因为他们数量很多，保证他们尽可能在一个device上会加快推理速度
        # for layer_name, layer_burden in layers.items():
        #     current_device = 0
        #     while (
        #         current_device < num_devices
        #         and layers_per_device[current_device] - layer_burden < 0
        #     ):
        #         current_device += 1
        #     device_map[layer_name] = devices_idx[current_device]
        #     layers_per_device[current_device] -= layer_burden

        for keys in must_allocate_keys:
            device_map[keys]=device_map["must_together"]
        
        del device_map["must_together"]
    
        if encoder_decoder:
            # 考虑到tied weights，他们需要在一起，而且我已经把他们的重量认定给lm_head了
            device_map["model.encoder.embed_tokens"] = device_map[
                model_type2lm_head_names[model.config.model_type]
            ]
            device_map["model.decoder.embed_tokens"] = device_map[
                model_type2lm_head_names[model.config.model_type]
            ]
        elif encoder_only:
            pass
        else:
            device_map["model.embed_tokens"] = device_map[
                model_type2lm_head_names[model.config.model_type]
            ]


        return device_map

    # 使用手动创建的 device_map
    device_map = create_manual_device_map(model, num_devices, is_distillation)

    # 打印 device_map 结果
    # 打印 device_map 结果和每个设备上的元素统计
    device_stats = {}
    for module, device in device_map.items():
        if device not in device_stats:
            device_stats[device] = {"count": 0, "modules": []}
        device_stats[device]["count"] += 1
        device_stats[device]["modules"].append(module)

    print("Device Map:")
    for device, stats in device_stats.items():
        print(f"Device {device}: {stats['count']} elements")
        print(f"  Modules: {', '.join(stats['modules'])}")

    del model
    if encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage="True",
            trust_remote_code=True,
        )
    elif encoder_only:
        model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage="True",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage="True",
            attn_implementation=(
                "eager"
                if "gemma" in model_dir.lower() or "phi" in model_dir.lower()
                else "sdpa"
            ),
            trust_remote_code=True,
        )

    # 更完整的提示，且按照module风格
    # def print_module_devices(module, indent=''):
    #     for name, child in module.named_children():
    #         if list(child.children()):
    #             print(f"{indent}{name}:")
    #             print_module_devices(child, indent + '  ')
    #         else:
    #             try:
    #                 device = next(child.parameters()).device
    #             except StopIteration:
    #                 device = "N/A (no parameters)"
    #             print(f"{indent}{name}: {device}")

    # print_module_devices(model)
    return model



if __name__=="__main__":
    # model=balanced_load("/mnt/rangehow/models/gte-multilingual-base",encoder_only=True)
    model=balanced_load("/mnt/rangehow/models/gemma-2b")
    print(model(torch.tensor([[1,2,3]],device=model.device)))
    
    # balanced_load("/mnt/rangehow/models/Meta-Llama-3.1-8B-Instruct")