# TorchAdapter 和 MegatronAdapter 接口说明文档

本文档详细说明了 `TorchAdapter` 和 `MegatronAdapter` 类中所有接口的使用方法、参数和返回值。

## 目录

- [TorchAdapter 接口](#torchadapter-接口)
  - [基础操作](#基础操作)
  - [分布式通信](#分布式通信)
  - [神经网络模块](#神经网络模块)
  - [神经网络函数](#神经网络函数)
  - [自动微分](#自动微分)
  - [CUDA 相关](#cuda-相关)
  - [NPU 相关](#npu-相关)
  - [设备网格](#设备网格)
  - [设备管理](#设备管理)
  - [模型方法](#模型方法)
- [MegatronAdapter 接口](#megatronadapter-接口)
  - [并行状态](#并行状态)
  - [优化器](#优化器)
  - [流水线并行](#流水线并行)
  - [分布式](#分布式)
  - [张量并行](#张量并行)
  - [模型](#模型)
  - [工具函数](#工具函数)
  - [优化器工具](#优化器工具)
  - [张量并行工具](#张量并行工具)
  - [流水线并行工具](#流水线并行工具)
  - [模型权重加载](#模型权重加载)
  - [模块构建](#模块构建)
  - [检查点](#检查点)

---

## TorchAdapter 接口

### 基础操作

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `manual_seed(seed: int)` | `seed`: 随机种子 | `None` | 设置随机数生成器的种子，对应 `torch.manual_seed(seed)` |
| `float16()` | 无 | `torch.dtype` | 返回 float16 数据类型，对应 `torch.float16` |
| `bfloat16()` | 无 | `torch.dtype` | 返回 bfloat16 数据类型，对应 `torch.bfloat16` |
| `float32()` | 无 | `torch.dtype` | 返回 float32 数据类型，对应 `torch.float32` |
| `int64()` | 无 | `torch.dtype` | 返回 int64 数据类型，对应 `torch.int64` |
| `long()` | 无 | `torch.dtype` | 返回 long 数据类型，对应 `torch.long` |
| `Tensor()` | 无 | `torch.Tensor` (类) | 返回 Tensor 类，对应 `torch.Tensor` |
| `tensor(data, dtype=None, device=None)` | `data`: 数据<br>`dtype`: 数据类型（可选）<br>`device`: 设备（可选） | `torch.Tensor` | 从数据创建张量，对应 `torch.tensor()` |
| `empty(size, dtype=None, device=None)` | `size`: 张量大小<br>`dtype`: 数据类型（可选）<br>`device`: 设备（可选） | `torch.Tensor` | 创建未初始化的张量，对应 `torch.empty()` |
| `cat(tensors, dim=0)` | `tensors`: 张量序列<br>`dim`: 拼接维度，默认 0 | `torch.Tensor` | 沿指定维度拼接张量，对应 `torch.cat()` |
| `no_grad()` | 无 | 上下文管理器 | 禁用梯度计算的上下文管理器，对应 `torch.no_grad()` |
| `empty_like(input, dtype=None, device=None)` | `input`: 输入张量<br>`dtype`: 数据类型（可选）<br>`device`: 设备（可选） | `torch.Tensor` | 创建与输入张量相同形状的未初始化张量，对应 `torch.empty_like()` |
| `Size(*args)` | `*args`: 尺寸参数 | `torch.Size` | 创建 Size 对象，对应 `torch.Size()` |
| `squeeze(input, dim=None)` | `input`: 输入张量<br>`dim`: 要压缩的维度（可选） | `torch.Tensor` | 移除大小为 1 的维度，对应 `torch.squeeze()` |
| `clip(input, min=None, max=None)` | `input`: 输入张量<br>`min`: 最小值（可选）<br>`max`: 最大值（可选） | `torch.Tensor` | 将张量值限制在指定范围内，对应 `torch.clip()` |
| `cumsum(input, dim, dtype=None)` | `input`: 输入张量<br>`dim`: 累积求和的维度<br>`dtype`: 数据类型（可选） | `torch.Tensor` | 计算累积和，对应 `torch.cumsum()` |
| `device(device)` | `device`: 设备字符串或设备对象 | `torch.device` | 创建设备对象，对应 `torch.device()` |
| `roll(input, shifts, dims)` | `input`: 输入张量<br>`shifts`: 移动量<br>`dims`: 移动的维度 | `torch.Tensor` | 沿指定维度滚动张量，对应 `torch.roll()` |

### 分布式通信

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `distributed_is_initialized()` | 无 | `bool` | 检查分布式环境是否已初始化，对应 `torch.distributed.is_initialized()` |
| `distributed_init_process_group(backend=None, timeout=None, init_method=None)` | `backend`: 后端（可选）<br>`timeout`: 超时时间（可选）<br>`init_method`: 初始化方法（可选） | `None` | 初始化进程组，对应 `torch.distributed.init_process_group()` |
| `distributed_barrier()` | 无 | `None` | 同步所有进程，对应 `torch.distributed.barrier()` |
| `distributed_broadcast(tensor, src, group=None, async_op=False)` | `tensor`: 要广播的张量<br>`src`: 源进程 rank<br>`group`: 进程组（可选）<br>`async_op`: 是否异步（可选） | `None` | 从源进程广播张量到所有进程，对应 `torch.distributed.broadcast()` |
| `distributed_get_rank()` | 无 | `int` | 获取当前进程的 rank，对应 `torch.distributed.get_rank()` |
| `distributed_all_gather_object(object_list, obj, group=None)` | `object_list`: 输出对象列表<br>`obj`: 要收集的对象<br>`group`: 进程组（可选） | `None` | 从所有进程收集对象，对应 `torch.distributed.all_gather_object()` |
| `distributed_broadcast_object_list(object_list, src, group=None)` | `object_list`: 对象列表<br>`src`: 源进程 rank<br>`group`: 进程组（可选） | `None` | 从源进程广播对象列表，对应 `torch.distributed.broadcast_object_list()` |
| `distributed_all_gather(tensor_list, tensor, group=None, async_op=False)` | `tensor_list`: 输出张量列表<br>`tensor`: 要收集的张量<br>`group`: 进程组（可选）<br>`async_op`: 是否异步（可选） | `None` | 从所有进程收集张量，对应 `torch.distributed.all_gather()` |
| `distributed_all_reduce(tensor, op=None, group=None, async_op=False)` | `tensor`: 输入/输出张量<br>`op`: 归约操作（可选）<br>`group`: 进程组（可选）<br>`async_op`: 是否异步（可选） | `None` | 对所有进程的张量进行归约操作，对应 `torch.distributed.all_reduce()` |
| `distributed_get_global_rank(group, group_rank)` | `group`: 进程组<br>`group_rank`: 组内 rank | `int` | 获取全局 rank，对应 `torch.distributed.get_global_rank()` |
| `distributed_get_world_size(group=None)` | `group`: 进程组（可选） | `int` | 获取进程组大小，对应 `torch.distributed.get_world_size()` |
| `distributed_ReduceOp()` | 无 | `torch.distributed.ReduceOp` (类) | 返回 ReduceOp 类，对应 `torch.distributed.ReduceOp` |

### 神经网络模块

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `nn_ModuleList(*args)` | `*args`: 模块参数 | `nn.ModuleList` | 创建模块列表，对应 `nn.ModuleList` |
| `nn_Module(*args, **kwargs)` | `*args, **kwargs`: 模块参数 | `nn.Module` | 创建基础模块，对应 `nn.Module` |
| `nn_Linear(in_features, out_features, bias=True, dtype=None)` | `in_features`: 输入特征数<br>`out_features`: 输出特征数<br>`bias`: 是否使用偏置，默认 True<br>`dtype`: 数据类型（可选） | `nn.Linear` | 创建线性层，对应 `nn.Linear` |
| `nn_Sequential(*args)` | `*args`: 模块参数 | `nn.Sequential` | 创建顺序模块，对应 `nn.Sequential` |

### 神经网络函数

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `F_silu(input)` | `input`: 输入张量 | `torch.Tensor` | SiLU 激活函数，对应 `F.silu()` |
| `F_pad(input, pad, mode="constant", value=0)` | `input`: 输入张量<br>`pad`: 填充大小<br>`mode`: 填充模式，默认 "constant"<br>`value`: 填充值，默认 0 | `torch.Tensor` | 填充张量，对应 `F.pad()` |

### 自动微分

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `autograd_Function()` | 无 | `torch.autograd.Function` (类) | 返回 Function 基类，对应 `torch.autograd.Function` |
| `compile(*args, **kwargs)` | `*args, **kwargs`: 编译参数 | 编译后的函数 | 编译函数以优化性能，对应 `torch.compile()` |

### CUDA 相关

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `cuda_is_available()` | 无 | `bool` | 检查 CUDA 是否可用，对应 `torch.cuda.is_available()` |
| `cuda_memory_set_allocator_settings(settings)` | `settings`: 分配器设置字符串 | `None` | 设置 CUDA 内存分配器设置，对应 `torch.cuda.memory._set_allocator_settings()` |

### NPU 相关

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `npu_is_available()` | 无 | `bool` | 检查 NPU 是否可用，对应 `torch.npu.is_available()`（如果可用） |

### 设备网格

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `init_device_mesh(device_type, mesh_shape, mesh_dim_names)` | `device_type`: 设备类型<br>`mesh_shape`: 网格形状<br>`mesh_dim_names`: 维度名称 | `DeviceMesh` | 初始化设备网格，对应 `torch.distributed.device_mesh.init_device_mesh()` |

### 设备管理

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `get_torch_device()` | 无 | 设备对象 | 获取 torch 设备对象，对应 `get_torch_device()` |
| `device_count()` | 无 | `int` | 获取设备数量，对应 `get_torch_device().device_count()` |
| `set_device(rank)` | `rank`: 设备 rank | `None` | 设置当前设备，对应 `get_torch_device().set_device()` |
| `get_rng_state()` | 无 | 随机数状态 | 获取随机数生成器状态，对应 `get_torch_device().get_rng_state()` |
| `set_rng_state(state)` | `state`: 随机数状态 | `None` | 设置随机数生成器状态，对应 `get_torch_device().set_rng_state()` |
| `manual_seed_device(seed)` | `seed`: 随机种子 | `None` | 在设备上设置随机种子，对应 `get_torch_device().manual_seed()` |
| `empty_cache()` | 无 | `None` | 清空设备缓存，对应 `get_torch_device().empty_cache()` |
| `max_memory_allocated()` | 无 | `int` | 获取最大已分配内存（字节），对应 `get_torch_device().max_memory_allocated()` |
| `max_memory_reserved()` | 无 | `int` | 获取最大保留内存（字节），对应 `get_torch_device().max_memory_reserved()` |

### 模型方法

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `model_train(model)` | `model`: 模型对象 | `None` | 将模型设置为训练模式，对应 `model.train()` |
| `zero_grad_buffer(model)` | `model`: 模型对象 | `None` | 清零梯度缓冲区，对应 `model.zero_grad_buffer()` |

---

## MegatronAdapter 接口

### 并行状态

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, use_sharp=False, context_parallel_size=1, expert_model_parallel_size=1, expert_tensor_parallel_size=1, nccl_communicator_config_path=None)` | `tensor_model_parallel_size`: 张量并行大小，默认 1<br>`pipeline_model_parallel_size`: 流水线并行大小，默认 1<br>`virtual_pipeline_model_parallel_size`: 虚拟流水线并行大小（可选）<br>`use_sharp`: 是否使用 SHARP，默认 False<br>`context_parallel_size`: 上下文并行大小，默认 1<br>`expert_model_parallel_size`: 专家模型并行大小，默认 1<br>`expert_tensor_parallel_size`: 专家张量并行大小，默认 1<br>`nccl_communicator_config_path`: NCCL 通信器配置路径（可选） | `None` | 初始化模型并行，对应 `mpu.initialize_model_parallel()` |
| `get_tensor_model_parallel_rank()` | 无 | `int` | 获取张量模型并行 rank，对应 `mpu.get_tensor_model_parallel_rank()` |
| `get_pipeline_model_parallel_rank()` | 无 | `int` | 获取流水线模型并行 rank，对应 `mpu.get_pipeline_model_parallel_rank()` |
| `get_pipeline_model_parallel_world_size()` | 无 | `int` | 获取流水线模型并行世界大小，对应 `mpu.get_pipeline_model_parallel_world_size()` |
| `get_pipeline_model_parallel_last_rank()` | 无 | `int` | 获取流水线模型并行的最后一个 rank，对应 `mpu.get_pipeline_model_parallel_last_rank()` |
| `get_pipeline_model_parallel_group()` | 无 | 进程组 | 获取流水线模型并行进程组，对应 `mpu.get_pipeline_model_parallel_group()` |
| `get_data_parallel_rank()` | 无 | `int` | 获取数据并行 rank，对应 `mpu.get_data_parallel_rank()` |
| `get_data_parallel_world_size()` | 无 | `int` | 获取数据并行世界大小，对应 `mpu.get_data_parallel_world_size()` |
| `get_context_parallel_rank()` | 无 | `int` | 获取上下文并行 rank，对应 `mpu.get_context_parallel_rank()` |
| `get_virtual_pipeline_model_parallel_world_size()` | 无 | `int` | 获取虚拟流水线模型并行世界大小，对应 `mpu.get_virtual_pipeline_model_parallel_world_size()` |
| `is_pipeline_last_stage(ignore_virtual=False)` | `ignore_virtual`: 是否忽略虚拟流水线，默认 False | `bool` | 检查是否为流水线最后阶段，对应 `mpu.is_pipeline_last_stage()` |
| `get_tensor_model_parallel_world_size()` | 无 | `int` | 获取张量模型并行世界大小，对应 `mpu.get_tensor_model_parallel_world_size()` |
| `get_context_parallel_world_size()` | 无 | `int` | 获取上下文并行世界大小，对应 `mpu.get_context_parallel_world_size()` |
| `get_tensor_model_parallel_group()` | 无 | 进程组 | 获取张量模型并行进程组，对应 `mpu.get_tensor_model_parallel_group()` |
| `get_expert_model_parallel_world_size()` | 无 | `int` | 获取专家模型并行世界大小，对应 `mpu.get_expert_model_parallel_world_size()` |
| `get_expert_tensor_parallel_world_size()` | 无 | `int` | 获取专家张量并行世界大小，对应 `mpu.get_expert_tensor_parallel_world_size()` |
| `get_expert_model_parallel_group()` | 无 | 进程组 | 获取专家模型并行进程组，对应 `mpu.get_expert_model_parallel_group()` |
| `get_expert_tensor_parallel_group()` | 无 | 进程组 | 获取专家张量并行进程组，对应 `mpu.get_expert_tensor_parallel_group()` |
| `set_virtual_pipeline_model_parallel_rank(rank)` | `rank`: 虚拟流水线 rank | `None` | 设置虚拟流水线模型并行 rank，对应 `mpu.set_virtual_pipeline_model_parallel_rank()` |
| `is_pipeline_first_stage(ignore_virtual=False, vp_stage=None)` | `ignore_virtual`: 是否忽略虚拟流水线，默认 False<br>`vp_stage`: 虚拟流水线阶段（可选） | `bool` | 检查是否为流水线第一阶段，对应 `mpu.is_pipeline_first_stage()` |
| `get_pipeline_model_parallel_split_rank()` | 无 | `int` | 获取流水线模型并行分割 rank，对应 `mpu.get_pipeline_model_parallel_split_rank()` |
| `is_pipeline_stage_before_split()` | 无 | `bool` | 检查是否为分割前的流水线阶段，对应 `mpu.is_pipeline_stage_before_split()` |
| `is_pipeline_stage_after_split()` | 无 | `bool` | 检查是否为分割后的流水线阶段，对应 `mpu.is_pipeline_stage_after_split()` |

### 优化器

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `DistributedOptimizer(*args, **kwargs)` | `*args, **kwargs`: 优化器参数 | `DistributedOptimizer` | 创建分布式优化器，对应 `megatron.core.optimizer.DistributedOptimizer` |

### 流水线并行

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `get_forward_backward_func()` | 无 | 函数 | 获取前向反向传播函数，对应 `megatron.core.pipeline_parallel.get_forward_backward_func()` |

### 分布式

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `finalize_model_grads(*args, **kwargs)` | `*args, **kwargs`: 参数 | 返回值取决于实现 | 完成模型梯度处理，对应 `megatron.core.distributed.finalize_model_grads()` |

### 张量并行

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `model_parallel_cuda_manual_seed(seed)` | `seed`: 随机种子 | `None` | 在模型并行 CUDA 上设置随机种子，对应 `tensor_parallel.model_parallel_cuda_manual_seed()` |
| `set_defaults_if_not_set_tensor_model_parallel_attributes(param)` | `param`: 参数对象 | `None` | 如果未设置，则设置张量模型并行属性的默认值，对应 `tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes()` |
| `vocab_parallel_cross_entropy(vocab_parallel_logits, target)` | `vocab_parallel_logits`: 词汇并行 logits<br>`target`: 目标标签 | `torch.Tensor` | 计算词汇并行交叉熵，对应 `tensor_parallel.vocab_parallel_cross_entropy()` |

### 模型

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `GPTModel(*args, **kwargs)` | `*args, **kwargs`: 模型参数 | `GPTModel` | 创建 GPT 模型，对应 `megatron.core.models.gpt.gpt_model.GPTModel` |

### 工具函数

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `load_megatron_model_to_gpu(model, load_grad=False)` | `model`: 模型对象<br>`load_grad`: 是否加载梯度，默认 False | 返回值取决于实现 | 将 Megatron 模型加载到 GPU，对应 `verl.utils.megatron_utils.load_megatron_model_to_gpu()` |
| `load_megatron_optimizer(optimizer)` | `optimizer`: 优化器对象 | 返回值取决于实现 | 加载 Megatron 优化器，对应 `verl.utils.megatron_utils.load_megatron_optimizer()` |
| `offload_megatron_model_to_cpu(model)` | `model`: 模型对象 | 返回值取决于实现 | 将 Megatron 模型卸载到 CPU，对应 `verl.utils.megatron_utils.offload_megatron_model_to_cpu()` |
| `offload_megatron_optimizer(optimizer)` | `optimizer`: 优化器对象 | 返回值取决于实现 | 将 Megatron 优化器卸载到 CPU，对应 `verl.utils.megatron_utils.offload_megatron_optimizer()` |
| `per_tensor_generator(module, model_config, weight_converter, tf_config, layer_name_mapping)` | `module`: 模块对象<br>`model_config`: 模型配置<br>`weight_converter`: 权重转换器<br>`tf_config`: Transformer 配置<br>`layer_name_mapping`: 层名称映射 | 生成器 | 生成每个张量，对应 `verl.utils.megatron_utils.per_tensor_generator()` |
| `register_megatron_training_hooks(module, optimizer)` | `module`: 模块对象<br>`optimizer`: 优化器对象 | 返回值取决于实现 | 注册 Megatron 训练钩子，对应 `verl.utils.megatron_utils.register_megatron_training_hooks()` |
| `get_model_config(module)` | `module`: 模块对象 | 配置对象 | 获取模型配置，对应 `verl.utils.megatron_utils.get_model_config()` |

### 优化器工具

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `get_megatron_optimizer(model, config)` | `model`: 模型对象<br>`config`: 配置对象 | 优化器对象 | 获取 Megatron 优化器，对应 `verl.utils.megatron.optimizer.get_megatron_optimizer()` |
| `get_megatron_optimizer_param_scheduler(optimizer, config)` | `optimizer`: 优化器对象<br>`config`: 配置对象 | 调度器对象 | 获取 Megatron 优化器参数调度器，对应 `verl.utils.megatron.optimizer.get_megatron_optimizer_param_scheduler()` |
| `init_megatron_optim_config(optim_config, fp16=False)` | `optim_config`: 优化配置<br>`fp16`: 是否使用 FP16，默认 False | 配置对象 | 初始化 Megatron 优化配置，对应 `verl.utils.megatron.optimizer.init_megatron_optim_config()` |
| `get_megatron_last_lr(optimizer)` | `optimizer`: 优化器对象 | `float` | 获取 Megatron 最后的学习率，对应 `verl.utils.megatron.optimizer.get_megatron_last_lr()` |

### 张量并行工具

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `vocab_parallel_entropy(logits)` | `logits`: logits 张量 | `torch.Tensor` | 计算词汇并行熵，对应 `verl.utils.megatron.tensor_parallel.vocab_parallel_entropy()` |
| `vocab_parallel_log_probs_from_logits(logits, label)` | `logits`: logits 张量<br>`label`: 标签张量 | `torch.Tensor` | 从 logits 计算词汇并行对数概率，对应 `verl.utils.megatron.tensor_parallel.vocab_parallel_log_probs_from_logits()` |

### 流水线并行工具

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `make_batch_generator(micro_batches, vpp_size)` | `micro_batches`: 微批次<br>`vpp_size`: 虚拟流水线并行大小 | 生成器或生成器列表 | 创建批次生成器，对应 `verl.utils.megatron.pipeline_parallel.make_batch_generator()` |

### 模型权重加载

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `load_mcore_dist_weights(module, dist_checkpointing_path, is_value_model=False)` | `module`: 模块对象<br>`dist_checkpointing_path`: 分布式检查点路径<br>`is_value_model`: 是否为价值模型，默认 False | 返回值取决于实现 | 加载 mcore 分布式权重，对应 `verl.utils.model.load_mcore_dist_weights()` |
| `load_megatron_gptmodel_weights(config, hf_config, module, params_dtype, is_value_model=False)` | `config`: 配置对象<br>`hf_config`: HuggingFace 配置<br>`module`: 模块对象<br>`params_dtype`: 参数数据类型<br>`is_value_model`: 是否为价值模型，默认 False | 返回值取决于实现 | 加载 Megatron GPT 模型权重，对应 `verl.utils.model.load_megatron_gptmodel_weights()` |

### 模块构建

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `McoreModuleWrapperConfig(*args, **kwargs)` | `*args, **kwargs`: 配置参数 | 配置对象 | 创建 Mcore 模块包装器配置，对应 `verl.utils.megatron_utils.McoreModuleWrapperConfig` |
| `make_megatron_module(wrap_config, tf_config, hf_config, bridge=None, override_model_config=None, override_ddp_config=None)` | `wrap_config`: 包装配置<br>`tf_config`: Transformer 配置<br>`hf_config`: HuggingFace 配置<br>`bridge`: 桥接对象（可选）<br>`override_model_config`: 覆盖模型配置（可选）<br>`override_ddp_config`: 覆盖 DDP 配置（可选） | 模块对象 | 创建 Megatron 模块，对应 `verl.utils.megatron_utils.make_megatron_module()` |

### 分布式数据并行

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `DistributedDataParallel(*args, **kwargs)` | `*args, **kwargs`: DDP 参数 | `DistributedDataParallel` | 创建分布式数据并行包装器，对应 `megatron.core.distributed.DistributedDataParallel` |
| `DistributedDataParallelConfig(*args, **kwargs)` | `*args, **kwargs`: 配置参数 | 配置对象 | 创建分布式数据并行配置，对应 `megatron.core.distributed.DistributedDataParallelConfig` |

### Transformer

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `Float16Module(*args, **kwargs)` | `*args, **kwargs`: 模块参数 | `Float16Module` | 创建 Float16 模块，对应 `megatron.core.transformer.module.Float16Module` |
| `TransformerConfig(*args, **kwargs)` | `*args, **kwargs`: 配置参数 | 配置对象 | 创建 Transformer 配置，对应 `megatron.core.transformer.TransformerConfig` |

### 核心配置

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `ModelParallelConfig(*args, **kwargs)` | `*args, **kwargs`: 配置参数 | 配置对象 | 创建模型并行配置，对应 `megatron.core.ModelParallelConfig` |
| `ModelType()` | 无 | `ModelType` (枚举类) | 返回模型类型枚举，对应 `megatron.core.enums.ModelType` |
| `ChainedOptimizer(*args, **kwargs)` | `*args, **kwargs`: 优化器参数 | `ChainedOptimizer` | 创建链式优化器，对应 `megatron.core.optimizer.ChainedOptimizer` |
| `OptimizerConfig(*args, **kwargs)` | `*args, **kwargs`: 配置参数 | 配置对象 | 创建优化器配置，对应 `megatron.core.optimizer.OptimizerConfig` |

### 工具函数

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `get_attr_wrapped_model(model, attr_name, allow_none=False)` | `model`: 模型对象<br>`attr_name`: 属性名称<br>`allow_none`: 是否允许 None，默认 False | 属性值 | 获取包装模型的属性，对应 `megatron.core.utils.get_attr_wrapped_model()` |

### 检查点

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `load_checkpoint(sharded_state_dict, checkpoint_path, strict=None)` | `sharded_state_dict`: 分片状态字典<br>`checkpoint_path`: 检查点路径<br>`strict`: 严格模式（可选） | `None` | 加载分布式检查点，对应 `megatron.core.dist_checkpointing.load()` |

---

## 使用示例

### TorchAdapter 使用示例

```python
from verl.utils.megatron_adapter import TorchAdapter

# 基础操作
dtype = TorchAdapter.float16()
tensor = TorchAdapter.tensor([1, 2, 3], dtype=dtype)
result = TorchAdapter.cat([tensor, tensor], dim=0)

# 分布式通信
if not TorchAdapter.distributed_is_initialized():
    TorchAdapter.distributed_init_process_group(backend="nccl")
rank = TorchAdapter.distributed_get_rank()

# 设备管理
TorchAdapter.set_device(0)
TorchAdapter.empty_cache()
```

### MegatronAdapter 使用示例

```python
from verl.utils.megatron_adapter import MegatronAdapter

# 初始化模型并行
MegatronAdapter.initialize_model_parallel(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4
)

# 获取并行信息
tp_rank = MegatronAdapter.get_tensor_model_parallel_rank()
pp_rank = MegatronAdapter.get_pipeline_model_parallel_rank()

# 创建模型
model = MegatronAdapter.GPTModel(...)

# 加载权重
MegatronAdapter.load_checkpoint(state_dict, checkpoint_path)
```

---

## 注意事项

1. **所有方法都是静态方法**：`TorchAdapter` 和 `MegatronAdapter` 中的所有方法都是静态方法，可以直接通过类名调用，无需实例化。

2. **接口封装**：这些适配器类封装了底层框架（PyTorch/Megatron）的接口，使得未来可以轻松替换为其他框架（如 MindSpore/MindFormers）。

3. **参数传递**：大部分接口的参数与原框架接口保持一致，可以直接参考原框架文档了解详细参数说明。

4. **返回值**：返回值类型与原框架接口保持一致，具体类型取决于实现。

---

## 版本信息

- 文档生成时间：2024年
- 适配器版本：基于当前代码库
- 支持的框架：PyTorch, Megatron-Core

