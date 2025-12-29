# Adapter Interfaces Documentation

本文档详细说明了 `FrameWorkAdapter` 和 `BackendAdapter` 两个基类的所有接口。

## FrameWorkAdapter 接口说明

`FrameWorkAdapter` 是框架适配器的基类，定义了所有 torch 相关接口的规范。

### 数据类型 (Data Types)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `float16()` | 无 | dtype | 返回 float16 数据类型 |
| `bfloat16()` | 无 | dtype | 返回 bfloat16 数据类型 |
| `float32()` | 无 | dtype | 返回 float32 数据类型 |
| `int64()` | 无 | dtype | 返回 int64 数据类型 |
| `long()` | 无 | dtype | 返回 long 数据类型 |
| `dtype()` | 无 | dtype | 返回 dtype 类型 |
| `PrecisionType()` | 无 | PrecisionType | 返回 PrecisionType 类（来自 verl.utils.torch_dtypes） |

### 张量创建 (Tensor Creation)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `Tensor()` | 无 | Tensor 类 | 返回 Tensor 类 |
| `tensor(data, dtype=None, device=None)` | data: 数据, dtype: 数据类型, device: 设备 | Tensor | 从数据创建张量 |
| `empty(size, dtype=None, device=None)` | size: 形状, dtype: 数据类型, device: 设备 | Tensor | 创建未初始化的张量 |
| `empty_like(input)` | input: 输入张量 | Tensor | 创建与输入相同形状的未初始化张量 |
| `zeros(size, dtype=None, device=None, requires_grad=False)` | size: 形状, dtype: 数据类型, device: 设备, requires_grad: 是否需要梯度 | Tensor | 创建全零张量 |
| `zeros_like(input, dtype=None, device=None, requires_grad=False)` | input: 输入张量, dtype: 数据类型, device: 设备, requires_grad: 是否需要梯度 | Tensor | 创建与输入相同形状的全零张量 |
| `ones_like(input, dtype=None, device=None, requires_grad=False)` | input: 输入张量, dtype: 数据类型, device: 设备, requires_grad: 是否需要梯度 | Tensor | 创建与输入相同形状的全一张量 |
| `arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False)` | start: 起始值, end: 结束值, step: 步长, dtype: 数据类型, device: 设备, requires_grad: 是否需要梯度 | Tensor | 创建等差数列张量 |
| `Size(*args)` | *args: 尺寸参数 | Size | 创建 Size 对象 |

### 张量操作 (Tensor Operations)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `cat(tensors, dim=0)` | tensors: 张量列表, dim: 拼接维度 | Tensor | 沿指定维度拼接张量 |
| `gather(input, dim, index)` | input: 输入张量, dim: 维度, index: 索引张量 | Tensor | 根据索引从输入张量中收集元素 |
| `cumsum(input, dim, dtype=None)` | input: 输入张量, dim: 维度, dtype: 数据类型 | Tensor | 计算累积和 |
| `nonzero(input, as_tuple=False)` | input: 输入张量, as_tuple: 是否返回元组 | Tensor/元组 | 返回非零元素的索引 |
| `argmax(input, dim=None, keepdim=False)` | input: 输入张量, dim: 维度, keepdim: 是否保持维度 | Tensor | 返回最大值的索引 |
| `roll(input, shifts, dims)` | input: 输入张量, shifts: 偏移量, dims: 维度 | Tensor | 沿指定维度滚动张量 |

### 神经网络模块 (Neural Network Modules)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `nn_ModuleList(*args, **kwargs)` | *args, **kwargs: 模块参数 | ModuleList | 创建模块列表 |
| `nn_functional_pad(input, pad, mode='constant', value=0)` | input: 输入张量, pad: 填充大小, mode: 填充模式, value: 填充值 | Tensor | 对张量进行填充 |
| `nn_functional_silu(input)` | input: 输入张量 | Tensor | 应用 SiLU 激活函数 |

### 自动微分 (Automatic Differentiation)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `no_grad()` | 无 | 上下文管理器 | 禁用梯度计算的上下文管理器 |
| `autograd_Function()` | 无 | Function 类 | 返回 autograd.Function 类 |

### 随机数生成 (Random Number Generation)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `manual_seed(seed: int)` | seed: 随机种子 | None | 设置全局随机种子 |
| `get_rng_state()` | 无 | 状态对象 | 获取设备随机数生成器状态 |
| `set_rng_state(state)` | state: 状态对象 | None | 设置设备随机数生成器状态 |
| `manual_seed_device(seed)` | seed: 随机种子 | None | 设置设备随机种子 |
| `get_rng_state_torch()` | 无 | 状态对象 | 获取 torch 随机数生成器状态 |
| `set_rng_state_torch(state)` | state: 状态对象 | None | 设置 torch 随机数生成器状态 |

### 设备管理 (Device Management)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `get_torch_device()` | 无 | 设备对象 | 获取 torch 设备对象 |
| `device_count()` | 无 | int | 获取可用设备数量 |
| `set_device(rank)` | rank: 设备编号 | None | 设置当前设备 |
| `current_device()` | 无 | int | 获取当前设备编号 |
| `cuda_is_available()` | 无 | bool | 检查 CUDA 是否可用 |
| `npu_is_available()` | 无 | bool | 检查 NPU 是否可用 |
| `get_device_module(device_name)` | device_name: 设备名称 | 设备模块 | 根据名称获取设备模块 |
| `cuda_memory_set_allocator_settings(settings)` | settings: 分配器设置 | None | 设置 CUDA 内存分配器 |

### 内存管理 (Memory Management)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `empty_cache()` | 无 | None | 清空设备缓存 |
| `max_memory_allocated()` | 无 | int | 获取最大已分配内存 |
| `max_memory_reserved()` | 无 | int | 获取最大保留内存 |

### 分布式通信 (Distributed Communication)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `distributed_is_initialized()` | 无 | bool | 检查分布式是否已初始化 |
| `distributed_init_process_group(backend=None, timeout=None, init_method=None)` | backend: 后端, timeout: 超时, init_method: 初始化方法 | None | 初始化进程组 |
| `distributed_barrier()` | 无 | None | 进程同步屏障 |
| `distributed_get_rank()` | 无 | int | 获取当前进程排名 |
| `distributed_get_world_size(group=None)` | group: 进程组 | int | 获取进程组大小 |
| `distributed_get_global_rank(group, group_rank)` | group: 进程组, group_rank: 组内排名 | int | 获取全局排名 |
| `distributed_broadcast(tensor, src, group=None, async_op=False)` | tensor: 张量, src: 源进程, group: 进程组, async_op: 是否异步 | None | 广播张量 |
| `distributed_broadcast_object_list(object_list, src, group=None)` | object_list: 对象列表, src: 源进程, group: 进程组 | None | 广播对象列表 |
| `distributed_all_reduce(tensor, op=None, group=None, async_op=False)` | tensor: 张量, op: 操作, group: 进程组, async_op: 是否异步 | None | 全归约操作 |
| `distributed_all_gather(tensor_list, tensor, group=None, async_op=False)` | tensor_list: 张量列表, tensor: 输入张量, group: 进程组, async_op: 是否异步 | None | 全收集操作 |
| `distributed_all_gather_object(object_list, obj, group=None)` | object_list: 对象列表, obj: 对象, group: 进程组 | None | 全收集对象 |
| `distributed_ReduceOp()` | 无 | ReduceOp 类 | 返回 ReduceOp 类 |
| `broadcast_dict_tensor(tensors, src, group)` | tensors: 张量字典, src: 源进程, group: 进程组 | None | 广播字典中的张量 |

### 设备网格 (Device Mesh)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `init_device_mesh(device_type, mesh_shape, mesh_dim_names)` | device_type: 设备类型, mesh_shape: 网格形状, mesh_dim_names: 维度名称 | DeviceMesh | 初始化设备网格 |

### 编译 (Compilation)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `compile(func, dynamic=False)` | func: 函数, dynamic: 是否动态 | 编译后的函数 | 编译函数 |

### 序列化 (Serialization)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `serialization_add_safe_globals(classes)` | classes: 类列表 | None | 添加序列化安全全局类 |

---

## BackendAdapter 接口说明

`BackendAdapter` 是后端适配器的基类，定义了所有 megatron 相关接口的规范。

### 并行状态 (Parallel State)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, use_sharp=False, context_parallel_size=1, expert_model_parallel_size=1, expert_tensor_parallel_size=1, nccl_communicator_config_path=None)` | 各种并行大小参数 | None | 初始化模型并行 |
| `get_tensor_model_parallel_rank()` | 无 | int | 获取张量模型并行排名 |
| `get_tensor_model_parallel_world_size()` | 无 | int | 获取张量模型并行世界大小 |
| `get_tensor_model_parallel_group()` | 无 | 进程组 | 获取张量模型并行进程组 |
| `get_pipeline_model_parallel_rank()` | 无 | int | 获取流水线模型并行排名 |
| `get_pipeline_model_parallel_world_size()` | 无 | int | 获取流水线模型并行世界大小 |
| `get_pipeline_model_parallel_last_rank()` | 无 | int | 获取流水线模型并行最后一个排名 |
| `get_pipeline_model_parallel_group()` | 无 | 进程组 | 获取流水线模型并行进程组 |
| `get_data_parallel_rank()` | 无 | int | 获取数据并行排名 |
| `get_data_parallel_world_size()` | 无 | int | 获取数据并行世界大小 |
| `get_data_parallel_group(with_context_parallel=False)` | with_context_parallel: 是否包含上下文并行 | 进程组 | 获取数据并行进程组 |
| `get_context_parallel_rank()` | 无 | int | 获取上下文并行排名 |
| `get_context_parallel_world_size()` | 无 | int | 获取上下文并行世界大小 |
| `get_virtual_pipeline_model_parallel_world_size()` | 无 | int | 获取虚拟流水线模型并行世界大小 |
| `set_virtual_pipeline_model_parallel_rank(rank)` | rank: 排名 | None | 设置虚拟流水线模型并行排名 |
| `is_pipeline_first_stage(ignore_virtual=False, vp_stage=None)` | ignore_virtual: 是否忽略虚拟, vp_stage: 虚拟阶段 | bool | 判断是否为流水线第一阶段 |
| `is_pipeline_last_stage(ignore_virtual=False)` | ignore_virtual: 是否忽略虚拟 | bool | 判断是否为流水线最后阶段 |
| `get_pipeline_model_parallel_split_rank()` | 无 | int | 获取流水线模型并行分割排名 |
| `is_pipeline_stage_before_split()` | 无 | bool | 判断是否在分割之前 |
| `is_pipeline_stage_after_split()` | 无 | bool | 判断是否在分割之后 |
| `get_expert_model_parallel_world_size()` | 无 | int | 获取专家模型并行世界大小 |
| `get_expert_model_parallel_rank()` | 无 | int | 获取专家模型并行排名 |
| `get_expert_tensor_parallel_world_size()` | 无 | int | 获取专家张量并行世界大小 |
| `get_expert_model_parallel_group()` | 无 | 进程组 | 获取专家模型并行进程组 |
| `get_expert_tensor_parallel_group()` | 无 | 进程组 | 获取专家张量并行进程组 |

### 张量并行 (Tensor Parallel)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `model_parallel_cuda_manual_seed(seed)` | seed: 随机种子 | None | 设置模型并行 CUDA 随机种子 |
| `vocab_parallel_cross_entropy(vocab_parallel_logits, target)` | vocab_parallel_logits: 词汇并行 logits, target: 目标 | Tensor | 计算词汇并行交叉熵 |
| `set_defaults_if_not_set_tensor_model_parallel_attributes(param)` | param: 参数 | None | 设置张量模型并行属性默认值 |
| `get_cuda_rng_tracker()` | 无 | RNGTracker | 获取 CUDA 随机数生成器跟踪器 |

### 流水线并行 (Pipeline Parallel)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `get_forward_backward_func()` | 无 | 函数 | 获取前向反向传播函数 |

### 分布式训练 (Distributed Training)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `finalize_model_grads(*args, **kwargs)` | *args, **kwargs: 参数 | None | 完成模型梯度计算 |
| `DistributedDataParallel(*args, **kwargs)` | *args, **kwargs: 参数 | DDP 对象 | 创建分布式数据并行对象 |
| `DistributedDataParallelConfig(*args, **kwargs)` | *args, **kwargs: 参数 | DDPConfig 对象 | 创建分布式数据并行配置 |

### 优化器 (Optimizer)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `DistributedOptimizer(*args, **kwargs)` | *args, **kwargs: 参数 | Optimizer 对象 | 创建分布式优化器 |
| `OptimizerConfig(*args, **kwargs)` | *args, **kwargs: 参数 | OptimizerConfig 类 | 返回优化器配置类 |
| `get_megatron_optimizer_native(*args, **kwargs)` | *args, **kwargs: 参数 | 函数 | 获取原生 Megatron 优化器函数 |
| `ChainedOptimizer(*args, **kwargs)` | *args, **kwargs: 参数 | ChainedOptimizer 对象 | 创建链式优化器 |
| `OptimizerParamScheduler(*args, **kwargs)` | *args, **kwargs: 参数 | OptimizerParamScheduler 类 | 返回优化器参数调度器类 |

### 模型配置 (Model Configuration)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `ModelParallelConfig(*args, **kwargs)` | *args, **kwargs: 参数 | ModelParallelConfig 对象 | 创建模型并行配置 |
| `TransformerConfig(*args, **kwargs)` | *args, **kwargs: 参数 | TransformerConfig 对象 | 创建 Transformer 配置 |
| `ModelType()` | 无 | ModelType 枚举 | 返回模型类型枚举 |
| `AttnBackend()` | 无 | AttnBackend 枚举 | 返回注意力后端枚举 |

### 模型类 (Model Classes)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `GPTModel(*args, **kwargs)` | *args, **kwargs: 参数 | GPTModel 对象 | 创建 GPT 模型 |
| `Float16Module(*args, **kwargs)` | *args, **kwargs: 参数 | Float16Module 对象 | 创建 Float16 模块 |

### 模型工具 (Model Utilities)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `get_attr_wrapped_model(*args, **kwargs)` | *args, **kwargs: 参数 | 属性值 | 获取包装模型的属性 |

### 检查点 (Checkpointing)

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `save_checkpoint(*args, **kwargs)` | *args, **kwargs: 参数 | 保存结果 | 保存检查点 |
| `load_checkpoint(*args, **kwargs)` | *args, **kwargs: 参数 | 加载结果 | 加载检查点 |
| `ShardedObject(*args, **kwargs)` | *args, **kwargs: 参数 | ShardedObject 对象 | 创建分片对象 |

### verl.models.mcore

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|--------|------|
| `hf_to_mcore_config(hf_config, dtype, **override_transformer_config_kwargs)` | hf_config: HuggingFace 配置, dtype: 数据类型, **override_transformer_config_kwargs: 覆盖参数 | TransformerConfig | 将 HuggingFace 配置转换为 mcore 配置 |
| `mapping_string_to_attn_backend(args)` | args: 参数 | AttnBackend | 将字符串映射到注意力后端 |
| `McoreModuleWrapperConfig(*args, **kwargs)` | *args, **kwargs: 参数 | McoreModuleWrapperConfig 对象 | 创建 mcore 模块包装器配置 |
| `make_megatron_module(*args, **kwargs)` | *args, **kwargs: 参数 | 模块对象 | 创建 Megatron 模块 |
| `gptmodel_forward_1f1b_overlap(*args, **kwargs)` | *args, **kwargs: 参数 | 输出 | GPT 模型前向传播（1F1B 重叠） |
| `get_mcore_forward_fn(hf_config)` | hf_config: HuggingFace 配置 | 函数 | 获取 mcore 前向传播函数 |
| `get_mcore_forward_fused_fn(hf_config)` | hf_config: HuggingFace 配置 | 函数 | 获取 mcore 融合前向传播函数 |

---

## 总结

- **FrameWorkAdapter**: 共 58 个接口方法，涵盖数据类型、张量操作、神经网络、分布式通信等框架相关功能
- **BackendAdapter**: 共 49 个接口方法，涵盖并行状态、张量并行、流水线并行、优化器、模型配置等后端相关功能

所有方法在基类中都抛出 `NotImplementedError`，必须在子类中实现。当前实现：
- `TorchAdapter(FrameWorkAdapter)`: 实现了所有 torch 相关接口
- `MegatronAdapter(BackendAdapter)`: 实现了所有 megatron 相关接口

