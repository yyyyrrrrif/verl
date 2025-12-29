# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapters for extracting and encapsulating torch and megatron interfaces
used in megatron_workers.py and megatron_actor.py.
"""


class TorchAdapter:
    """
    Adapter class that extracts and encapsulates all torch interfaces.
    All methods are placeholders that should be implemented to call the actual torch APIs.
    """

    # ==================== Base Operations ====================

    @staticmethod
    def manual_seed(seed: int):
        """Replace torch.manual_seed(seed)"""
        import torch
        torch.manual_seed(seed)

    @staticmethod
    def float16():
        """Replace torch.float16"""
        import torch
        return torch.float16

    @staticmethod
    def bfloat16():
        """Replace torch.bfloat16"""
        import torch
        return torch.bfloat16

    @staticmethod
    def float32():
        """Replace torch.float32"""
        import torch
        return torch.float32

    @staticmethod
    def int64():
        """Replace torch.int64"""
        import torch
        return torch.int64

    @staticmethod
    def long():
        """Replace torch.long"""
        import torch
        return torch.long

    @staticmethod
    def Tensor():
        """Replace torch.Tensor()"""
        import torch
        return torch.Tensor

    @staticmethod
    def tensor(data, dtype=None, device=None):
        """Replace torch.tensor(data, dtype, device)"""
        import torch
        return torch.tensor(data, dtype=dtype, device=device)

    @staticmethod
    def empty(size, dtype=None, device=None):
        """Replace torch.empty(size, dtype, device)"""
        import torch
        return torch.empty(size, dtype=dtype, device=device)

    @staticmethod
    def cat(tensors, dim=0):
        """Replace torch.cat(tensors, dim)"""
        import torch
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def no_grad():
        """Replace torch.no_grad() context manager"""
        import torch
        return torch.no_grad()

    @staticmethod
    def empty_like(input, dtype=None, device=None):
        """Replace torch.empty_like(input, dtype, device)"""
        import torch
        return torch.empty_like(input, dtype=dtype, device=device)

    @staticmethod
    def Size(*args):
        """Replace torch.Size(*args)"""
        import torch
        return torch.Size(*args)

    @staticmethod
    def squeeze(input, dim=None):
        """Replace torch.squeeze(input, dim)"""
        import torch
        return torch.squeeze(input, dim=dim)

    @staticmethod
    def clip(input, min=None, max=None):
        """Replace torch.clip(input, min, max)"""
        import torch
        return torch.clip(input, min=min, max=max)

    @staticmethod
    def cumsum(input, dim, dtype=None):
        """Replace torch.cumsum(input, dim, dtype)"""
        import torch
        return torch.cumsum(input, dim=dim, dtype=dtype)

    @staticmethod
    def device(device):
        """Replace torch.device(device)"""
        import torch
        return torch.device(device)

    @staticmethod
    def roll(input, shifts, dims):
        """Replace torch.roll(input, shifts, dims)"""
        import torch
        return torch.roll(input, shifts=shifts, dims=dims)

    # ==================== torch.distributed ====================

    @staticmethod
    def distributed_is_initialized():
        """Replace torch.distributed.is_initialized()"""
        import torch.distributed
        return torch.distributed.is_initialized()

    @staticmethod
    def distributed_init_process_group(backend=None, timeout=None, init_method=None):
        """Replace torch.distributed.init_process_group(backend, timeout, init_method)"""
        import torch.distributed
        torch.distributed.init_process_group(
            backend=backend,
            timeout=timeout,
            init_method=init_method,
        )

    @staticmethod
    def distributed_barrier():
        """Replace torch.distributed.barrier()"""
        import torch.distributed
        torch.distributed.barrier()

    @staticmethod
    def distributed_broadcast(tensor, src, group=None, async_op=False):
        """Replace torch.distributed.broadcast(tensor, src, group, async_op)"""
        import torch.distributed
        torch.distributed.broadcast(
            tensor=tensor,
            src=src,
            group=group,
            async_op=async_op,
        )

    @staticmethod
    def distributed_get_rank():
        """Replace torch.distributed.get_rank()"""
        import torch.distributed
        return torch.distributed.get_rank()

    @staticmethod
    def distributed_all_gather_object(object_list, obj, group=None):
        """Replace torch.distributed.all_gather_object(object_list, obj, group)"""
        import torch.distributed
        torch.distributed.all_gather_object(object_list=object_list, obj=obj, group=group)

    @staticmethod
    def distributed_broadcast_object_list(object_list, src, group=None):
        """Replace torch.distributed.broadcast_object_list(object_list, src, group)"""
        import torch.distributed
        torch.distributed.broadcast_object_list(object_list=object_list, src=src, group=group)

    @staticmethod
    def distributed_all_gather(tensor_list, tensor, group=None, async_op=False):
        """Replace torch.distributed.all_gather(tensor_list, tensor, group, async_op)"""
        import torch.distributed
        torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)

    @staticmethod
    def distributed_all_reduce(tensor, op=None, group=None, async_op=False):
        """Replace torch.distributed.all_reduce(tensor, op, group, async_op)"""
        import torch.distributed
        torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)

    @staticmethod
    def distributed_get_global_rank(group, group_rank):
        """Replace torch.distributed.get_global_rank(group, group_rank)"""
        import torch.distributed
        return torch.distributed.get_global_rank(group=group, group_rank=group_rank)

    @staticmethod
    def distributed_get_world_size(group=None):
        """Replace torch.distributed.get_world_size(group)"""
        import torch.distributed
        return torch.distributed.get_world_size(group=group)

    @staticmethod
    def distributed_ReduceOp():
        """Replace torch.distributed.ReduceOp"""
        import torch.distributed
        return torch.distributed.ReduceOp

    # ==================== torch.nn ====================

    @staticmethod
    def nn_ModuleList(*args):
        """Replace nn.ModuleList"""
        from torch import nn
        return nn.ModuleList(*args)

    @staticmethod
    def nn_Module(*args, **kwargs):
        """Replace nn.Module"""
        from torch import nn
        return nn.Module(*args, **kwargs)

    @staticmethod
    def nn_Linear(in_features, out_features, bias=True, dtype=None):
        """Replace nn.Linear"""
        from torch import nn
        return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    @staticmethod
    def nn_Sequential(*args):
        """Replace nn.Sequential"""
        from torch import nn
        return nn.Sequential(*args)

    # ==================== torch.nn.functional ====================

    @staticmethod
    def F_silu(input):
        """Replace F.silu(input)"""
        import torch.nn.functional as F
        return F.silu(input)

    @staticmethod
    def F_pad(input, pad, mode="constant", value=0):
        """Replace F.pad(input, pad, mode, value)"""
        import torch.nn.functional as F
        return F.pad(input, pad, mode=mode, value=value)

    # ==================== torch.autograd ====================

    @staticmethod
    def autograd_Function():
        """Replace torch.autograd.Function"""
        import torch.autograd
        return torch.autograd.Function

    @staticmethod
    def compile(*args, **kwargs):
        """Replace torch.compile(*args, **kwargs)"""
        import torch
        return torch.compile(*args, **kwargs)

    # ==================== torch.cuda ====================

    @staticmethod
    def cuda_is_available():
        """Replace torch.cuda.is_available()"""
        import torch
        return torch.cuda.is_available()

    @staticmethod
    def cuda_memory_set_allocator_settings(settings):
        """Replace torch.cuda.memory._set_allocator_settings(settings)"""
        import torch
        torch.cuda.memory._set_allocator_settings(settings)

    # ==================== torch.npu ====================

    @staticmethod
    def npu_is_available():
        """Replace torch.npu.is_available() if available"""
        import torch
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False

    # ==================== torch.distributed.device_mesh ====================

    @staticmethod
    def init_device_mesh(device_type, mesh_shape, mesh_dim_names):
        """Replace init_device_mesh from torch.distributed.device_mesh"""
        from torch.distributed.device_mesh import init_device_mesh
        return init_device_mesh(
            device_type=device_type,
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

    # ==================== Device Management (get_torch_device) ====================

    @staticmethod
    def get_torch_device():
        """Replace get_torch_device()"""
        from verl.utils.device import get_torch_device
        return get_torch_device()

    @staticmethod
    def device_count():
        """Replace get_torch_device().device_count()"""
        from verl.utils.device import get_torch_device
        return get_torch_device().device_count()

    @staticmethod
    def set_device(rank):
        """Replace get_torch_device().set_device(rank)"""
        from verl.utils.device import get_torch_device
        get_torch_device().set_device(rank)

    @staticmethod
    def get_rng_state():
        """Replace get_torch_device().get_rng_state()"""
        from verl.utils.device import get_torch_device
        return get_torch_device().get_rng_state()

    @staticmethod
    def set_rng_state(state):
        """Replace get_torch_device().set_rng_state(state)"""
        from verl.utils.device import get_torch_device
        get_torch_device().set_rng_state(state)

    @staticmethod
    def manual_seed_device(seed):
        """Replace get_torch_device().manual_seed(seed)"""
        from verl.utils.device import get_torch_device
        get_torch_device().manual_seed(seed)

    @staticmethod
    def empty_cache():
        """Replace get_torch_device().empty_cache()"""
        from verl.utils.device import get_torch_device
        get_torch_device().empty_cache()

    @staticmethod
    def max_memory_allocated():
        """Replace get_torch_device().max_memory_allocated()"""
        from verl.utils.device import get_torch_device
        return get_torch_device().max_memory_allocated()

    @staticmethod
    def max_memory_reserved():
        """Replace get_torch_device().max_memory_reserved()"""
        from verl.utils.device import get_torch_device
        return get_torch_device().max_memory_reserved()

    # ==================== Model Methods ====================

    @staticmethod
    def model_train(model):
        """Replace model.train()"""
        model.train()

    @staticmethod
    def zero_grad_buffer(model):
        """Replace model.zero_grad_buffer()"""
        model.zero_grad_buffer()


class MegatronAdapter:
    """
    Adapter class that extracts and encapsulates all megatron interfaces.
    All methods are placeholders that should be implemented to call the actual megatron APIs.
    """

    # ==================== Parallel State ====================

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        use_sharp=False,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        nccl_communicator_config_path=None,
    ):
        """Replace mpu.initialize_model_parallel(...)"""
        from megatron.core import parallel_state as mpu
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            use_sharp=use_sharp,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            nccl_communicator_config_path=nccl_communicator_config_path,
        )

    @staticmethod
    def get_tensor_model_parallel_rank():
        """Replace mpu.get_tensor_model_parallel_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_tensor_model_parallel_rank()

    @staticmethod
    def get_pipeline_model_parallel_rank():
        """Replace mpu.get_pipeline_model_parallel_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_pipeline_model_parallel_rank()

    @staticmethod
    def get_pipeline_model_parallel_world_size():
        """Replace mpu.get_pipeline_model_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_pipeline_model_parallel_world_size()

    @staticmethod
    def get_pipeline_model_parallel_last_rank():
        """Replace mpu.get_pipeline_model_parallel_last_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_pipeline_model_parallel_last_rank()

    @staticmethod
    def get_pipeline_model_parallel_group():
        """Replace mpu.get_pipeline_model_parallel_group()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_pipeline_model_parallel_group()

    @staticmethod
    def get_data_parallel_rank():
        """Replace mpu.get_data_parallel_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_data_parallel_rank()

    @staticmethod
    def get_data_parallel_world_size():
        """Replace mpu.get_data_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_data_parallel_world_size()

    @staticmethod
    def get_context_parallel_rank():
        """Replace mpu.get_context_parallel_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_context_parallel_rank()

    @staticmethod
    def get_virtual_pipeline_model_parallel_world_size():
        """Replace mpu.get_virtual_pipeline_model_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_virtual_pipeline_model_parallel_world_size()

    @staticmethod
    def is_pipeline_last_stage(ignore_virtual=False):
        """Replace mpu.is_pipeline_last_stage(ignore_virtual)"""
        from megatron.core import parallel_state as mpu
        return mpu.is_pipeline_last_stage(ignore_virtual=ignore_virtual)

    @staticmethod
    def get_tensor_model_parallel_world_size():
        """Replace mpu.get_tensor_model_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_tensor_model_parallel_world_size()

    @staticmethod
    def get_context_parallel_world_size():
        """Replace mpu.get_context_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_context_parallel_world_size()

    @staticmethod
    def get_tensor_model_parallel_group():
        """Replace mpu.get_tensor_model_parallel_group()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_tensor_model_parallel_group()

    @staticmethod
    def get_expert_model_parallel_world_size():
        """Replace mpu.get_expert_model_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_expert_model_parallel_world_size()

    @staticmethod
    def get_expert_tensor_parallel_world_size():
        """Replace mpu.get_expert_tensor_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_expert_tensor_parallel_world_size()

    @staticmethod
    def get_expert_model_parallel_group():
        """Replace mpu.get_expert_model_parallel_group()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_expert_model_parallel_group()

    @staticmethod
    def get_expert_tensor_parallel_group():
        """Replace mpu.get_expert_tensor_parallel_group()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_expert_tensor_parallel_group()

    @staticmethod
    def set_virtual_pipeline_model_parallel_rank(rank):
        """Replace mpu.set_virtual_pipeline_model_parallel_rank(rank)"""
        from megatron.core import parallel_state as mpu
        mpu.set_virtual_pipeline_model_parallel_rank(rank)

    @staticmethod
    def is_pipeline_first_stage(ignore_virtual=False, vp_stage=None):
        """Replace mpu.is_pipeline_first_stage(ignore_virtual, vp_stage)"""
        from megatron.core import parallel_state as mpu
        kwargs = {}
        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage
        return mpu.is_pipeline_first_stage(ignore_virtual=ignore_virtual, **kwargs)

    @staticmethod
    def get_pipeline_model_parallel_split_rank():
        """Replace mpu.get_pipeline_model_parallel_split_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_pipeline_model_parallel_split_rank()

    @staticmethod
    def is_pipeline_stage_before_split():
        """Replace mpu.is_pipeline_stage_before_split()"""
        from megatron.core import parallel_state as mpu
        return mpu.is_pipeline_stage_before_split()

    @staticmethod
    def is_pipeline_stage_after_split():
        """Replace mpu.is_pipeline_stage_after_split()"""
        from megatron.core import parallel_state as mpu
        return mpu.is_pipeline_stage_after_split()

    # ==================== megatron.core.optimizer ====================

    @staticmethod
    def DistributedOptimizer(*args, **kwargs):
        """Replace DistributedOptimizer from megatron.core.optimizer"""
        from megatron.core.optimizer import DistributedOptimizer
        return DistributedOptimizer(*args, **kwargs)

    # ==================== megatron.core.pipeline_parallel ====================

    @staticmethod
    def get_forward_backward_func():
        """Replace get_forward_backward_func from megatron.core.pipeline_parallel"""
        from megatron.core.pipeline_parallel import get_forward_backward_func
        return get_forward_backward_func()

    # ==================== megatron.core.distributed ====================

    @staticmethod
    def finalize_model_grads(*args, **kwargs):
        """Replace finalize_model_grads from megatron.core.distributed"""
        from megatron.core.distributed import finalize_model_grads
        return finalize_model_grads(*args, **kwargs)

    # ==================== Tensor Parallel ====================

    @staticmethod
    def model_parallel_cuda_manual_seed(seed):
        """Replace tensor_parallel.model_parallel_cuda_manual_seed(seed)"""
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)

    @staticmethod
    def set_defaults_if_not_set_tensor_model_parallel_attributes(param):
        """Replace tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)"""
        from megatron.core import tensor_parallel
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    @staticmethod
    def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
        """Replace tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits, target)"""
        from megatron.core import tensor_parallel
        return tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=vocab_parallel_logits, target=target)

    # ==================== megatron.core.models.gpt.gpt_model ====================

    @staticmethod
    def GPTModel(*args, **kwargs):
        """Replace GPTModel from megatron.core.models.gpt.gpt_model"""
        from megatron.core.models.gpt.gpt_model import GPTModel
        return GPTModel(*args, **kwargs)

    # ==================== Megatron Utils Functions ====================

    @staticmethod
    def load_megatron_model_to_gpu(model, load_grad=False):
        """Replace load_megatron_model_to_gpu from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import load_megatron_model_to_gpu
        return load_megatron_model_to_gpu(model, load_grad=load_grad)

    @staticmethod
    def load_megatron_optimizer(optimizer):
        """Replace load_megatron_optimizer from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import load_megatron_optimizer
        return load_megatron_optimizer(optimizer)

    @staticmethod
    def offload_megatron_model_to_cpu(model):
        """Replace offload_megatron_model_to_cpu from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import offload_megatron_model_to_cpu
        return offload_megatron_model_to_cpu(model)

    @staticmethod
    def offload_megatron_optimizer(optimizer):
        """Replace offload_megatron_optimizer from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import offload_megatron_optimizer
        return offload_megatron_optimizer(optimizer)

    @staticmethod
    def per_tensor_generator(module, model_config, weight_converter, tf_config, layer_name_mapping):
        """Replace per_tensor_generator from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import per_tensor_generator
        return per_tensor_generator(
            module, model_config, weight_converter, tf_config, layer_name_mapping
        )

    @staticmethod
    def register_megatron_training_hooks(module, optimizer):
        """Replace register_megatron_training_hooks from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import register_megatron_training_hooks
        return register_megatron_training_hooks(module, optimizer)

    @staticmethod
    def get_model_config(module):
        """Replace get_model_config from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import get_model_config
        return get_model_config(module)

    # ==================== Megatron Optimizer Utils ====================

    @staticmethod
    def get_megatron_optimizer(model, config):
        """Replace get_megatron_optimizer from verl.utils.megatron.optimizer"""
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        return get_megatron_optimizer(model=model, config=config)

    @staticmethod
    def get_megatron_optimizer_param_scheduler(optimizer, config):
        """Replace get_megatron_optimizer_param_scheduler from verl.utils.megatron.optimizer"""
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler
        return get_megatron_optimizer_param_scheduler(optimizer=optimizer, config=config)

    @staticmethod
    def init_megatron_optim_config(optim_config, fp16=False):
        """Replace init_megatron_optim_config from verl.utils.megatron.optimizer"""
        from verl.utils.megatron.optimizer import init_megatron_optim_config
        return init_megatron_optim_config(optim_config, fp16=fp16)

    @staticmethod
    def get_megatron_last_lr(optimizer):
        """Replace get_megatron_last_lr from verl.utils.megatron.optimizer"""
        from verl.utils.megatron.optimizer import get_megatron_last_lr
        return get_megatron_last_lr(optimizer)

    # ==================== Megatron Tensor Parallel Utils ====================

    @staticmethod
    def vocab_parallel_entropy(logits):
        """Replace vocab_parallel_entropy from verl.utils.megatron.tensor_parallel"""
        from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
        return vocab_parallel_entropy(logits)

    @staticmethod
    def vocab_parallel_log_probs_from_logits(logits, label):
        """Replace vocab_parallel_log_probs_from_logits from verl.utils.megatron.tensor_parallel"""
        from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
        return vocab_parallel_log_probs_from_logits(logits, label)

    # ==================== Megatron Pipeline Parallel Utils ====================

    @staticmethod
    def make_batch_generator(micro_batches, vpp_size):
        """Replace make_batch_generator from verl.utils.megatron.pipeline_parallel"""
        from verl.utils.megatron.pipeline_parallel import make_batch_generator
        return make_batch_generator(micro_batches, vpp_size=vpp_size)

    # ==================== Model Weight Loading ====================

    @staticmethod
    def load_mcore_dist_weights(module, dist_checkpointing_path, is_value_model=False):
        """Replace load_mcore_dist_weights from verl.utils.model"""
        from verl.utils.model import load_mcore_dist_weights
        return load_mcore_dist_weights(
            module, dist_checkpointing_path, is_value_model=is_value_model
        )

    @staticmethod
    def load_megatron_gptmodel_weights(config, hf_config, module, params_dtype, is_value_model=False):
        """Replace load_megatron_gptmodel_weights from verl.utils.model"""
        from verl.utils.model import load_megatron_gptmodel_weights
        return load_megatron_gptmodel_weights(
            config, hf_config, module, params_dtype=params_dtype, is_value_model=is_value_model
        )

    # ==================== Module Building ====================

    @staticmethod
    def McoreModuleWrapperConfig(*args, **kwargs):
        """Replace McoreModuleWrapperConfig from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import McoreModuleWrapperConfig
        return McoreModuleWrapperConfig(*args, **kwargs)

    @staticmethod
    def make_megatron_module(
        wrap_config,
        tf_config,
        hf_config,
        bridge=None,
        override_model_config=None,
        override_ddp_config=None,
    ):
        """Replace make_megatron_module from verl.utils.megatron_utils"""
        from verl.utils.megatron_utils import make_megatron_module
        return make_megatron_module(
            wrap_config=wrap_config,
            tf_config=tf_config,
            hf_config=hf_config,
            bridge=bridge,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )

    # ==================== megatron.core.distributed ====================

    @staticmethod
    def DistributedDataParallel(*args, **kwargs):
        """Replace DistributedDataParallel from megatron.core.distributed"""
        from megatron.core.distributed import DistributedDataParallel as DDP
        return DDP(*args, **kwargs)

    @staticmethod
    def DistributedDataParallelConfig(*args, **kwargs):
        """Replace DistributedDataParallelConfig from megatron.core.distributed"""
        from megatron.core.distributed import DistributedDataParallelConfig
        return DistributedDataParallelConfig(*args, **kwargs)

    # ==================== megatron.core.transformer ====================

    @staticmethod
    def Float16Module(*args, **kwargs):
        """Replace Float16Module from megatron.core.transformer.module"""
        from megatron.core.transformer.module import Float16Module
        return Float16Module(*args, **kwargs)

    @staticmethod
    def TransformerConfig(*args, **kwargs):
        """Replace TransformerConfig from megatron.core.transformer"""
        from megatron.core.transformer import TransformerConfig
        return TransformerConfig(*args, **kwargs)

    # ==================== megatron.core ====================

    @staticmethod
    def ModelParallelConfig(*args, **kwargs):
        """Replace ModelParallelConfig from megatron.core"""
        from megatron.core import ModelParallelConfig
        return ModelParallelConfig(*args, **kwargs)

    @staticmethod
    def ModelType():
        """Replace ModelType from megatron.core.enums"""
        from megatron.core.enums import ModelType
        return ModelType

    @staticmethod
    def ChainedOptimizer(*args, **kwargs):
        """Replace ChainedOptimizer from megatron.core.optimizer"""
        from megatron.core.optimizer import ChainedOptimizer
        return ChainedOptimizer(*args, **kwargs)

    # ==================== megatron.core.utils ====================

    @staticmethod
    def get_attr_wrapped_model(model, attr_name, allow_none=False):
        """Replace get_attr_wrapped_model from megatron.core.utils"""
        from megatron.core.utils import get_attr_wrapped_model
        return get_attr_wrapped_model(model, attr_name, allow_none=allow_none)

    # ==================== megatron.core.optimizer ====================

    @staticmethod
    def OptimizerConfig(*args, **kwargs):
        """Replace OptimizerConfig from megatron.core.optimizer"""
        from megatron.core.optimizer import OptimizerConfig
        return OptimizerConfig(*args, **kwargs)

    # ==================== Checkpointing ====================

    @staticmethod
    def load_checkpoint(sharded_state_dict, checkpoint_path, strict=None):
        """Replace dist_checkpointing.load(sharded_state_dict, checkpoint_path, strict)"""
        from megatron.core import dist_checkpointing
        dist_checkpointing.load(sharded_state_dict, checkpoint_path, strict=strict)
