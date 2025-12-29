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
used in mindspore_workers.py and mindspore_actor.py.

TorchAdapter: Encapsulates torch interfaces (to be replaced with MindSpore interfaces)
MegatronAdapter: Encapsulates megatron interfaces (to be replaced with MindFormers interfaces)
"""


class FrameWorkAdapter:
    """
    Base adapter class that defines the interface for framework adapters.
    All methods raise NotImplementedError and must be implemented by subclasses.
    """

    # ==================== Data Types ====================

    @staticmethod
    def float16():
        """Replace torch.float16"""
        raise NotImplementedError("float16 must be implemented by subclass")

    @staticmethod
    def bfloat16():
        """Replace torch.bfloat16"""
        raise NotImplementedError("bfloat16 must be implemented by subclass")

    @staticmethod
    def float32():
        """Replace torch.float32"""
        raise NotImplementedError("float32 must be implemented by subclass")

    @staticmethod
    def int64():
        """Replace torch.int64"""
        raise NotImplementedError("int64 must be implemented by subclass")

    @staticmethod
    def long():
        """Replace torch.long"""
        raise NotImplementedError("long must be implemented by subclass")

    @staticmethod
    def dtype():
        """Replace torch.dtype"""
        raise NotImplementedError("dtype must be implemented by subclass")

    @staticmethod
    def PrecisionType():
        """Replace PrecisionType from verl.utils.torch_dtypes"""
        raise NotImplementedError("PrecisionType must be implemented by subclass")

    # ==================== Tensor Creation ====================

    @staticmethod
    def Tensor():
        """Replace torch.Tensor()"""
        raise NotImplementedError("Tensor must be implemented by subclass")

    @staticmethod
    def tensor(data, dtype=None, device=None):
        """Replace torch.tensor(data, dtype, device)"""
        raise NotImplementedError("tensor must be implemented by subclass")

    @staticmethod
    def empty(size, dtype=None, device=None):
        """Replace torch.empty(size, dtype, device)"""
        raise NotImplementedError("empty must be implemented by subclass")

    @staticmethod
    def empty_like(input):
        """Replace torch.empty_like(input)"""
        raise NotImplementedError("empty_like must be implemented by subclass")

    @staticmethod
    def zeros(size, dtype=None, device=None, requires_grad=False):
        """Replace torch.zeros(size, dtype, device, requires_grad)"""
        raise NotImplementedError("zeros must be implemented by subclass")

    @staticmethod
    def zeros_like(input, dtype=None, device=None, requires_grad=False):
        """Replace torch.zeros_like(input, dtype, device, requires_grad)"""
        raise NotImplementedError("zeros_like must be implemented by subclass")

    @staticmethod
    def ones_like(input, dtype=None, device=None, requires_grad=False):
        """Replace torch.ones_like(input, dtype, device, requires_grad)"""
        raise NotImplementedError("ones_like must be implemented by subclass")

    @staticmethod
    def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
        """Replace torch.arange(start, end, step, dtype, device, requires_grad)"""
        raise NotImplementedError("arange must be implemented by subclass")

    @staticmethod
    def Size(*args):
        """Replace torch.Size(*args)"""
        raise NotImplementedError("Size must be implemented by subclass")

    # ==================== Tensor Operations ====================

    @staticmethod
    def cat(tensors, dim=0):
        """Replace torch.cat(tensors, dim)"""
        raise NotImplementedError("cat must be implemented by subclass")

    @staticmethod
    def gather(input, dim, index):
        """Replace torch.gather(input, dim, index)"""
        raise NotImplementedError("gather must be implemented by subclass")

    @staticmethod
    def cumsum(input, dim, dtype=None):
        """Replace torch.cumsum(input, dim, dtype)"""
        raise NotImplementedError("cumsum must be implemented by subclass")

    @staticmethod
    def nonzero(input, as_tuple=False):
        """Replace torch.nonzero(input, as_tuple)"""
        raise NotImplementedError("nonzero must be implemented by subclass")

    @staticmethod
    def argmax(input, dim=None, keepdim=False):
        """Replace torch.argmax(input, dim, keepdim)"""
        raise NotImplementedError("argmax must be implemented by subclass")

    @staticmethod
    def roll(input, shifts, dims):
        """Replace torch.roll(input, shifts, dims)"""
        raise NotImplementedError("roll must be implemented by subclass")

    # ==================== Neural Network Modules ====================

    @staticmethod
    def nn_ModuleList(*args, **kwargs):
        """Replace torch.nn.ModuleList(*args, **kwargs)"""
        raise NotImplementedError("nn_ModuleList must be implemented by subclass")

    @staticmethod
    def nn_functional_pad(input, pad, mode='constant', value=0):
        """Replace torch.nn.functional.pad(input, pad, mode, value)"""
        raise NotImplementedError("nn_functional_pad must be implemented by subclass")

    @staticmethod
    def nn_functional_silu(input):
        """Replace torch.nn.functional.silu(input)"""
        raise NotImplementedError("nn_functional_silu must be implemented by subclass")

    # ==================== Automatic Differentiation ====================

    @staticmethod
    def no_grad():
        """Replace torch.no_grad() context manager"""
        raise NotImplementedError("no_grad must be implemented by subclass")

    @staticmethod
    def autograd_Function():
        """Replace torch.autograd.Function"""
        raise NotImplementedError("autograd_Function must be implemented by subclass")

    # ==================== Random Number Generation ====================

    @staticmethod
    def manual_seed(seed: int):
        """Replace torch.manual_seed(seed)"""
        raise NotImplementedError("manual_seed must be implemented by subclass")

    @staticmethod
    def get_rng_state():
        """Replace get_torch_device().get_rng_state()"""
        raise NotImplementedError("get_rng_state must be implemented by subclass")

    @staticmethod
    def set_rng_state(state):
        """Replace get_torch_device().set_rng_state(state)"""
        raise NotImplementedError("set_rng_state must be implemented by subclass")

    @staticmethod
    def manual_seed_device(seed):
        """Replace get_torch_device().manual_seed(seed)"""
        raise NotImplementedError("manual_seed_device must be implemented by subclass")

    @staticmethod
    def get_rng_state_torch():
        """Replace torch.get_rng_state()"""
        raise NotImplementedError("get_rng_state_torch must be implemented by subclass")

    @staticmethod
    def set_rng_state_torch(state):
        """Replace torch.set_rng_state(state)"""
        raise NotImplementedError("set_rng_state_torch must be implemented by subclass")

    # ==================== Device Management ====================

    @staticmethod
    def get_torch_device():
        """Replace get_torch_device()"""
        raise NotImplementedError("get_torch_device must be implemented by subclass")

    @staticmethod
    def device_count():
        """Replace get_torch_device().device_count()"""
        raise NotImplementedError("device_count must be implemented by subclass")

    @staticmethod
    def set_device(rank):
        """Replace get_torch_device().set_device(rank)"""
        raise NotImplementedError("set_device must be implemented by subclass")

    @staticmethod
    def current_device():
        """Replace get_torch_device().current_device()"""
        raise NotImplementedError("current_device must be implemented by subclass")

    @staticmethod
    def cuda_is_available():
        """Replace torch.cuda.is_available()"""
        raise NotImplementedError("cuda_is_available must be implemented by subclass")

    @staticmethod
    def npu_is_available():
        """Replace torch.npu.is_available() if available"""
        raise NotImplementedError("npu_is_available must be implemented by subclass")

    @staticmethod
    def get_device_module(device_name):
        """Replace getattr(torch, device_name)"""
        raise NotImplementedError("get_device_module must be implemented by subclass")

    @staticmethod
    def cuda_memory_set_allocator_settings(settings):
        """Replace torch.cuda.memory._set_allocator_settings(settings)"""
        raise NotImplementedError("cuda_memory_set_allocator_settings must be implemented by subclass")

    # ==================== Memory Management ====================

    @staticmethod
    def empty_cache():
        """Replace get_torch_device().empty_cache()"""
        raise NotImplementedError("empty_cache must be implemented by subclass")

    @staticmethod
    def max_memory_allocated():
        """Replace get_torch_device().max_memory_allocated()"""
        raise NotImplementedError("max_memory_allocated must be implemented by subclass")

    @staticmethod
    def max_memory_reserved():
        """Replace get_torch_device().max_memory_reserved()"""
        raise NotImplementedError("max_memory_reserved must be implemented by subclass")

    # ==================== Distributed Communication ====================

    @staticmethod
    def distributed_is_initialized():
        """Replace torch.distributed.is_initialized()"""
        raise NotImplementedError("distributed_is_initialized must be implemented by subclass")

    @staticmethod
    def distributed_init_process_group(backend=None, timeout=None, init_method=None):
        """Replace torch.distributed.init_process_group(backend, timeout, init_method)"""
        raise NotImplementedError("distributed_init_process_group must be implemented by subclass")

    @staticmethod
    def distributed_barrier():
        """Replace torch.distributed.barrier()"""
        raise NotImplementedError("distributed_barrier must be implemented by subclass")

    @staticmethod
    def distributed_get_rank():
        """Replace torch.distributed.get_rank()"""
        raise NotImplementedError("distributed_get_rank must be implemented by subclass")

    @staticmethod
    def distributed_get_world_size(group=None):
        """Replace torch.distributed.get_world_size(group)"""
        raise NotImplementedError("distributed_get_world_size must be implemented by subclass")

    @staticmethod
    def distributed_get_global_rank(group, group_rank):
        """Replace torch.distributed.get_global_rank(group, group_rank)"""
        raise NotImplementedError("distributed_get_global_rank must be implemented by subclass")

    @staticmethod
    def distributed_broadcast(tensor, src, group=None, async_op=False):
        """Replace torch.distributed.broadcast(tensor, src, group, async_op)"""
        raise NotImplementedError("distributed_broadcast must be implemented by subclass")

    @staticmethod
    def distributed_broadcast_object_list(object_list, src, group=None):
        """Replace torch.distributed.broadcast_object_list(object_list, src, group)"""
        raise NotImplementedError("distributed_broadcast_object_list must be implemented by subclass")

    @staticmethod
    def distributed_all_reduce(tensor, op=None, group=None, async_op=False):
        """Replace torch.distributed.all_reduce(tensor, op, group, async_op)"""
        raise NotImplementedError("distributed_all_reduce must be implemented by subclass")

    @staticmethod
    def distributed_all_gather(tensor_list, tensor, group=None, async_op=False):
        """Replace torch.distributed.all_gather(tensor_list, tensor, group, async_op)"""
        raise NotImplementedError("distributed_all_gather must be implemented by subclass")

    @staticmethod
    def distributed_all_gather_object(object_list, obj, group=None):
        """Replace torch.distributed.all_gather_object(object_list, obj, group)"""
        raise NotImplementedError("distributed_all_gather_object must be implemented by subclass")

    @staticmethod
    def distributed_ReduceOp():
        """Replace torch.distributed.ReduceOp"""
        raise NotImplementedError("distributed_ReduceOp must be implemented by subclass")

    @staticmethod
    def broadcast_dict_tensor(tensors, src, group):
        """Replace broadcast_dict_tensor from verl.utils.torch_functional"""
        raise NotImplementedError("broadcast_dict_tensor must be implemented by subclass")

    # ==================== Device Mesh ====================

    @staticmethod
    def init_device_mesh(device_type, mesh_shape, mesh_dim_names):
        """Replace init_device_mesh from torch.distributed.device_mesh"""
        raise NotImplementedError("init_device_mesh must be implemented by subclass")

    # ==================== Compilation ====================

    @staticmethod
    def compile(func, dynamic=False):
        """Replace torch.compile(func, dynamic)"""
        raise NotImplementedError("compile must be implemented by subclass")

    # ==================== Serialization ====================

    @staticmethod
    def serialization_add_safe_globals(classes):
        """Replace torch.serialization.add_safe_globals(classes)"""
        raise NotImplementedError("serialization_add_safe_globals must be implemented by subclass")


class TorchAdapter(FrameWorkAdapter):
    """
    Adapter class that extracts and encapsulates all torch interfaces.
    All methods are placeholders that should be implemented to call the actual torch APIs.
    This adapter will be replaced with MindSpore interfaces in the future.
    """

    # ==================== Data Types ====================

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
    def dtype():
        """Replace torch.dtype"""
        import torch
        return torch.dtype

    @staticmethod
    def PrecisionType():
        """Replace PrecisionType from verl.utils.torch_dtypes"""
        from verl.utils.torch_dtypes import PrecisionType
        return PrecisionType

    # ==================== Tensor Creation ====================

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
    def empty_like(input):
        """Replace torch.empty_like(input)"""
        import torch
        return torch.empty_like(input)

    @staticmethod
    def zeros(size, dtype=None, device=None, requires_grad=False):
        """Replace torch.zeros(size, dtype, device, requires_grad)"""
        import torch
        return torch.zeros(size, dtype=dtype, device=device, requires_grad=requires_grad)

    @staticmethod
    def zeros_like(input, dtype=None, device=None, requires_grad=False):
        """Replace torch.zeros_like(input, dtype, device, requires_grad)"""
        import torch
        return torch.zeros_like(input, dtype=dtype, device=device, requires_grad=requires_grad)

    @staticmethod
    def ones_like(input, dtype=None, device=None, requires_grad=False):
        """Replace torch.ones_like(input, dtype, device, requires_grad)"""
        import torch
        return torch.ones_like(input, dtype=dtype, device=device, requires_grad=requires_grad)

    @staticmethod
    def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
        """Replace torch.arange(start, end, step, dtype, device, requires_grad)"""
        import torch
        if end is None:
            return torch.arange(start, dtype=dtype, device=device, requires_grad=requires_grad)
        return torch.arange(start, end, step=step, dtype=dtype, device=device, requires_grad=requires_grad)

    @staticmethod
    def Size(*args):
        """Replace torch.Size(*args)"""
        import torch
        return torch.Size(*args)

    # ==================== Tensor Operations ====================

    @staticmethod
    def cat(tensors, dim=0):
        """Replace torch.cat(tensors, dim)"""
        import torch
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def gather(input, dim, index):
        """Replace torch.gather(input, dim, index)"""
        import torch
        return torch.gather(input, dim=dim, index=index)

    @staticmethod
    def cumsum(input, dim, dtype=None):
        """Replace torch.cumsum(input, dim, dtype)"""
        import torch
        return torch.cumsum(input, dim=dim, dtype=dtype)

    @staticmethod
    def nonzero(input, as_tuple=False):
        """Replace torch.nonzero(input, as_tuple)"""
        import torch
        return torch.nonzero(input, as_tuple=as_tuple)

    @staticmethod
    def argmax(input, dim=None, keepdim=False):
        """Replace torch.argmax(input, dim, keepdim)"""
        import torch
        return torch.argmax(input, dim=dim, keepdim=keepdim)

    @staticmethod
    def roll(input, shifts, dims):
        """Replace torch.roll(input, shifts, dims)"""
        import torch
        return torch.roll(input, shifts=shifts, dims=dims)

    # ==================== Neural Network Modules ====================

    @staticmethod
    def nn_ModuleList(*args, **kwargs):
        """Replace torch.nn.ModuleList(*args, **kwargs)"""
        from torch import nn
        return nn.ModuleList(*args, **kwargs)

    @staticmethod
    def nn_functional_pad(input, pad, mode='constant', value=0):
        """Replace torch.nn.functional.pad(input, pad, mode, value)"""
        import torch.nn.functional as F
        return F.pad(input, pad, mode=mode, value=value)

    @staticmethod
    def nn_functional_silu(input):
        """Replace torch.nn.functional.silu(input)"""
        import torch.nn.functional as F
        return F.silu(input)

    # ==================== Automatic Differentiation ====================

    @staticmethod
    def no_grad():
        """Replace torch.no_grad() context manager"""
        import torch
        return torch.no_grad()

    @staticmethod
    def autograd_Function():
        """Replace torch.autograd.Function"""
        import torch
        return torch.autograd.Function

    # ==================== Random Number Generation ====================

    @staticmethod
    def manual_seed(seed: int):
        """Replace torch.manual_seed(seed)"""
        import torch
        torch.manual_seed(seed)

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
    def get_rng_state_torch():
        """Replace torch.get_rng_state()"""
        import torch
        return torch.get_rng_state()

    @staticmethod
    def set_rng_state_torch(state):
        """Replace torch.set_rng_state(state)"""
        import torch
        torch.set_rng_state(state)

    # ==================== Device Management ====================

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
    def current_device():
        """Replace get_torch_device().current_device()"""
        from verl.utils.device import get_torch_device
        return get_torch_device().current_device()

    @staticmethod
    def cuda_is_available():
        """Replace torch.cuda.is_available()"""
        import torch
        return torch.cuda.is_available()

    @staticmethod
    def npu_is_available():
        """Replace torch.npu.is_available() if available"""
        import torch
        try:
            if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
                return torch.npu.is_available()
            return False
        except ImportError:
            return False

    @staticmethod
    def get_device_module(device_name):
        """Replace getattr(torch, device_name)"""
        import torch
        try:
            return getattr(torch, device_name)
        except AttributeError:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
            return torch.cuda

    @staticmethod
    def cuda_memory_set_allocator_settings(settings):
        """Replace torch.cuda.memory._set_allocator_settings(settings)"""
        import torch
        torch.cuda.memory._set_allocator_settings(settings)

    # ==================== Memory Management ====================

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

    # ==================== Distributed Communication ====================

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
    def distributed_get_rank():
        """Replace torch.distributed.get_rank()"""
        import torch.distributed
        return torch.distributed.get_rank()

    @staticmethod
    def distributed_get_world_size(group=None):
        """Replace torch.distributed.get_world_size(group)"""
        import torch.distributed
        return torch.distributed.get_world_size(group=group)

    @staticmethod
    def distributed_get_global_rank(group, group_rank):
        """Replace torch.distributed.get_global_rank(group, group_rank)"""
        import torch.distributed
        return torch.distributed.get_global_rank(group=group, group_rank=group_rank)

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
    def distributed_broadcast_object_list(object_list, src, group=None):
        """Replace torch.distributed.broadcast_object_list(object_list, src, group)"""
        import torch.distributed
        torch.distributed.broadcast_object_list(object_list=object_list, src=src, group=group)

    @staticmethod
    def distributed_all_reduce(tensor, op=None, group=None, async_op=False):
        """Replace torch.distributed.all_reduce(tensor, op, group, async_op)"""
        import torch.distributed as dist
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)

    @staticmethod
    def distributed_all_gather(tensor_list, tensor, group=None, async_op=False):
        """Replace torch.distributed.all_gather(tensor_list, tensor, group, async_op)"""
        import torch.distributed
        torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)

    @staticmethod
    def distributed_all_gather_object(object_list, obj, group=None):
        """Replace torch.distributed.all_gather_object(object_list, obj, group)"""
        import torch.distributed
        torch.distributed.all_gather_object(object_list=object_list, obj=obj, group=group)

    @staticmethod
    def distributed_ReduceOp():
        """Replace torch.distributed.ReduceOp"""
        import torch.distributed as dist
        return dist.ReduceOp

    @staticmethod
    def broadcast_dict_tensor(tensors, src, group):
        """Replace broadcast_dict_tensor from verl.utils.torch_functional"""
        from verl.utils.torch_functional import broadcast_dict_tensor
        return broadcast_dict_tensor(tensors=tensors, src=src, group=group)

    # ==================== Device Mesh ====================

    @staticmethod
    def init_device_mesh(device_type, mesh_shape, mesh_dim_names):
        """Replace init_device_mesh from torch.distributed.device_mesh"""
        from torch.distributed.device_mesh import init_device_mesh
        return init_device_mesh(
            device_type=device_type,
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

    # ==================== Compilation ====================

    @staticmethod
    def compile(func, dynamic=False):
        """Replace torch.compile(func, dynamic)"""
        import torch
        return torch.compile(func, dynamic=dynamic)

    # ==================== Serialization ====================

    @staticmethod
    def serialization_add_safe_globals(classes):
        """Replace torch.serialization.add_safe_globals(classes)"""
        import torch
        torch.serialization.add_safe_globals(classes)


class BackendAdapter:
    """
    Base adapter class that defines the interface for backend adapters.
    All methods raise NotImplementedError and must be implemented by subclasses.
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
        raise NotImplementedError("initialize_model_parallel must be implemented by subclass")

    @staticmethod
    def get_tensor_model_parallel_rank():
        """Replace mpu.get_tensor_model_parallel_rank()"""
        raise NotImplementedError("get_tensor_model_parallel_rank must be implemented by subclass")

    @staticmethod
    def get_tensor_model_parallel_world_size():
        """Replace mpu.get_tensor_model_parallel_world_size()"""
        raise NotImplementedError("get_tensor_model_parallel_world_size must be implemented by subclass")

    @staticmethod
    def get_tensor_model_parallel_group():
        """Replace mpu.get_tensor_model_parallel_group()"""
        raise NotImplementedError("get_tensor_model_parallel_group must be implemented by subclass")

    @staticmethod
    def get_pipeline_model_parallel_rank():
        """Replace mpu.get_pipeline_model_parallel_rank()"""
        raise NotImplementedError("get_pipeline_model_parallel_rank must be implemented by subclass")

    @staticmethod
    def get_pipeline_model_parallel_world_size():
        """Replace mpu.get_pipeline_model_parallel_world_size()"""
        raise NotImplementedError("get_pipeline_model_parallel_world_size must be implemented by subclass")

    @staticmethod
    def get_pipeline_model_parallel_last_rank():
        """Replace mpu.get_pipeline_model_parallel_last_rank()"""
        raise NotImplementedError("get_pipeline_model_parallel_last_rank must be implemented by subclass")

    @staticmethod
    def get_pipeline_model_parallel_group():
        """Replace mpu.get_pipeline_model_parallel_group()"""
        raise NotImplementedError("get_pipeline_model_parallel_group must be implemented by subclass")

    @staticmethod
    def get_data_parallel_rank():
        """Replace mpu.get_data_parallel_rank()"""
        raise NotImplementedError("get_data_parallel_rank must be implemented by subclass")

    @staticmethod
    def get_data_parallel_world_size():
        """Replace mpu.get_data_parallel_world_size()"""
        raise NotImplementedError("get_data_parallel_world_size must be implemented by subclass")

    @staticmethod
    def get_data_parallel_group(with_context_parallel=False):
        """Replace mpu.get_data_parallel_group(with_context_parallel)"""
        raise NotImplementedError("get_data_parallel_group must be implemented by subclass")

    @staticmethod
    def get_context_parallel_rank():
        """Replace mpu.get_context_parallel_rank()"""
        raise NotImplementedError("get_context_parallel_rank must be implemented by subclass")

    @staticmethod
    def get_context_parallel_world_size():
        """Replace mpu.get_context_parallel_world_size()"""
        raise NotImplementedError("get_context_parallel_world_size must be implemented by subclass")

    @staticmethod
    def get_virtual_pipeline_model_parallel_world_size():
        """Replace mpu.get_virtual_pipeline_model_parallel_world_size()"""
        raise NotImplementedError("get_virtual_pipeline_model_parallel_world_size must be implemented by subclass")

    @staticmethod
    def set_virtual_pipeline_model_parallel_rank(rank):
        """Replace mpu.set_virtual_pipeline_model_parallel_rank(rank)"""
        raise NotImplementedError("set_virtual_pipeline_model_parallel_rank must be implemented by subclass")

    @staticmethod
    def is_pipeline_first_stage(ignore_virtual=False, vp_stage=None):
        """Replace mpu.is_pipeline_first_stage(ignore_virtual, vp_stage)"""
        raise NotImplementedError("is_pipeline_first_stage must be implemented by subclass")

    @staticmethod
    def is_pipeline_last_stage(ignore_virtual=False):
        """Replace mpu.is_pipeline_last_stage(ignore_virtual)"""
        raise NotImplementedError("is_pipeline_last_stage must be implemented by subclass")

    @staticmethod
    def get_pipeline_model_parallel_split_rank():
        """Replace mpu.get_pipeline_model_parallel_split_rank()"""
        raise NotImplementedError("get_pipeline_model_parallel_split_rank must be implemented by subclass")

    @staticmethod
    def is_pipeline_stage_before_split():
        """Replace mpu.is_pipeline_stage_before_split()"""
        raise NotImplementedError("is_pipeline_stage_before_split must be implemented by subclass")

    @staticmethod
    def is_pipeline_stage_after_split():
        """Replace mpu.is_pipeline_stage_after_split()"""
        raise NotImplementedError("is_pipeline_stage_after_split must be implemented by subclass")

    @staticmethod
    def get_expert_model_parallel_world_size():
        """Replace mpu.get_expert_model_parallel_world_size()"""
        raise NotImplementedError("get_expert_model_parallel_world_size must be implemented by subclass")

    @staticmethod
    def get_expert_model_parallel_rank():
        """Replace mpu.get_expert_model_parallel_rank()"""
        raise NotImplementedError("get_expert_model_parallel_rank must be implemented by subclass")

    @staticmethod
    def get_expert_tensor_parallel_world_size():
        """Replace mpu.get_expert_tensor_parallel_world_size()"""
        raise NotImplementedError("get_expert_tensor_parallel_world_size must be implemented by subclass")

    @staticmethod
    def get_expert_model_parallel_group():
        """Replace mpu.get_expert_model_parallel_group()"""
        raise NotImplementedError("get_expert_model_parallel_group must be implemented by subclass")

    @staticmethod
    def get_expert_tensor_parallel_group():
        """Replace mpu.get_expert_tensor_parallel_group()"""
        raise NotImplementedError("get_expert_tensor_parallel_group must be implemented by subclass")

    # ==================== Tensor Parallel ====================

    @staticmethod
    def model_parallel_cuda_manual_seed(seed):
        """Replace tensor_parallel.model_parallel_cuda_manual_seed(seed)"""
        raise NotImplementedError("model_parallel_cuda_manual_seed must be implemented by subclass")

    @staticmethod
    def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
        """Replace tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits, target)"""
        raise NotImplementedError("vocab_parallel_cross_entropy must be implemented by subclass")

    @staticmethod
    def set_defaults_if_not_set_tensor_model_parallel_attributes(param):
        """Replace tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)"""
        raise NotImplementedError("set_defaults_if_not_set_tensor_model_parallel_attributes must be implemented by subclass")

    @staticmethod
    def get_cuda_rng_tracker():
        """Replace tensor_parallel.get_cuda_rng_tracker()"""
        raise NotImplementedError("get_cuda_rng_tracker must be implemented by subclass")

    # ==================== Pipeline Parallel ====================

    @staticmethod
    def get_forward_backward_func():
        """Replace get_forward_backward_func from megatron.core.pipeline_parallel"""
        raise NotImplementedError("get_forward_backward_func must be implemented by subclass")

    # ==================== Distributed Training ====================

    @staticmethod
    def finalize_model_grads(*args, **kwargs):
        """Replace finalize_model_grads from megatron.core.distributed"""
        raise NotImplementedError("finalize_model_grads must be implemented by subclass")

    @staticmethod
    def DistributedDataParallel(*args, **kwargs):
        """Replace DistributedDataParallel from megatron.core.distributed"""
        raise NotImplementedError("DistributedDataParallel must be implemented by subclass")

    @staticmethod
    def DistributedDataParallelConfig(*args, **kwargs):
        """Replace DistributedDataParallelConfig from megatron.core.distributed"""
        raise NotImplementedError("DistributedDataParallelConfig must be implemented by subclass")

    # ==================== Optimizer ====================

    @staticmethod
    def DistributedOptimizer(*args, **kwargs):
        """Replace DistributedOptimizer from megatron.core.optimizer"""
        raise NotImplementedError("DistributedOptimizer must be implemented by subclass")

    @staticmethod
    def OptimizerConfig(*args, **kwargs):
        """Replace OptimizerConfig from megatron.core.optimizer"""
        raise NotImplementedError("OptimizerConfig must be implemented by subclass")

    @staticmethod
    def get_megatron_optimizer_native(*args, **kwargs):
        """Replace get_megatron_optimizer from megatron.core.optimizer"""
        raise NotImplementedError("get_megatron_optimizer_native must be implemented by subclass")

    @staticmethod
    def ChainedOptimizer(*args, **kwargs):
        """Replace ChainedOptimizer from megatron.core.optimizer"""
        raise NotImplementedError("ChainedOptimizer must be implemented by subclass")

    @staticmethod
    def OptimizerParamScheduler(*args, **kwargs):
        """Replace OptimizerParamScheduler from megatron.core.optimizer_param_scheduler"""
        raise NotImplementedError("OptimizerParamScheduler must be implemented by subclass")

    # ==================== Model Configuration ====================

    @staticmethod
    def ModelParallelConfig(*args, **kwargs):
        """Replace ModelParallelConfig from megatron.core"""
        raise NotImplementedError("ModelParallelConfig must be implemented by subclass")

    @staticmethod
    def TransformerConfig(*args, **kwargs):
        """Replace TransformerConfig from megatron.core.transformer"""
        raise NotImplementedError("TransformerConfig must be implemented by subclass")

    @staticmethod
    def ModelType():
        """Replace ModelType from megatron.core.enums"""
        raise NotImplementedError("ModelType must be implemented by subclass")

    @staticmethod
    def AttnBackend():
        """Replace AttnBackend from megatron.core.transformer.enums"""
        raise NotImplementedError("AttnBackend must be implemented by subclass")

    # ==================== Model Classes ====================

    @staticmethod
    def GPTModel(*args, **kwargs):
        """Replace GPTModel from megatron.core.models.gpt.gpt_model"""
        raise NotImplementedError("GPTModel must be implemented by subclass")

    @staticmethod
    def Float16Module(*args, **kwargs):
        """Replace Float16Module from megatron.core.transformer.module"""
        raise NotImplementedError("Float16Module must be implemented by subclass")

    # ==================== Model Utilities ====================

    @staticmethod
    def get_attr_wrapped_model(*args, **kwargs):
        """Replace get_attr_wrapped_model from megatron.core.utils"""
        raise NotImplementedError("get_attr_wrapped_model must be implemented by subclass")

    # ==================== Checkpointing ====================

    @staticmethod
    def save_checkpoint(*args, **kwargs):
        """Replace dist_checkpointing.save from megatron.core.dist_checkpointing"""
        raise NotImplementedError("save_checkpoint must be implemented by subclass")

    @staticmethod
    def load_checkpoint(*args, **kwargs):
        """Replace dist_checkpointing.load from megatron.core.dist_checkpointing"""
        raise NotImplementedError("load_checkpoint must be implemented by subclass")

    @staticmethod
    def ShardedObject(*args, **kwargs):
        """Replace ShardedObject from megatron.core.dist_checkpointing.mapping"""
        raise NotImplementedError("ShardedObject must be implemented by subclass")

    # ==================== verl.models.mcore ====================

    @staticmethod
    def hf_to_mcore_config(hf_config, dtype, **override_transformer_config_kwargs):
        """Replace hf_to_mcore_config from verl.models.mcore"""
        raise NotImplementedError("hf_to_mcore_config must be implemented by subclass")

    @staticmethod
    def mapping_string_to_attn_backend(args):
        """Replace mapping_string_to_attn_backend from verl.models.mcore.config_converter"""
        raise NotImplementedError("mapping_string_to_attn_backend must be implemented by subclass")

    @staticmethod
    def McoreModuleWrapperConfig(*args, **kwargs):
        """Replace McoreModuleWrapperConfig from verl.utils.mindspore_utils"""
        raise NotImplementedError("McoreModuleWrapperConfig must be implemented by subclass")

    @staticmethod
    def make_megatron_module(*args, **kwargs):
        """Replace make_megatron_module from verl.utils.mindspore_utils"""
        raise NotImplementedError("make_megatron_module must be implemented by subclass")

    @staticmethod
    def gptmodel_forward_1f1b_overlap(*args, **kwargs):
        """Replace gptmodel_forward_1f1b_overlap from verl.models.mcore.model_forward_1f1b_overlap"""
        raise NotImplementedError("gptmodel_forward_1f1b_overlap must be implemented by subclass")

    @staticmethod
    def get_mcore_forward_fn(hf_config):
        """Replace get_mcore_forward_fn from verl.models.mcore"""
        raise NotImplementedError("get_mcore_forward_fn must be implemented by subclass")

    @staticmethod
    def get_mcore_forward_fused_fn(hf_config):
        """Replace get_mcore_forward_fused_fn from verl.models.mcore"""
        raise NotImplementedError("get_mcore_forward_fused_fn must be implemented by subclass")


class MegatronAdapter(BackendAdapter):
    """
    Adapter class that extracts and encapsulates all megatron interfaces.
    All methods are placeholders that should be implemented to call the actual megatron APIs.
    This adapter will be replaced with MindFormers interfaces in the future.
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
    def get_tensor_model_parallel_world_size():
        """Replace mpu.get_tensor_model_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_tensor_model_parallel_world_size()

    @staticmethod
    def get_tensor_model_parallel_group():
        """Replace mpu.get_tensor_model_parallel_group()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_tensor_model_parallel_group()

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
    def get_data_parallel_group(with_context_parallel=False):
        """Replace mpu.get_data_parallel_group(with_context_parallel)"""
        from megatron.core import parallel_state as mpu
        return mpu.get_data_parallel_group(with_context_parallel=with_context_parallel)

    @staticmethod
    def get_context_parallel_rank():
        """Replace mpu.get_context_parallel_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_context_parallel_rank()

    @staticmethod
    def get_context_parallel_world_size():
        """Replace mpu.get_context_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_context_parallel_world_size()

    @staticmethod
    def get_virtual_pipeline_model_parallel_world_size():
        """Replace mpu.get_virtual_pipeline_model_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_virtual_pipeline_model_parallel_world_size()

    @staticmethod
    def set_virtual_pipeline_model_parallel_rank(rank):
        """Replace mpu.set_virtual_pipeline_model_parallel_rank(rank)"""
        from megatron.core import parallel_state as mpu
        mpu.set_virtual_pipeline_model_parallel_rank(rank)

    @staticmethod
    def is_pipeline_first_stage(ignore_virtual=False, vp_stage=None):
        """Replace mpu.is_pipeline_first_stage(ignore_virtual, vp_stage)"""
        from megatron.core import parallel_state as mpu
        if vp_stage is not None:
            return mpu.is_pipeline_first_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
        return mpu.is_pipeline_first_stage(ignore_virtual=ignore_virtual)

    @staticmethod
    def is_pipeline_last_stage(ignore_virtual=False):
        """Replace mpu.is_pipeline_last_stage(ignore_virtual)"""
        from megatron.core import parallel_state as mpu
        return mpu.is_pipeline_last_stage(ignore_virtual=ignore_virtual)

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

    @staticmethod
    def get_expert_model_parallel_world_size():
        """Replace mpu.get_expert_model_parallel_world_size()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_expert_model_parallel_world_size()

    @staticmethod
    def get_expert_model_parallel_rank():
        """Replace mpu.get_expert_model_parallel_rank()"""
        from megatron.core import parallel_state as mpu
        return mpu.get_expert_model_parallel_rank()

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

    # ==================== Tensor Parallel ====================

    @staticmethod
    def model_parallel_cuda_manual_seed(seed):
        """Replace tensor_parallel.model_parallel_cuda_manual_seed(seed)"""
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)

    @staticmethod
    def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
        """Replace tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits, target)"""
        from megatron.core import tensor_parallel
        return tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=vocab_parallel_logits, target=target)

    @staticmethod
    def set_defaults_if_not_set_tensor_model_parallel_attributes(param):
        """Replace tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)"""
        from megatron.core import tensor_parallel
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    @staticmethod
    def get_cuda_rng_tracker():
        """Replace tensor_parallel.get_cuda_rng_tracker()"""
        from megatron.core import tensor_parallel
        return tensor_parallel.get_cuda_rng_tracker()

    # ==================== Pipeline Parallel ====================

    @staticmethod
    def get_forward_backward_func():
        """Replace get_forward_backward_func from megatron.core.pipeline_parallel"""
        from megatron.core.pipeline_parallel import get_forward_backward_func
        return get_forward_backward_func()

    # ==================== Distributed Training ====================

    @staticmethod
    def finalize_model_grads(*args, **kwargs):
        """Replace finalize_model_grads from megatron.core.distributed"""
        from megatron.core.distributed import finalize_model_grads
        return finalize_model_grads(*args, **kwargs)

    @staticmethod
    def DistributedDataParallel(*args, **kwargs):
        """Replace DistributedDataParallel from megatron.core.distributed"""
        from megatron.core.distributed import DistributedDataParallel
        return DistributedDataParallel(*args, **kwargs)

    @staticmethod
    def DistributedDataParallelConfig(*args, **kwargs):
        """Replace DistributedDataParallelConfig from megatron.core.distributed"""
        from megatron.core.distributed import DistributedDataParallelConfig
        return DistributedDataParallelConfig(*args, **kwargs)

    # ==================== Optimizer ====================

    @staticmethod
    def DistributedOptimizer(*args, **kwargs):
        """Replace DistributedOptimizer from megatron.core.optimizer"""
        from megatron.core.optimizer import DistributedOptimizer
        return DistributedOptimizer(*args, **kwargs)

    @staticmethod
    def OptimizerConfig(*args, **kwargs):
        """Replace OptimizerConfig from megatron.core.optimizer"""
        from megatron.core.optimizer import OptimizerConfig
        return OptimizerConfig

    @staticmethod
    def get_megatron_optimizer_native(*args, **kwargs):
        """Replace get_megatron_optimizer from megatron.core.optimizer"""
        from megatron.core.optimizer import get_megatron_optimizer
        return get_megatron_optimizer

    @staticmethod
    def ChainedOptimizer(*args, **kwargs):
        """Replace ChainedOptimizer from megatron.core.optimizer"""
        from megatron.core.optimizer import ChainedOptimizer
        return ChainedOptimizer(*args, **kwargs)

    @staticmethod
    def OptimizerParamScheduler(*args, **kwargs):
        """Replace OptimizerParamScheduler from megatron.core.optimizer_param_scheduler"""
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
        return OptimizerParamScheduler

    # ==================== Model Configuration ====================

    @staticmethod
    def ModelParallelConfig(*args, **kwargs):
        """Replace ModelParallelConfig from megatron.core"""
        from megatron.core import ModelParallelConfig
        return ModelParallelConfig(*args, **kwargs)

    @staticmethod
    def TransformerConfig(*args, **kwargs):
        """Replace TransformerConfig from megatron.core.transformer"""
        from megatron.core.transformer import TransformerConfig
        return TransformerConfig(*args, **kwargs)

    @staticmethod
    def ModelType():
        """Replace ModelType from megatron.core.enums"""
        from megatron.core.enums import ModelType
        return ModelType

    @staticmethod
    def AttnBackend():
        """Replace AttnBackend from megatron.core.transformer.enums"""
        from megatron.core.transformer.enums import AttnBackend
        return AttnBackend

    # ==================== Model Classes ====================

    @staticmethod
    def GPTModel(*args, **kwargs):
        """Replace GPTModel from megatron.core.models.gpt.gpt_model"""
        from megatron.core.models.gpt.gpt_model import GPTModel
        return GPTModel(*args, **kwargs)

    @staticmethod
    def Float16Module(*args, **kwargs):
        """Replace Float16Module from megatron.core.transformer.module"""
        from megatron.core.transformer.module import Float16Module
        return Float16Module(*args, **kwargs)

    # ==================== Model Utilities ====================

    @staticmethod
    def get_attr_wrapped_model(*args, **kwargs):
        """Replace get_attr_wrapped_model from megatron.core.utils"""
        from megatron.core.utils import get_attr_wrapped_model
        return get_attr_wrapped_model(*args, **kwargs)

    # ==================== Checkpointing ====================

    @staticmethod
    def save_checkpoint(*args, **kwargs):
        """Replace dist_checkpointing.save from megatron.core.dist_checkpointing"""
        from megatron.core import dist_checkpointing
        return dist_checkpointing.save(*args, **kwargs)

    @staticmethod
    def load_checkpoint(*args, **kwargs):
        """Replace dist_checkpointing.load from megatron.core.dist_checkpointing"""
        from megatron.core import dist_checkpointing
        return dist_checkpointing.load(*args, **kwargs)

    @staticmethod
    def ShardedObject(*args, **kwargs):
        """Replace ShardedObject from megatron.core.dist_checkpointing.mapping"""
        from megatron.core.dist_checkpointing.mapping import ShardedObject
        return ShardedObject(*args, **kwargs)

    # ==================== verl.models.mcore ====================

    @staticmethod
    def hf_to_mcore_config(hf_config, dtype, **override_transformer_config_kwargs):
        """Replace hf_to_mcore_config from verl.models.mcore"""
        from verl.models.mcore import hf_to_mcore_config
        return hf_to_mcore_config(hf_config, dtype, **override_transformer_config_kwargs)

    @staticmethod
    def mapping_string_to_attn_backend(args):
        """Replace mapping_string_to_attn_backend from verl.models.mcore.config_converter"""
        from verl.models.mcore.config_converter import mapping_string_to_attn_backend
        return mapping_string_to_attn_backend(args)

    @staticmethod
    def McoreModuleWrapperConfig(*args, **kwargs):
        """Replace McoreModuleWrapperConfig from verl.utils.mindspore_utils"""
        from verl.utils.mindspore_utils import McoreModuleWrapperConfig
        return McoreModuleWrapperConfig(*args, **kwargs)

    @staticmethod
    def make_megatron_module(*args, **kwargs):
        """Replace make_megatron_module from verl.utils.mindspore_utils"""
        from verl.utils.mindspore_utils import make_megatron_module
        return make_megatron_module(*args, **kwargs)

    @staticmethod
    def gptmodel_forward_1f1b_overlap(*args, **kwargs):
        """Replace gptmodel_forward_1f1b_overlap from verl.models.mcore.model_forward_1f1b_overlap"""
        from verl.models.mcore.model_forward_1f1b_overlap import gptmodel_forward_1f1b_overlap
        return gptmodel_forward_1f1b_overlap(*args, **kwargs)

    @staticmethod
    def get_mcore_forward_fn(hf_config):
        """Replace get_mcore_forward_fn from verl.models.mcore"""
        from verl.models.mcore import get_mcore_forward_fn
        return get_mcore_forward_fn(hf_config)

    @staticmethod
    def get_mcore_forward_fused_fn(hf_config):
        """Replace get_mcore_forward_fused_fn from verl.models.mcore"""
        from verl.models.mcore import get_mcore_forward_fused_fn
        return get_mcore_forward_fused_fn(hf_config)


class MindsporeAdapter(FrameWorkAdapter):
    """MindSpore adapter class that implements the interface for MindSpore."""
    pass

class MindFormersAdapter(FrameWorkAdapter):
    """MindFormers adapter class that implements the interface for MindFormers."""
    pass
