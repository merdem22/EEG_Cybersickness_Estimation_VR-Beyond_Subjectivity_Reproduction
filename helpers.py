import torch
import inspect


def detach_input_tensors(fn):
    def wrapped_fn(self, *args, **kwds):
        return fn(self, *map(torch.Tensor.detach,args), **kwds)
    return wrapped_fn


def flatten_input_tensors(fn):
    def wrapped_fn(self, *args, **kwds):
        return fn(self, *map(torch.Tensor.flatten,args), **kwds)
    return wrapped_fn


def to_numpy_input_tensors(fn):
    def wrapped_fn(self, *args, **kwds):
        return fn(self, *map(torch.Tensor.numpy,args), **kwds)
    return wrapped_fn


def to_cpu_input_tensors(fn):
    def wrapped_fn(self, *args, **kwds):
        return fn(self, *map(torch.Tensor.cpu,args), **kwds)
    return wrapped_fn


def to_cuda_input_tensors(fn):
    def wrapped_fn(self, *args, **kwds):
        return fn(self, *map(torch.Tensor.cuda,args),**kwds)
    return wrapped_fn


def to_tensor_input_arrays(fn):
    def wrapped_fn(self, *args, **kwds):
        return fn(self, *map(torch.from_numpy,args),**kwds)
    return wrapped_fn
