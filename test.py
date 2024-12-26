import torch
import torchutils.utils as mt
import torchutils.datasets as td

x = torch.rand(12).requires_grad_(True)


@mt.detach_input_tensors
@mt.to_numpy_input_tensors
@mt.from_numpy_input_arrays
@mt.to_cuda_input_tensors
def foo(a):
    print(a)


@td.register_builder('cifar')
def cifar_builder(**kwds):
    print(kwds)
    return 1, 2
    

foo(x)
