from typing import Any
# from typing import List, Tuple, Union
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
# from torch_geometric.utils.repeat import repeat
# import torch
# from torch_sparse import SparseTensor
from torch import Tensor
import math
# from torch.nn import Parameter


def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)
     
def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)
def zeros(value: Any):
    constant(value, 0.)
    
basestring = (str, bytes)
def is_list_of_strings(lst):
        if lst and isinstance(lst, list):
            return all(isinstance(elem, basestring) for elem in lst)
        else:
            return False
        

from typing import Tuple, Union, Optional
Adj = Union[Tensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]