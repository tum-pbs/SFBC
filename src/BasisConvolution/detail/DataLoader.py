import torch
from typing import List, Optional, Sequence, Union

from torch.utils.data import DataLoader

# class DataLoader(torch.utils.data.DataLoader):
#     r"""A data loader which merges data objects from a
#     :class:`torch_geometric.data.Dataset` to a mini-batch.
#     Data objects can be either of type :class:`~torch_geometric.data.Data` or
#     :class:`~torch_geometric.data.HeteroData`.

#     Args:
#         dataset (Dataset): The dataset from which to load the data.
#         batch_size (int, optional): How many samples per batch to load.
#             (default: :obj:`1`)
#         shuffle (bool, optional): If set to :obj:`True`, the data will be
#             reshuffled at every epoch. (default: :obj:`False`)
#         follow_batch (List[str], optional): Creates assignment batch
#             vectors for each key in the list. (default: :obj:`None`)
#         exclude_keys (List[str], optional): Will exclude each key in the
#             list. (default: :obj:`None`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch.utils.data.DataLoader`.
#     """
#     def __init__(
#         self,
#         dataset,
#         batch_size: int = 1,
#         shuffle: bool = False,
#         follow_batch: Optional[List[str]] = None,
#         exclude_keys: Optional[List[str]] = None,
#         **kwargs,
#     ):
#         # Remove for PyTorch Lightning:
#         kwargs.pop('collate_fn', None)

#         # Save for PyTorch Lightning < 1.6:
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys

#         super().__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             collate_fn=Collater(dataset, follow_batch, exclude_keys),
#             **kwargs,
#         )
