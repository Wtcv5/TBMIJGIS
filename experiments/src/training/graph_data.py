"""Utilities for batching graph-sequence training samples."""

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.graph.sequence import GraphSnapshot


class GraphSequenceDataset(Dataset):
    """Dataset of (graph_seq, monitoring_seq, target, chainage) samples."""

    def __init__(self,
                 graph_seqs: Sequence[Sequence[GraphSnapshot]],
                 monitoring_seqs,
                 targets,
                 chainages):
        if not (len(graph_seqs) == len(monitoring_seqs) == len(targets) == len(chainages)):
            raise ValueError("Graph-sequence sample inputs must have matching lengths.")

        self.graph_seqs = list(graph_seqs)
        self.monitoring_seqs = torch.as_tensor(monitoring_seqs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.chainages = torch.as_tensor(chainages, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[List[GraphSnapshot], torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            list(self.graph_seqs[idx]),
            self.monitoring_seqs[idx],
            self.targets[idx],
            self.chainages[idx],
        )


def collate_graph_sequence_batch(batch):
    """Collate graph sequences without trying to tensorize graph snapshots."""
    graph_seqs = [item[0] for item in batch]
    monitoring = torch.stack([item[1] for item in batch], dim=0)
    targets = torch.stack([item[2] for item in batch], dim=0)
    chainages = torch.stack([item[3] for item in batch], dim=0)
    return graph_seqs, monitoring, targets, chainages
