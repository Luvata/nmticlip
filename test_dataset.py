import torch
import sys
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Tuple, Optional


class ClipDataset(Dataset):
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.captions_tokens[item],
            self.masks[item],
            self.prefixes[item],
        )

    def __init__(self, data_path: str, prefix_length: int):
        """
        Args:
            data_path: path to train.pkl, result of parse_viecap.py
            prefix_length:
        """
        self.prefix_length = prefix_length
        self.max_seq_len = 64
        dt = torch.load(data_path)
        sys.stdout.flush()
        self.captions_tokens = dt["target"]
        self.captions_tokens[self.captions_tokens.eq(1)] = 0
        self.prefixes = dt["clip_embedding"].float()
        self.masks = []
        for tokens in self.captions_tokens:
            # 5 is token <pad> in tokenizer
            mask = (tokens.greater(0)).long()
            mask = torch.cat((torch.ones(prefix_length), mask))
            self.masks.append(mask)

def test_ds(ds):
    print(len(ds))
    for c, m, f in ds:
        assert c.shape[0] == 64
        assert m.shape[0] == 74
        assert f.shape[0] == 512


prefix_length = 10
dataset1 = ClipDataset("./viecap_clean/train_sat_5k.pt", prefix_length)
dataset2 = ClipDataset("./viecap_clean/test_sat_1k.pt", prefix_length)
dataset3 = ClipDataset("./viecap_clean/train_viecap_5k.pt", prefix_length)
dataset4 = ClipDataset("./viecap_clean/test_viecap_1k.pt", prefix_length)


test_ds(dataset1)
test_ds(dataset2)
test_ds(dataset3)
test_ds(dataset4)
