from typing import Any, Callable, Dict

import torch
from torch.utils.data import Dataset


class BirdsetDataset(Dataset):
    def __init__(self,
                 transforms: Callable,

                 **kwargs):
        super().__init__()



class RandomDataset(Dataset):
    def __init__(
        self,
        data_path: str = None,
        transforms: Callable = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self) -> int:
        return 1000

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """return one training instance"""
        data = torch.rand(40)  # 40 datapoints
        label = torch.randint(low=0, high=1, size=(1,))  # for 1 class
        return {"input": data.float(), "label": label.float()}

