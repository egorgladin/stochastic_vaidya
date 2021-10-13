import torch
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    def __init__(self, data_dict, mode='train'):
        self.inp = data_dict[mode]['features']
        self.oup = data_dict[mode]['labels']

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])

        return {'inp': inpt, 'oup': oupt}
