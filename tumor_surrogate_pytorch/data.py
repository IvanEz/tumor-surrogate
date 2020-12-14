import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TumorDataset(Dataset):
    def __init__(self, data_path, dataset):
        data_dir = os.path.join(data_path, dataset)
        self.data_list = sorted(glob.glob(data_dir + '*'))
        self.args = {}
        self.y_range = []
        self.y_num = []
        with open(os.path.join(data_path, 'tumor_mparam/args.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self.args[arg] = arg_value

        self.c_num = int(self.args['num_param'])
        for i in range(self.c_num):
            p_name = self.args['p%d' % i]
            p_min = float(self.args['min_{}'.format(p_name)])
            p_max = float(self.args['max_{}'.format(p_name)])
            p_num = int(self.args['num_{}'.format(p_name)])
            self.y_range.append([p_min, p_max])
            self.y_num.append(p_num)

    def __len__(self):
        return len(self.data_list)

    def crop(self, x, center_x, center_y, center_z):
        center_x = int(round(center_x * 128))
        center_y = int(round(center_y * 128))
        center_z = int(round(center_z * 128))
        return x[center_x - 32:center_x + 32,
               center_y - 32:center_y + 32,
               center_z - 32:center_z + 32]

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        x = data['x'][:, :, :, 1:]
        parameters = data['y']
        output = data['x'][:, :, :, 0:1]

        x = self.crop(x, parameters[3], parameters[4], parameters[5])
        output = self.crop(output, parameters[3], parameters[4], parameters[5])
        for i, ri in enumerate(self.y_range):
            parameters[i] = (parameters[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1

        x = torch.tensor(x).permute((3, 0, 1, 2)).float()
        parameters = torch.tensor(parameters).float()
        output = torch.tensor(output).permute((3, 0, 1, 2)).float()

        return x, torch.round(parameters[:3] * 10 ** 2) / 10 ** 2, output


if __name__ == '__main__':
    # path = '/mnt/Drive2/ivan/data/tumor_mparam/v/'
    data_dir = '/mnt/Drive2/ivan/data'
    dataset = 'tumor_mparam/v/' #or valid
    dataset = TumorDataset(data_dir, dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (x, y, z) in enumerate(loader):
        if i == 100:
            break
