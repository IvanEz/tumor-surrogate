import os
import torch
from torch.utils.data import DataLoader

from tumor_surrogate_pytorch.config import get_config
from tumor_surrogate_pytorch.data import TumorDataset
from tumor_surrogate_pytorch.model import TumorSurrogate
from tumor_surrogate_pytorch.utils import create_hists


def load_weights(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def test(data_path):
    net = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device=device)

    if os.path.exists('./tumor_surrogate_pytorch/saved_model/model'):
        net = load_weights(net, path=config.save_path)
    else:
        raise Exception("No trained model exists. Please train a model before testing.")
    net.to(device=device)

    save_path = './tumor_surrogate-pytorch/test_output/'
    dataset = TumorDataset(data_path=data_path, dataset='valid/')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=32)
    create_hists(net, loader, device, save_path)


if __name__ == '__main__':
    config, unparsed = get_config()
    test(config.data_path)
