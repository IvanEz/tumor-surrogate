import math
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def loss_function(u_sim, u_pred, csf):
    pred_loss = torch.mean(torch.abs(u_sim[u_sim >= 0.001] - u_pred[u_sim >= 0.001]))
    csf_loss = torch.mean(torch.abs(u_sim[csf >= 0.001] - u_pred[csf >= 0.001]))

    if math.isnan(pred_loss.item()):
        pred_loss = 0
    loss = pred_loss + csf_loss
    return loss


def compute_dice_score(u_pred, u_sim, threshold):
    tp = torch.sum((u_pred > threshold) * (u_sim > threshold)).float()
    tpfp = torch.sum(u_pred > threshold).float()
    tpfn = torch.sum(u_sim > threshold).float()
    if tpfn + tpfp == 0:
        return None
    return torch.mean(2 * tp / (tpfn + tpfp))

def mean_absolute_error(ground_truth, output, input):
    ground_truth = np.array(ground_truth[:, 0].detach().cpu())
    output = np.array(output[:, 0].detach().cpu())
    input = input.cpu()
    wm = np.ma.masked_where(np.logical_and(input[:, 0] > 0.0001, ground_truth > 0.0001), input[:, 0])
    gm = np.ma.masked_where(np.logical_and(input[:, 1] > 0.0001, ground_truth > 0.0001), input[:, 1])
    csf = np.ma.masked_where(np.logical_and(input[:, 2] > 0.0001, output > 0.0001), input[:, 2])
    if wm.mask.sum() == 0 or gm.mask.sum() == 0 or csf.mask.sum() == 0:
        return None, None, None
    mae_wm = np.mean(np.abs(output[wm.mask].ravel() - ground_truth[wm.mask].ravel()))
    mae_gm = np.mean(np.abs(output[gm.mask].ravel() - ground_truth[gm.mask].ravel()))
    mae_csf = np.mean(np.abs(output[csf.mask].ravel() - ground_truth[csf.mask].ravel()))

    return mae_wm, mae_gm, mae_csf


def create_hists(model, val_loader, device, save_path):
    model.eval()
    mae_wm = []
    mae_gm = []
    mae_csf = []
    dice_score_02 = []
    dice_score_04 = []
    dice_score_08 = []
    losses = []

    with torch.no_grad():
        print("Dataloader lenght: ", len(val_loader))
        for i, (input_batch, parameters, ground_truth_batch) in enumerate(val_loader):
            print(f'iteration {i} of {len(val_loader)}')
            input_batch, parameters, ground_truth_batch = input_batch.to(device), parameters.to(device), ground_truth_batch.to(device)
            # compute output
            output_batch = model(input_batch, parameters)
            # measure mae, dice score and record loss
            loss = loss_function(u_sim=ground_truth_batch, u_pred=output_batch, csf=input_batch[:,2:3])
            print(loss)
            losses.append(loss.item())

            for output, ground_truth, input in zip(output_batch, ground_truth_batch, input_batch):
                output = output[None]
                ground_truth = ground_truth[None]
                input = input[None]
                # visualize input and output - very slow, only activate when needed
                # fig, axs = plt.subplots(1, 5, sharey=False)
                # wm = input[0,0, :, :, 32]
                # gm = input[0,1, :, :, 32]
                # csf = input[0,2, :, :, 32]
                # axs[0].imshow(wm.cpu())
                # axs[1].imshow(gm.cpu())
                # axs[2].imshow(csf.cpu())
                # axs[3].imshow(ground_truth[0,0, :, :, 32].cpu())
                # axs[4].imshow(output[0,0,:,:,32].cpu())
                # plt.show()


                dice_02 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.2)
                dice_04 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.4)
                dice_08 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.8)

                input = input.cpu()
                mae_wm_value, mae_gm_value, mae_csf_value = mean_absolute_error(ground_truth=ground_truth, output=output, input=input)
                if mae_wm_value is None:
                    continue

                mae_wm.append(mae_wm_value)
                mae_gm.append(mae_gm_value)
                mae_csf.append(mae_csf_value)

                if dice_02 is not None:
                    dice_score_02.append(dice_02.item())
                if dice_04 is not None:
                    dice_score_04.append(dice_04.item())
                if dice_08 is not None:
                    dice_score_08.append(dice_08.item())


    print(sum(losses)/len(losses))

    Path(save_path).mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(dice_score_02, bins=50)
    axs[1].hist(dice_score_04, bins=50)
    axs[2].hist(dice_score_08, bins=50)
    plt.savefig(save_path + 'dice.png')

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(mae_wm, bins=50)
    axs[1].hist(mae_gm, bins=50)
    axs[2].hist(mae_csf, bins=50)
    plt.savefig(save_path + 'mae.png')