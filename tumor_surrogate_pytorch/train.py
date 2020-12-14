import math

import torch
from torch.utils.tensorboard import SummaryWriter

from tumor_surrogate_pytorch.config import get_config
from tumor_surrogate_pytorch.data import TumorDataset
from tumor_surrogate_pytorch.model import TumorSurrogate
from tumor_surrogate_pytorch.utils import AverageMeter, loss_function, compute_dice_score, mean_absolute_error


class Trainer():

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir='runs/')
        self.save_path = config.save_path
        self.global_step = 0

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        T_total = self.config.max_epoch * nBatch
        T_cur = epoch * nBatch + batch
        lr = 0.5 * (self.config.lr_max - self.config.lr_min) * (1 + math.cos(math.pi * T_cur / T_total)) + self.config.lr_min
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self):
        train_dataset = TumorDataset(data_path=self.config.data_path, dataset='tumor_mparam/v/')
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.train_batch_size,
                                                  num_workers=16, pin_memory=True, shuffle=True)

        nBatch = len(data_loader)

        net = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
        net = net.to(device=self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), self.config.lr_min, weight_decay=self.config.weight_decay, betas=(self.config.beta1, self.config.beta2)
        )

        for epoch in range(self.config.max_epoch):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            save_frequency = 1
            validation_frequency = 1
            losses = AverageMeter()
            mae = AverageMeter()
            dice_score = AverageMeter()

            # switch to train mode
            net.train()

            for i, (input, parameters, ground_truth) in enumerate(data_loader):
                # lr
                self.adjust_learning_rate(optimizer, epoch, batch=i, nBatch=nBatch)
                # train weight parameters if not fix_net_weights
                input, parameters, ground_truth = input.to(self.device), parameters.to(self.device), ground_truth.to(self.device)

                output = net(input, parameters)  # forward (DataParallel)
                # loss
                loss = loss_function(u_sim=ground_truth, u_pred=output, csf=input[:, 2:3])
                # measure accuracy and record loss
                mae_wm_value, mae_gm_value, mae_csf_value = mean_absolute_error(ground_truth=ground_truth, output=output, input=input)
                if mae_wm_value is not None and mae_gm_value is not None and mae_csf_value is not None:
                    mae_mean_value = (mae_wm_value.item() + mae_gm_value.item() + mae_csf_value.item()) / 3
                    mae.update(mae_mean_value, input.size(0))
                    self.writer.add_scalar("Mae/train", mae_mean_value, self.global_step)

                dice = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.4)
                losses.update(loss, input.size(0))
                if dice is not None:
                    dice_score.update(dice, input.size(0))
                    self.writer.add_scalar("Dice/train", dice.item(), self.global_step)

                # tensorboard logging
                self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
                self.writer.flush()
                self.global_step += 1

                # compute gradient and do SGD step
                net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                optimizer.step()  # update weight parameters

                if i % save_frequency == 0:
                    # save model
                    torch.save(net.state_dict(), self.save_path)

            # validate
            if (epoch + 1) % validation_frequency == 0:
                val_loss, val_mae, val_dice = self.validate(net=net, writer=self.writer)

                # tensorboard logging
                self.writer.add_scalar("Loss/train-val", val_loss, self.global_step)
                self.writer.add_scalar("Mae/train-val", val_mae, self.global_step)
                self.writer.add_scalar("Dice/train-val", val_dice, self.global_step)
                self.writer.flush()

    def validate(self, net, writer=None, step=0):
        valid_dataset = TumorDataset(data_dir=self.config.data_path, dataset='valid/')
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.config.val_batch_size, num_workers=16, pin_memory=True,
                                                        shuffle=True)

        net.eval()

        losses = AverageMeter()
        mae = AverageMeter()
        dice_score_avg = AverageMeter()
        dice_score_02 = []
        dice_score_04 = []
        dice_score_08 = []

        mae_wm = []
        mae_gm = []
        mae_csf = []

        with torch.no_grad():
            ground_truths = []
            outputs = []
            for i, (input_batch, parameters, ground_truth_batch) in enumerate(valid_data_loader):
                input_batch, parameters, ground_truth_batch = input_batch.to(self.device), parameters.to(self.device), ground_truth_batch.to(
                    self.device)

                # compute output
                output_batch = net(input_batch, parameters)

                loss = loss_function(u_sim=ground_truth_batch, u_pred=output_batch, csf=input_batch[:, 2:3])
                losses.update(loss, input_batch.size(0))

                for output, ground_truth, input in zip(output_batch, ground_truth_batch, input_batch):
                    output, ground_truth, input = output[None, :], ground_truth[None, :], input[None, :]  # get batch dim back
                    if i == 0:
                        outputs.append(output)
                        ground_truths.append(ground_truth)
                    # measure mae, dice score and record loss
                    mae_wm_value, mae_gm_value, mae_csf_value = mean_absolute_error(ground_truth=ground_truth, output=output, input=input)
                    if mae_wm_value is None:
                        print("MAE was None, skipping this sample in validation")
                        continue

                    mae_wm.append(mae_wm_value)
                    mae_gm.append(mae_gm_value)
                    mae_csf.append(mae_csf_value)

                    mae_mean_value = (mae_wm_value.item() + mae_gm_value.item() + mae_csf_value.item()) / 3
                    dice_02 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.2)
                    dice_04 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.4)
                    dice_08 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.8)

                    mae.update(mae_mean_value, input.size(0))
                    if dice_04 is not None:
                        dice_score_avg.update(dice_04, input.size(0))
                        dice_score_04.append(dice_04.cpu().item())
                    if dice_02 is not None:
                        dice_score_02.append(dice_02.cpu().item())
                    if dice_08 is not None:
                        dice_score_08.append(dice_08.cpu().item())

            return losses.avg, mae.avg, dice_score_avg.avg


if __name__ == '__main__':
    config, unparsed = get_config()
    trainer = Trainer(config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    trainer.train()
