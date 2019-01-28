import numpy as np
import random
import torch
from torchvision.utils import make_grid
from base import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, num_classes, losses, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, losses, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros((len(self.metrics), self.num_classes))
        for i, metric in enumerate(self.metrics):
            result = metric(output, target)
            for j in range(self.num_classes):
                acc_metrics[i][j] += result[j]
                self.writer.add_scalars(f'{metric.__name__}', {'class {}'.format(j): acc_metrics[i][j]})
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0
        total_metrics = np.zeros((len(self.metrics), self.num_classes))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            for i, metric in enumerate(self.losses):
                if i == 0:
                    loss = self.losses[i](output, target) * float(1.0 / len(self.losses))
                else:
                    loss += self.losses[i](output, target) * float(1.0 / len(self.losses))
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                '''
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                '''
                if data.dim() == 5:
                    slice_id = random.randint(0, data.size(2)-1)
                    pred = torch.argmax(output, dim=1, keepdim=True).float()
                    visual_label = torch.cat((target, pred), dim=4)
                    self.writer.add_image('input', make_grid(data[:, :, slice_id, :, :].cpu(), 
                                                             nrow=1, padding=5, normalize=True, scale_each=True))
                    self.writer.add_image('label', make_grid(visual_label[:, :, slice_id, :, :].cpu(), 
                                                             nrow=1, padding=5, normalize=True, range=(0, 3), scale_each=True))
                else:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=4, normalize=True))
            

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros((len(self.metrics), self.num_classes))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                for i, metric in enumerate(self.losses):
                    if i == 0:
                        loss = self.losses[i](output, target) * float(1.0 / len(self.losses))
                    else:
                        loss += self.losses[i](output, target) * float(1.0 / len(self.losses))

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                if data.dim() == 5:
                    slice_id = random.randint(0, data.size(2)-1)
                    pred = torch.argmax(output, dim=1, keepdim=True).float()
                    visual_label = torch.cat((target, pred), dim=4)
                    self.writer.add_image('input', make_grid(data[:, :, slice_id, :, :].cpu(), 
                                                             nrow=1, padding=5, normalize=True, scale_each=True))
                    self.writer.add_image('label', make_grid(visual_label[:, :, slice_id, :, :].cpu(), 
                                                             nrow=1, padding=5, normalize=True, range=(0, 3), scale_each=True))
                else:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=1, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }