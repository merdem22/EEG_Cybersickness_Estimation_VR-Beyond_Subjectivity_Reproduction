import torch
import numpy as np
import torch.nn.functional as F
from torchutils.models import TrainerModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from helpers import to_numpy_input_tensors, detach_input_tensors, to_cpu_input_tensors, flatten_input_tensors
from torchutils.metrics import AverageScore
from metrics import leaky_accuracy

import torch.nn as nn

class _LambdaLoss(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, preds, targs):
        # keep your flattening behavior
        return self.fn(preds.flatten(), targs.flatten())


class MyTrainerModel(TrainerModel):
    def __init__(self, train_joy_mean, input_type, task, **kwds):
        if task == 'regression':
            compute_fn = self.compute_regression
            writable_scores = {'mse_loss', 'mean/l1_loss', 'mean/mse_loss', 
                               'leaky-acc@0.05', 'mean/leaky-acc@0.05',
                               'leaky-acc@0.10', 'mean/leaky-acc@0.10',
                               'leaky-acc@0.20', 'mean/leaky-acc@0.20'}
            joy_mean = float(train_joy_mean)
        elif task == 'classification':
            compute_fn = self.compute_classification
            writable_scores = {'accuracy@non-sick', 'accuracy@sick', 'accuracy@sick-5', 'f1_score',
                               'mean/accuracy@non-sick', 'mean/accuracy@sick', 'mean/accuracy@sick-5', 'mean/f1_score'}
            joy_mean = 0


        super().__init__(**kwds, writable_scores=writable_scores)
        if task == 'regression':
            fn = self.criterion
            self.criterion = _LambdaLoss(fn)
            writable_scores.add(self.criterion_name)
        self._buffer['getter_keys'] = ('eeg', 'psd', 'kinematic') if input_type == 'multi-segment' else ('kinematic',)
        self._buffer['lr'] = AverageScore(name='lr')
        self._buffer['mean'] = joy_mean
        self._buffer['compute_fn'] = compute_fn
        self._buffer['writable_scores'] = writable_scores
        self._buffer['acc'] = 0

    def writable_scores(self):
        return self._buffer['writable_scores']

    @torch.no_grad()
    @to_cpu_input_tensors
    @detach_input_tensors
    @to_numpy_input_tensors
    def compute_leaky_accuracy(self, preds, targets):
        #preds = preds.flatten() > 0.1
        #targets = targets.flatten() > 0.1
        means = np.full_like(preds, self._buffer['mean'])
        if np.any(targets > 0.0):
            for eps in [5e-2,1e-1,2e-1]:
                self.log_score(f'leaky-acc@{eps:3.2f}', float(leaky_accuracy(preds, targets, span=5, threshold=eps)[0]))
                self.log_score(f'mean/leaky-acc@{eps:3.2f}', float(leaky_accuracy(means, targets, span=5, threshold=eps)[0]))

    @torch.no_grad()
    @to_cpu_input_tensors
    @detach_input_tensors
    @to_numpy_input_tensors
    def compute_classification(self, preds, targets):
        #preds, targets = preds.argmax(1), targets.argmax(1)
        preds = preds > 0.1
        means = np.full_like(preds, self._buffer['mean'])

        cm_nnet = confusion_matrix(y_pred=preds, y_true=targets, labels=range(2))
        cm_mean = confusion_matrix(y_pred=means, y_true=targets, labels=range(2))

        classes = 'non-sick', 'sick'

        for ind, (acc1, acc2) in enumerate(zip(cm_nnet.diagonal(), cm_nnet.sum(1))):
            if acc2 != 0:
                self.log_score(f'accuracy@{classes[ind]}', float(acc1 / acc2))
                
        for ind, (acc1, acc2) in enumerate(zip(cm_mean.diagonal(), cm_mean.sum(1))):
            if acc2 != 0:
                self.log_score(f'mean/accuracy@{classes[ind]}', float(acc1 / acc2))
        
        self.log_score('f1_score', f1_score(y_pred=preds, y_true=targets, average='macro'))
        self.log_score('mean/f1_score', f1_score(y_pred=means, y_true=targets, average='macro'))

        if np.any(targets == 1):
            self.log_score('accuracy@sick-5', float(leaky_accuracy(preds, targets.flatten())))
            self.log_score('mean/accuracy@sick-5', float(leaky_accuracy(means, targets.flatten())))

    @torch.no_grad()
    @detach_input_tensors
    @flatten_input_tensors
    def compute_regression(self, preds, targets):
        means = torch.full_like(preds, self._buffer['mean'])
        self.log_score('mse_loss', F.mse_loss(preds, targets).item())
        self.log_score('mean/l1_loss', F.l1_loss(means, targets).item())
        self.log_score('mean/mse_loss', F.mse_loss(means, targets).item())

    @torch.no_grad()
    def integral(self, output, acc = 0):
        return torch.stack([output[:i+1].sum(0) for i in range(output.shape[0])]) + acc
    
    def model_output(self, batch):
        return self.model(**{k: v for k, v in batch.items() if k != 'observation'})
        input = zip(self._buffer['getter_keys'],
                    map(batch.__getitem__, self._buffer['getter_keys']))
        return self.model(**dict(input))

    
    def forward(self, batch, batch_idx=None):
        predictions = self.model_output(batch)
        loss = self.criterion(predictions, batch['observation'])
        self._buffer['lr'].update(self.optimizer.param_groups[0]['lr'])
        return predictions, loss
        
        losses = list()
        for pred, targ in zip(predictions, batch['observation']):
            losses.append(self.criterion(pred, targ))
            self._buffer['compute_fn'](pred, targ)
        #import pdb ;pdb.set_trace()
        #.flatten(0, 1)
        #targets = batch['observation'].flatten(0, 1)
        #import pdb; pdb.set_trace()
        #loss = self.criterion(predictions, targets)
        return predictions, sum(losses) / len(losses)
    
    @torch.no_grad()
    def forward_pass_on_evauluation_step(self, batch, batch_idx=None):
        predictions = self.model_output(batch)
        loss = self.criterion(predictions, batch['observation'])
        self._loss.update(loss.item())
        self._buffer['compute_fn'](predictions, batch['observation'])
        self.log(f"REGRESSION_OUTPUT: {predictions[:, 0].squeeze().tolist()}")
        #self.log(f"CLASSIFICATION_OUTPUT: {output_one_hot.argmax(-1).tolist()}")
        self.log(f"REGRESSION_TARGET: {batch['observation'][:, 0].squeeze().tolist()}")
        
        self.compute_leaky_accuracy(predictions, batch['observation'])

        return predictions
        
        losses = list()
        for pred, targ in zip(predictions, batch['observation']):
            losses.append(self.criterion(pred, targ))
            self._loss.update(losses[-1].item())
            self._buffer['compute_fn'](pred, targ)
            self.log(f"REGRESSION_OUTPUT: {pred[:, 0].squeeze().tolist()}")
            #self.log(f"CLASSIFICATION_OUTPUT: {output_one_hot.argmax(-1).tolist()}")
            self.log(f"REGRESSION_TARGET: {targ[:, 0].squeeze().tolist()}")

        #import pdb ;pdb.set_trace()
        #.flatten(0, 1)
        #targets = batch['observation'].flatten(0, 1)
        #import pdb; pdb.set_trace()
        #loss = self.criterion(predictions, targets)
        return predictions