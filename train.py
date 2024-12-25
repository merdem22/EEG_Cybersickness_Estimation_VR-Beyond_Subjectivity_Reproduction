from model import CombinedModel
from loader import EEGVideoDataset
from torch_runner.pipeline import train, train_step
import logging
import torch
import torch.nn.functional as F


dataset = EEGVideoDataset(
    root="/home/adhd/data/juliette-eeg/", patient=3, position="FN"
)


def custom_loss(y_pred, _, beta: float = 0.1):
    (
        eeg_cybersickness_level,
        cognitive_representation,
        vid_cybersickness_level,
        cognitive_features,
    ) = y_pred

    eeg_cybersickness_level = F.softmax(eeg_cybersickness_level, dim=1)
    vid_cybersickness_level = F.softmax(vid_cybersickness_level, dim=1)
    ce_loss = F.cross_entropy(eeg_cybersickness_level, vid_cybersickness_level)
    re_loss = F.mse_loss(cognitive_features, cognitive_representation)
    return (1 - beta) * ce_loss + beta * re_loss


ff = train(
    model=CombinedModel().cuda(),
    dataset=dataset,
    criterion=custom_loss,
    batch_size=2,
    num_epochs=10,
    handlers={logging.StreamHandler()},
)
