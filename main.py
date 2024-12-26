import loader
import networks
import logging
import torch
import os
import numpy as np
import random
import argparse
from torchvision.models import get_model
from trainer import MyTrainerModel
from torchutils.trainer import Trainer
from torchutils.callbacks import AverageScoreLogger
from torchutils.callbacks import EarlyStopping



def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--patient', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'regression'])
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--input-type', type=str, default='multi-segment', choices=['kinematic', 'power-spectral-difference', 'power-spectral-no-eeg', 'power-spectral-no-kinematic'])
    parser.add_argument('--logprefix', type=str, default=None)
    parser.add_argument('--output',  action='store_true', default=False)
    parser.add_argument('--no-cuda',  action='store_true', default=False)
    parser.add_argument('--no-save-model',  action='store_true', default=False)
    parser.add_argument('--no-load-model',  action='store_true', default=False)

    params = parser.parse_args()
    params.patient = f'{params.patient:04d}'
    set_seed(params.seed)

    train_dataset, valid_dataset, test_datasets = loader.load_train_test_datasets(patient=params.patient, input_type=params.input_type, task=params.task, validation=True)
    device='cpu' if params.no_cuda else 'cuda'

    net = MyTrainerModel(
        model=f'{params.input_type}-model',
        task=params.task,
        n_channels=128, 
        input_type=params.input_type,
        hidden_size=32 if params.input_type == 'kinematic' else 64,
        criterion='binary_cross_entropy' if params.task == 'classification' else 'l1_loss',
        scheduler='ReduceLROnPlateau',
        optimizer='Adam', lr=4e-5, weight_decay=1e-6, device=device,
        num_classes=1 if params.task == 'classification' else 1,
        train_joy_mean=np.mean(train_dataset.train_mean)
    )

    trainer = Trainer(
        model=net,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        #train_dataloader_kwargs={'collate_fn': loader.collate_fn},
        #valid_dataloader_kwargs={'collate_fn': loader.collate_fn}
    )

    lgr = logging.getLogger(__name__)
    lgr.setLevel(logging.INFO)
    #os.sys.stderr = lgr.error

    stream_hdlr = logging.StreamHandler()
    stream_hdlr.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    stream_hdlr.setLevel(10)
    handlers = []

    if params.output:
        handlers.append(stream_hdlr)

    if params.logprefix is not None:
        os.makedirs(os.path.join(params.logprefix, 'logs'), exist_ok=True)
        file_hdlr = logging.FileHandler(os.path.join(params.logprefix, 'logs', f'{params.patient}-{params.seed}.log'), mode='w')
        file_hdlr.setFormatter(stream_hdlr.formatter)
        file_hdlr.setLevel(20)
        handlers.append(file_hdlr)

    for hdlr in handlers:
        lgr.addHandler(hdlr)

    #import pdb; pdb.set_trace()
    if params.logprefix is not None:
        model_path = os.path.join(params.logprefix, 'ckpt', f'{params.patient}-{params.seed}.ckpt')
        params.no_load_model = not os.path.isfile(model_path)
    else:
        params.no_save_model = True
        params.no_load_model = True

    trainer.compile(handlers=handlers)
    if not params.no_load_model:
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        trainer.train(num_epochs=params.num_epochs,
            batch_size=params.batch_size,
            callbacks=[EarlyStopping(monitor=net.criterion_name, goal='minimize', patience=15, delta=1e-4, verbose=30),
                        AverageScoreLogger(net.criterion_name, 'lr', level=20)],
            num_epochs_per_validation=10)
            
        if not params.no_save_model:
            os.makedirs(os.path.join(params.logprefix, 'ckpt'), exist_ok=True)
            torch.save(net.state_dict(), model_path)

    for test_name, test_dataset in test_datasets.items():
        if len(test_dataset) == 0:
            continue
        lgr.info(f'Evaluating: {test_name}')
        try:
            trainer.predict(test_dataset,
                            callbacks=[AverageScoreLogger(*net.writable_scores(), level=20)],
                            dataloader_kwargs={'batch_size': len(test_dataset)})
        except np.AxisError:
            lgr.error(f'Axis error in {test_name}')
