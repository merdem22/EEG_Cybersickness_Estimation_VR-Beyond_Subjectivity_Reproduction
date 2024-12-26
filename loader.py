import torch.utils.data as d
import numpy as np
import os
import scipy.signal as sig
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict


class Windows(d.Dataset):
    def __init__(self, windows):
        super().__init__()
        self.windows = tuple(w.astype(np.float32) for w in windows)

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        return np.expand_dims(self.windows[index].transpose(), 0)

class Dataset(d.Dataset):
    def __init__(self, train_mean=None, **args):
        super().__init__()
        self.args = args
        self.train_mean = train_mean
        self.map = lambda x: torch.from_numpy(x).cuda()

    def __len__(self):
        return len(next(iter(self.args.values())))
    
    def __getitem__(self, index):
        return {key: self.map(arg[index]) for key, arg in self.args.items()}


def collate_fn(batch):
    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        res = map(lambda seq: map(seq.__getitem__, keys), batch)
        return dict(zip(keys, zip(*res)))
    return tuple(zip(*batch))


def load_train_test_datasets(prefix='../../datasets/juliete/.cache', patient='0001', task='regression', input_type='multi-segment', validation=False):
    assert isinstance(task, str) and task in ['classification', 'regression']
    assert isinstance(input_type, str) #and input_type in ['multi-segment', 'kinematic', 'power-spectral-coeff', 'power-spectral-difference']
    assert isinstance(patient, str) and len(patient) == 4
    train_dataset = defaultdict(list)
    test_datasets = dict()

    for fname in os.listdir(prefix):
        data = np.load(os.path.join(prefix, fname), allow_pickle=True)['dataset'].item()
        #data['eeg'] = [(e - e.mean(0)) / (e.std(0) + 1e-5) for e in data['eeg']]
        data['eeg'] = data['eeg'].transpose(0, 2, 1)
        data['psd'] = data['psd'].transpose(0, 2, 1)

        kinematic = np.full((len(data['eeg']), 16, 30), np.nan)
        for idx, (tf, pth) in enumerate(zip(data['tf'], data['pth'])):
            if len(tf) < len(pth):
                tf = np.concatenate([tf, np.zeros((len(pth) - len(tf), 6))], axis=0)
            elif len(pth) < len(tf):
                pth = np.concatenate([pth, np.zeros((len(tf) - len(pth), 7))], axis=0)
            tf_diff = np.concatenate([np.diff(tf, axis=0), np.zeros((1, 6))], axis=0)
            lin = np.sqrt(np.square(pth[:,1:4]).sum(1, keepdims=True))
            ang = np.sqrt(np.square(pth[:,4:7]).sum(1, keepdims=True))
            
            pth_val = np.concatenate([lin, ang], axis=1)
            pth_diff = np.concatenate([np.diff(pth_val, axis=0), np.zeros((1, 2))], axis=0)
            
            knm = np.concatenate([tf, tf_diff, pth_val, pth_diff], axis=1)
            knm = sig.resample(knm, 30, axis=0, domain='time').astype(np.float32).transpose()
            kinematic[idx] = knm
            del knm
        
        seq = 5 # multi-segment seq length
        joystick_regression = np.stack(data['joy']).astype(np.float32).clip(0.0, 0.9)
        #joystick_summation = np.stack([joystick_regression[:idx+1].sum(0) for idx in range(len(joystick_regression))])
        #joystick_onehot = np.eye(3).astype(np.float32)[(joystick_regression // 0.3).astype(int).squeeze(1)]
        joystick_difference = joystick_regression - np.roll(joystick_regression, 1); joystick_difference[0, :] = 0

        joystick_erp = [joystick_difference[idx] - joystick_difference[:idx].max() for idx in range(1, len(joystick_difference))]
        joystick_erp = np.stack([[0]] + joystick_erp)
        joystick_erp[np.where(joystick_erp < 0.1)] = 0

        # joystick_onehot = np.zeros((len(joystick_difference), 3))
        # joystick_onehot[np.where(joystick_difference[:,0] > 0.1), 0] = 1
        # joystick_onehot[np.where(joystick_difference[:,0] < -0.1), 2] = 1
        # joystick_onehot[:, 1] = 1 - joystick_onehot.sum(axis=1)
        joystick_onehot = np.sign(joystick_erp).astype(np.float32)
        

        #if joystick_erp.sum() != 0: print(fname)

        if task == 'classification':
            train_output = joystick_onehot
            test_output = joystick_onehot
            dataset_mean = joystick_onehot.mean(0)
        else:
            output = [joystick_erp[:joy_idx + 1].max(keepdims=True) for joy_idx in range(joystick_erp.shape[0])]
            train_output = np.concatenate(output)
            test_output = np.concatenate(output)
            dataset_mean = np.concatenate(output).mean(0)

        if input_type == 'multi-segment':
            knm_seq = np.stack([np.roll(kinematic, -idx, axis=0) for idx in range(seq)], axis=1)[:-seq + 1]
            eeg_seq = np.stack([np.roll(data['eeg'], -idx, axis=0) for idx in range(seq)], axis=1)[:-seq + 1]
            psd_seq = np.stack([np.roll(data['psd'], -idx, axis=0) for idx in range(seq)], axis=1)[:-seq + 1]

            input = {
                'kinematic': knm_seq.astype(np.float32),
                'eeg': eeg_seq.astype(np.float32),
                'psd': psd_seq.astype(np.float32)
            }

            train_output = np.stack([np.roll(train_output, -idx, axis=0) for idx in range(seq)], axis=1)[:-seq + 1]
            test_output = np.stack([np.roll(test_output, -idx, axis=0) for idx in range(seq)], axis=1)[:-seq + 1]
        elif input_type == 'power-spectral-coeff':
            lin, quad = np.expand_dims(data['spectral_coeff'], 1).transpose(3, 1, 0, 2).astype(np.float32)
            input = dict(linear_coeffs=lin, quadratic_coeffs=quad)
            train_output = np.expand_dims(train_output, 0)
            test_output = np.expand_dims(test_output, 0)
        elif input_type == 'power-spectral-difference':
            power_spectra = np.log10(data['psd_raw'])
            ref_sequences = power_spectra[:3].mean(0, keepdims=True)

            psd_difference = [power_spectra[seq_idx:seq_idx+3].mean(0, keepdims=True) - ref_sequences
                              for seq_idx in range(power_spectra.shape[0])]
            psd_difference = np.concatenate(psd_difference, axis=0)

            input = dict(psd_difference=psd_difference.astype(np.float32),
                         kinematic=kinematic.astype(np.float32), 
                         train_mean=dataset_mean.repeat(psd_difference.shape[0]))
            dataset_mean = dataset_mean.repeat(psd_difference.shape[0])
            assert task == 'regression'
        elif input_type == 'power-spectral-no-eeg':
            input = dict(kinematic=kinematic.astype(np.float32), 
                         train_mean=dataset_mean.repeat(kinematic.shape[0]))
            dataset_mean = dataset_mean.repeat(kinematic.shape[0])
            assert task == 'regression'
        elif input_type == 'power-spectral-no-kinematic':
            power_spectra = np.log10(data['psd_raw'])
            ref_sequences = power_spectra[:3].mean(0, keepdims=True)

            psd_difference = [power_spectra[seq_idx:seq_idx+3].mean(0, keepdims=True) - ref_sequences
                              for seq_idx in range(power_spectra.shape[0])]
            psd_difference = np.concatenate(psd_difference, axis=0)

            input = dict(psd_difference=psd_difference.astype(np.float32),
                         train_mean=dataset_mean.repeat(psd_difference.shape[0]))
            dataset_mean = dataset_mean.repeat(psd_difference.shape[0])
            assert task == 'regression'
            
        elif input_type == 'kinematic':
            input = {
                'kinematic': np.stack(kinematic).astype(np.float32), 
                #'eeg': np.stack(data['eeg']).astype(np.float32),
                'train_mean': dataset_mean.repeat(kinematic.shape[0]).astype(np.float32)
            }
        else:
            raise AssertionError(f'{input_type} is invalid.')

        if fname[:4] == patient:
            test_dataset = defaultdict(list)
            for name, value in input.items():
                test_dataset[name].extend(value)
            test_dataset['observation'].extend(test_output)
            test_datasets[fname[:7]] = test_dataset

        else:
            for name, value in input.items():
                train_dataset[name].extend(value)
            train_dataset['observation'].extend(train_output)
                
    test_datasets = {patient_name: Dataset(**test_dataset) for patient_name, test_dataset in test_datasets.items()}
    
    if validation:
        keys = tuple(train_dataset.keys())
        splitted_items = train_test_split(*map(train_dataset.__getitem__, keys), shuffle=True, test_size=0.1, random_state=42)
        return Dataset(**dict(zip(keys, splitted_items[::2]))), Dataset(**dict(zip(keys, splitted_items[1::2]))), test_datasets
    
    return Dataset(**train_dataset), test_datasets, 

