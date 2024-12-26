from specparam import SpectralModel
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tqdm
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--start-segment', default=None)
    parser.add_argument('--eeg-channel', type=int, choices=range(24))

    params = parser.parse_args()

    assert params.start_segment is None or params.eeg_channel is not None, 'set --eeg-channel as well'

    fname = params.fname
    start_segment = int(params.start_segment) if params.start_segment is not None else None
    channel = int(params.eeg_channel) if params.eeg_channel is not None else None

    assert os.path.isfile(fname) and fname.endswith('.npz')
    data = np.load(fname, allow_pickle=True)['dataset'].item()

    fg = SpectralModel()
    #import pdb; pdb.set_trace()

    #fg.fit(data['freqs'], data['psd_raw'][0], [1, 40])
    #exps = fg.get_params('aperiodic_params', 'exponent') 

    # Fit the power spectrum model across all channels
    freq_range = [30, 50]
    idx_10hz = (data['freqs'] >= freq_range[0]).tolist().index(True)
    freqs = data['freqs']
    spectra_segments = np.log10(data['psd_raw'])

    joystick_data = data['joy'].squeeze(1)
    #fig, axes = plt.subplots(6, 4)
    prefix = os.path.split(fname)[1]

    os.makedirs(os.path.split(fname)[1], exist_ok=True)

    lin_coeff, quad_coeff = data['spectral_coeff'].transpose(2, 0, 1)

    CHANNEL_NAMES = 'P3,C3,F3,Fz,F4,C4,P4,Cz,Pz,A1,Fp1,Fp2,T3,T5,O1,O2,X3,X2,F7,F8,X1,A2,T6,T4'.split(',')
    fig, axes = plt.subplots(nrows=25, ncols=1, sharex=True, figsize=(18, 28))
    fig.set_tight_layout(True)

    for channel_idx in tqdm.tqdm(range(spectra_segments.shape[1])):
        A = list()
        Qa = list()
        Qb = list()
        Qc = list()

        if start_segment is not None and channel_idx != channel:
            continue

        for segment_idx in range(spectra_segments.shape[0]):

            a, b = np.polyfit(freqs[idx_10hz:], spectra_segments[segment_idx,channel_idx,idx_10hz:], deg=1)
            A.append(float(a))
            
            # aperiodic fit curve
            #fg.fit(data['freqs'], data['psd_raw'][segment_idx, channel_idx], freq_range)
            Qa.append(float(lin_coeff[segment_idx, channel_idx]))
            Qb.append(float(quad_coeff[segment_idx, channel_idx]))
            #print(segment_idx, channel_idx)

            #idx_45hz = (fg.freqs >= 45).tolist().index(True)
            #m = (fg._ap_fit[idx_45hz + 1] - fg._ap_fit[idx_45hz - 1]) / (fg.freqs[idx_45hz + 1] - fg.freqs[idx_45hz - 1])
            #n = - m * fg.freqs[idx_45hz] + fg._ap_fit[idx_45hz]
            #print(b, n)

            if start_segment is None or segment_idx < start_segment:
                continue

            # linear fit
            x = np.linspace(*freq_range, 10)

            plt.plot(fg.freqs, fg._ap_fit, label='aperiodic fit')

            plt.title(f'channel={channel_idx} segment={segment_idx} slope={float(a):.3f} intercept={float(b):.3f}')
            plt.plot(freqs[idx_10hz:], spectra_segments[segment_idx,channel_idx,idx_10hz:], label='log power spectra')
            
            plt.plot(x, a * x + b, label='linear fit')
            #plt.plot(x, m * x + n, label='linear fit 2')
            plt.legend()
            plt.ylim(-.75, 3.75)
            plt.savefig('fig.png')
            plt.cla()
            
            #import time; time.sleep(0.5)
            import pdb; pdb.set_trace()

        ax = axes[channel_idx]

        ax.plot(A, label=f'linear fit slope for Ch{channel_idx}={CHANNEL_NAMES[channel_idx]}')
        ###plt.plot(Qa, label='quadratic fit hessian')
        ax.plot(Qb, label=f'quadratic fit slope for Ch{channel_idx}={CHANNEL_NAMES[channel_idx]}')
        #plt.plot(Qc, label='quadratic fit constant')
        ax.set_ylim(-0.2, 0.2)
        #ax.legend()
        ax.set_ylabel(CHANNEL_NAMES[channel_idx])
    ax = axes[-1]
    ax.plot(joystick_data, label='joystick value', color='red')
    ax.set_ylim(-0.1, 1.1)
    #plt.legend()
    fig.suptitle(f'File {prefix} EEG Power Spectra')
    plt.savefig(os.path.join(prefix, f'{prefix[:-4]}_all-channels.png'))
    plt.cla()