import glob
import os

import numpy as np
from scipy import signal

from pathlib import Path
from tqdm import tqdm

import spikeinterface as si
import spikeinterface.extractors as se

import one.alf.io as alfio

from utils import WindowGenerator, fscale, hp, rms

import argparse

def extract_continuous(sorting_folder,results_folder,
                       RMS_WIN_LENGTH_SECS = 3,
                       WELCH_WIN_LENGTH_SAMPLES=1025,
                       TOTAL_SECS = 100):
    session_folder = Path(str(sorting_folder).split('_sorted')[0])


    scratch_folder = Path('/scratch')

    ecephys_folder = session_folder / "ecephys_clipped"
    ecephys_compressed_folder = session_folder / 'ecephys_compressed'

    sorting_curated_folder = sorting_folder / "sorting_precurated"
    postprocessed_folder = sorting_folder / 'postprocessed'

    # extract stream names
    stream_names, stream_ids = se.get_neo_streams("openephys", ecephys_folder)

    neuropix_streams = [s for s in stream_names if 'Neuropix' in s]
    probe_names = [s.split('.')[1].split('-')[0] for s in neuropix_streams]

    for idx, stream_name in enumerate(neuropix_streams):

        print(stream_name)

        probe_name = probe_names[idx]

        output_folder = Path(results_folder) / probe_name

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if '-LFP' in stream_name:
            is_lfp = True
            np2 = False
            ap_stream_name = neuropix_streams[idx-1]
        elif '-AP' in stream_name:
            is_lfp = False
            ap_stream_name = stream_name
        else: # Neuropixels 2.0
            is_lfp = True
            ap_stream_name = stream_name

        waveform_folder = postprocessed_folder / f'experiment1_{ap_stream_name}_recording1'

        print(waveform_folder)

        we_recless = si.load_waveforms(waveform_folder, 
                                       with_recording=False)

        channel_inds = np.array([int(name[2:])-1 for name in we_recless.channel_ids])

        recording = si.read_zarr(ecephys_compressed_folder / f"experiment1_{stream_name}.zarr")

        print(f'Stream sample rate: {recording.sampling_frequency}')

        rms_win_length_samples = 2 ** np.ceil(np.log2(recording.sampling_frequency * RMS_WIN_LENGTH_SECS))
        total_samples = int(np.min([recording.sampling_frequency * TOTAL_SECS, recording.get_num_samples()]))

        # the window generator will generates window indices
        wingen = WindowGenerator(ns=total_samples, nswin=rms_win_length_samples, overlap=0)

        win = {'TRMS': np.zeros((wingen.nwin, recording.get_num_channels())),
               'nsamples': np.zeros((wingen.nwin,)),
               'fscale': fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / recording.sampling_frequency, one_sided=True),
               'tscale': wingen.tscale(fs=recording.sampling_frequency)}

        win['spectral_density'] = np.zeros((len(win['fscale']), recording.get_num_channels()))

        with tqdm(total=wingen.nwin) as pbar:

            for first, last in wingen.firstlast:

                D = recording.get_traces(start_frame=first, end_frame=last).T

                # remove low frequency noise below 1 Hz
                D = hp(D, 1 / recording.sampling_frequency, [0, 1])
                iw = wingen.iw
                win['TRMS'][iw, :] = rms(D)
                win['nsamples'][iw] = D.shape[1]

                # the last window may be smaller than what is needed for welch
                if last - first < WELCH_WIN_LENGTH_SAMPLES:
                    continue

                # compute a smoothed spectrum using welch method
                _, w = signal.welch(
                    D, fs=recording.sampling_frequency, window='hann', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                    detrend='constant', return_onesided=True, scaling='density', axis=-1
                )
                win['spectral_density'] += w.T
                # print at least every 20 windows
                if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
                    pbar.update(iw)

        win['TRMS'] = win['TRMS'][:,channel_inds]
        win['spectral_density'] = win['spectral_density'][:,channel_inds]

        if is_lfp:
            alf_object_time = f'ephysTimeRmsLF'
            alf_object_freq = f'ephysSpectralDensityLF'
        else:
            alf_object_time = f'ephysTimeRmsAP'
            alf_object_freq = f'ephysSpectralDensityAP'

        tdict = {'rms': win['TRMS'].astype(np.single), 'timestamps': win['tscale'].astype(np.single)}
        alfio.save_object_npy(output_folder, object=alf_object_time, dico=tdict, namespace='iblqc')

        fdict = {'power': win['spectral_density'].astype(np.single),
                 'freqs': win['fscale'].astype(np.single)}
        alfio.save_object_npy(
            output_folder, object=alf_object_freq, dico=fdict, namespace='iblqc')
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        dest = 'sorting_folder',
                        default = None,
                        help = 'Sorted Folder to use as baseline')
    args = parser.parse_args()
    
    
    # set up directories
    if args.sorting_folder is None:
        sorting_folder = Path(glob.glob('/data/ecephys_*sorted*')[0])
        results_folder = Path('/results/')
    else:
        sorting_folder = Path(os.path.join('/data/',args.sorting_folder))
        session_name = Path(str(sorting_folder).split('_sorted')[0]).name
        results_folder = Path('/results/')/session_name
        os.makedirs(results_folder,exist_ok=True)

    extract_continuous(sorting_folder,results_folder)