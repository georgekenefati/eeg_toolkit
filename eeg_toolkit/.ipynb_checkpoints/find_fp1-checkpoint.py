import os
import numpy as np
import mne
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels
from ._utils import *
from ._preprocess import *

def get_fp1_subs(data_path, sub_num):  
    """  
    Preprocess raw EDF data to filtered FIF format.  
    """  
    for folder in os.listdir(data_path):  
        if folder.startswith(sub_num):  
            sub_id = folder  
            save_fname_fif = sub_num[:3] + '_preprocessed-raw.fif'  
            print(sub_id, save_fname_fif)  
            break  

    eeg_data_raw_file = os.path.join(data_path, sub_id, next(subfile for subfile in os.listdir(os.path.join(data_path,sub_id)) if (subfile.endswith(('.edf','.EDF')))))  

    # read data, set EOG channel, and drop unused channels
    print(f"{sub_id}\nreading raw file...")
    raw = mne.io.read_raw_edf(eeg_data_raw_file)  

    Fp1_eog_flag=0
    # 32 channel case
    if 'X' in raw.ch_names and len(raw.ch_names)<64:  
        raw = load_raw_data(eeg_data_raw_file, 'Fp1')  
        Fp1_eog_flag=1

    # 64 channel case
    else:
        wrong_64_mtg_flag=0
        if "FT7" in raw.ch_names:
            wrong_64_mtg_flag=1
            eog_adj = 4
        else:
            eog_adj = 5
        if 'VEO' in raw.ch_names or 'VEOG' in raw.ch_names:  
            raw = load_raw_data(eeg_data_raw_file, 'VEO' if 'VEO' in raw.ch_names else 'VEOG')  
            non_eeg_chs = ['HEOG', 'EKG', 'EMG', 'Trigger'] if 'HEOG' in raw.ch_names else ['HEO', 'EKG', 'EMG', 'Trigger']  
            raw.drop_channels(non_eeg_chs)
            custom_montage = '../Misc/Montage/Hydro_Neo_Net_64_xyz_cms_No_FID.sfp'

        if 'X' in raw.ch_names:  
            raw = load_raw_data(eeg_data_raw_file, 'Fp1') 
            Fp1_eog_flag=1

    display.clear_output(wait=True)
    
    if Fp1_eog_flag==1:
        return sub_id
    else:
        pass