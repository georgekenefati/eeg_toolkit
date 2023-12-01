import os
import numpy as np
import mne
import shutil
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels
from ._utils import *

RESAMPLE_FREQ = 400  

def load_raw_data(eeg_data_raw_file, eog):  
    """  
    Load raw EDF data with specified EOG channel.  
    """  
    return mne.io.read_raw_edf(eeg_data_raw_file, eog=[eog], preload=True)  

def set_montage(mne_obj, montage_path):  
    """  
    Set custom montage for Raw or Epochs object.  
    """  
    relative_path = os.path.join(os.path.dirname(__file__),montage_path)
    # relative_path = os.path.join('../../Lab code/wanglab_eeg/montages', montage_path)
    custom_montage = mne.channels.read_custom_montage(relative_path)  
    mne_obj.set_montage(custom_montage)  

def to_raw(data_path, sub_id, save_path):  
    """  
    Preprocess raw EDF data to filtered FIF format.  
    """  
    sub_id=''
    for folder in os.listdir(data_path):  
        if folder.startswith(sub_id):  
            sub_id = folder  
            save_fname_fif = sub_id[:3] + '_preprocessed-raw.fif'  
            print(sub_id, save_fname_fif)  
            break  
    
    eeg_data_raw_file = os.path.join(data_path, sub_id, next(subfile for subfile in os.listdir(os.path.join(data_path,sub_id)) if (subfile.endswith(('.edf', '.EDF')))))  
    
    # read data, set EOG channel, and drop unused channels
    print(f"{sub_id}\nreading raw file...")
    raw = mne.io.read_raw_edf(eeg_data_raw_file)  
    
    Fp1_eog_flag=0
    # 32 channel case
    if 'X' in raw.ch_names and len(raw.ch_names)<64:  
        raw = load_raw_data(eeg_data_raw_file, 'Fp1')  
        Fp1_eog_flag=1
        
        non_eeg_chs = ['X', 'Y', 'Z'] if 'X' in raw.ch_names else []  
        non_eeg_chs += ['Oth4'] if 'Oth4' in raw.ch_names else []  
    
        raw.drop_channels(non_eeg_chs)
        custom_montage = '../montages/Hydro_Neo_Net_32_xyz_cms_No_Fp1.sfp'
    
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
            custom_montage = '../montages/Hydro_Neo_Net_64_xyz_cms_No_FID.sfp'
    
        if "EEG66" in raw.ch_names:
            non_eeg_chs = ['EEG66','EEG67','EEG68','EEG69']
            raw.drop_channels(non_eeg_chs)
        
        # make adjustment for wrong montage subjects
        if wrong_64_mtg_flag:
            raw.drop_channels(['FT7','FT8','PO5','PO6']) # for subjects C24, 055, 056, 047
            custom_montage = '../montages/Hydro_Neo_Net_64_xyz_cms_No_FID_Caps.sfp' 
    
    # 007 and 010 had extremely noisy data near the ends of their recordings.
    # Crop it out.
    if sub_id=='007':
        raw = raw.crop(tmax=1483) 
    elif sub_id=='010':
        raw.crop(tmax=1997.8)
    
    print(f"{sub_id}\nsetting custom montage...")
    set_montage(raw,custom_montage)
    
    # high level inspection
    print(raw.ch_names)
    print(len(raw.ch_names))
                          
    # apply notch filter
    print(f"{sub_id}\napplying notch filter...")
    raw = raw.notch_filter(60., notch_widths=3)
    clear_display()
    
    # apply bandpass filter
    print(f"{sub_id}\napplying bandpass filter...")
    raw = raw.filter(l_freq=1., h_freq=100.)
    clear_display()
    
    # resample data to decrease file size
    print(f"{sub_id}\nresampling data from {raw.info['sfreq']} Hz to {RESAMPLE_FREQ} Hz...")
    raw.resample(RESAMPLE_FREQ, npad='auto')
    clear_display()
    
    # find bad channels automatically
    print(f"{sub_id}\nremoving bad channels...")
    raw_pyprep = NoisyChannels(raw, random_state=RANDOM_STATE)
    raw_pyprep.find_all_bads(ransac=False, channel_wise=False, max_chunk_size=None)
    raw.info['bads'] = raw_pyprep.get_bads()
    raw.interpolate_bads(reset_bads=True)
    clear_display()
    
    # re-reference channels
    print(f"{sub_id}\nre-referencing channels to average...")
    raw, _ = mne.set_eeg_reference(raw, ref_channels='average', copy=True)
    clear_display()
    
    # fit ICA 
    print(f"{sub_id}\nfitting ICA...")
    num_goods = len(raw.ch_names) - len(raw.info['bads']) - 1 # adjust for EOG
    ica = ICA(n_components=int(np.floor(num_goods/2)), random_state=RANDOM_STATE, max_iter='auto')
    ica.fit(raw)
    clear_display()
    
    # find EOG artifacts
    print(f"{sub_id}\nfinding EOG artifacts...")
    
    # ica.find_bads_eog is BROKEN. alt: exclude first two components
    try: 
        eog_indices, eog_scores = ica.find_bads_eog(raw,
                                                    threshold='auto')
        ica.exclude = eog_indices
    
    except:
        ica.exclude = [0,1]
    clear_display()
    
    # apply ICA
    print(f"{sub_id}\napplying ICA...")
    ica.apply(raw)
    clear_display()
    
    # save copy of data
    print(f"Saving processed data as '{save_fname_fif}'...")
    
    if 'VEO' in raw.ch_names:
        raw.drop_channels('VEO')
    elif 'VEOG' in raw.ch_names:
        raw.drop_channels('VEOG')
    elif Fp1_eog_flag:
        custom_montage = '../montages/Hydro_Neo_Net_32_xyz_cms_No_Fp1.sfp'
        set_montage(raw,custom_montage)
    
    raw.save(save_path+save_fname_fif, 
             verbose=True, overwrite=True)
    clear_display()
    
    # high level inspection
    print(raw.ch_names)
    print('\nNumber of remaining channels: ',len(raw.ch_names) - len(raw.info['bads']))
    print('\nDropped channels: ',raw.info['bads'])
    
    print("Raw data preprocessing complete.")
    
    clear_display()

    return raw

##############################################
from autoreject import AutoReject
from autoreject import get_rejection_threshold
import scipy.io as scio
import re
import openpyxl

def to_epo(raw, sub_id, data_path, save_path):  
    """  
    Preprocess the cleaned -raw.fif to epoched -epo.fif.  
    Removes noisy trials and trials with movement artifact.
    raw: gets raw from to_raw()
    save_path: save processed epoch info as .mat files
    """  
    # define functions for extracting relevant epochs
    def delete_multiple_elements(list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)## define functions for extracting relevant epochs
    
    def get_stim_epochs(epochs, val_list, key_list, events_from_annot_drop_repeats_list):
        for i in range( len(epochs)-1 ):
            # current epoch
            pre_curr_pos = val_list.index(events_from_annot_drop_repeats_list[i-1][-1]) # get position of epoch description value 
            pre_curr_key_str = key_list[pre_curr_pos] # get key at position (e.g., 'Yes Pain Hand')
            pre_curr_val = val_list[pre_curr_pos]
    
            curr_pos = val_list.index(events_from_annot_drop_repeats_list[i][-1]) # get position of epoch description value 
            curr_key_str = key_list[curr_pos] # get key at position (e.g., 'Yes Pain Hand')
            curr_val = val_list[curr_pos]
    
            next_pos = val_list.index(events_from_annot_drop_repeats_list[i+1][-1]) # get position of epoch description value 
            next_key_str = key_list[next_pos] # get key at position (e.g., 'Yes Pain Hand')
            next_val = val_list[next_pos]
    
            # for paradigms with NS, LS, HS pinprick keys_from_annot AND key presses
            if ((10 in val_list or 12 in val_list) and 3 in val_list):
                # print(0)
                if ( (curr_val in range(3,9)) and (next_val in range(10,14)) ):
                    # print('00')
                    StimOn_ids.append(i+1) # save pinprick marker
                    stim_labels.append(curr_key_str) # save label
                    key_to_pp_lag.append( (events_from_annot_drop_repeats_list[i+1][0] - events_from_annot_drop_repeats_list[i][0])*SAMPS_TO_MS )
                    # print(next_key_str)
    
                # check whether there are any pinprick keys_from_annot missing for some of the key presses
                elif ( (curr_val in range(3,9)) and (next_val not in range(10,14)) ):
                    # print(curr_val, next_val)
                    # print('01')
                    key_wo_pp_ids.append(i) # save pinprick marker
                    key_wo_pp_lbls.append(curr_key_str) # save label
                    key_wo_pp_samps_to_ms.append(events_from_annot_drop_repeats_list[i][0]*SAMPS_TO_MS)
                    # print(next_key_str)
    
            # for paradigms with NS and HS, no LS. Key presses but no pinprick keys_from_annot
            elif 10 not in val_list or 12 not in val_list:
                # print(1)
                if (curr_val in range(3,9)): #and curr_val <= 8 and curr_val != 4 and curr_val != 7:
                    # print('11')
                    StimOn_ids.append(i)
                    stim_labels.append(curr_key_str) # save label
                    # key_to_pp_lag.append( (events_from_annot_drop_repeats_list[i+1][0] - events_from_annot_drop_repeats_list[i][0])*SAMPS_TO_MS )
    
            # for data missing all key presses, but has pinprick keys_from_annot
            elif 3 not in val_list:
                # print(2)
                # if current is pinprick down and next is pinprick up within:
                # max_dur_stim = 2800 # milliseconds
                # min_dur_stim = 110 # milliseconds
                # # and if pinpricks occur at least gap_ITI apart:
                # gap_ITI = 2000 # milliseconds
    
                # if current is pinprick down and next is pinprick up within dur_stim:
                if ( (curr_val == 10 or curr_val == 12) and (next_val == 11 or next_val == 13)  and
                    # print('21')
                ((events_from_annot_drop_repeats_list[i+1][0] - events_from_annot_drop_repeats_list[i][0]) > min_dur_stim*MS_TO_SAMP ) and 
                ((events_from_annot_drop_repeats_list[i+1][0] - events_from_annot_drop_repeats_list[i][0]) < max_dur_stim*MS_TO_SAMP ) ): 
                # AND if last pinprick marker is greater than gap_ITI before current marker:
                   # ( (pre_curr_val in range(10,14)) and (curr_val in range(10,14))  and
                # ((events_from_annot_drop_repeats_list[i][0] - events_from_annot_drop_repeats_list[i-1][0]) > gap_ITI*MS_TO_SAMP ) ) ) :
                    StimOn_ids.append(i)
                    pp_updown_dur.append( (events_from_annot_drop_repeats_list[i+1][0] - events_from_annot_drop_repeats_list[i][0])*SAMPS_TO_MS )
                    ITI_stim_gap.append( (events_from_annot_drop_repeats_list[i][0] - events_from_annot_drop_repeats_list[i-1][0])*SAMPS_TO_MS )
        
        return stim_labels, StimOn_ids, key_wo_pp_ids, key_wo_pp_lbls, key_wo_pp_samps_to_ms, key_to_pp_lag, pp_updown_dur, ITI_stim_gap
    
    def labels_to_keys(txt_labels_list, val_list):
        stim_labels = [0]*len(txt_labels_list)
        if (10 in val_list or 11 in val_list or 12 in val_list or 13 in val_list):
            for i in range(0,len(stim_labels)):
                if txt_labels_list[i] in yes_hand_pain_list:
                        stim_labels[i] = 3
                elif txt_labels_list[i] in med_hand_pain_list:
                        stim_labels[i] = 4
                elif txt_labels_list[i] in no_hand_pain_list:
                        stim_labels[i] = 5
                elif txt_labels_list[i] in yes_back_pain_list:
                        stim_labels[i] = 6
                elif txt_labels_list[i] in med_back_pain_list:
                        stim_labels[i] = 7
                elif txt_labels_list[i] in no_back_pain_list:
                        stim_labels[i] = 8
                    
        else:
            for i in range(0,len(stim_labels)):
                if txt_labels_list[i] in yes_hand_pain_list:
                    stim_labels[i] = 3
                elif txt_labels_list[i] in no_hand_pain_list:
                    stim_labels[i] = 4
                elif txt_labels_list[i] in yes_back_pain_list:
                    stim_labels[i] = 5
                elif txt_labels_list[i] in no_back_pain_list:
                    stim_labels[i] = 6
    
        # extract only integer keys
        key_els = [num for num in stim_labels if isinstance(num, (int))]        
        
        return key_els
    
    ## define label dictionary for epoch annotations
    ## different keys account for different naming conventions
    ## NOTE: the Trigger#X naming convention does not specify between hand and back stimulus
    custom_mapping = {'eyes closed': 1, 'Trigger#1': 1, 'EYES CLOSED': 1, # eyes closed
                      'eyes open': 2, 'eyes opened': 2, 'Trigger#2': 2, 'EYES OPEN': 2, 'eyes openned':2, # eyes open
                      'pinprick hand': 3, 'hand pinprick': 3, 'Yes Pain Hand': 3, 'Trigger#3': 3, 'HAND PINPRICK': 3, 'hand 32 gauge pinprick': 3, 'Yes Hand Pain': 3, 'Hand YES Pain prick' : 3,
                      # highest intensity pain stimulus
                      'Med Pain Hand': 4 , 'Med Hand Pain': 4 , 'Hand Medium Pain prick' : 4, # intermediate intensity pain stimulus (HAND)
                      'No Pain Hand': 5, 'hand plastic':5, 'plastic hand': 5, 'Trigger#4': 5,  'HAND PLASTIC': 5, 'hand plastic filament':5, 'No Hand Pain': 5, 'Hand NO Pain' :5 ,
                      # sensory stimulus, no pain 
                      'pinprick back': 6, 'back pinprick':6, 'Yes Pain Back': 6, 'BACK  PINPRICK': 6, 'BACK PINPRICK': 6, 'Trigger#5': 6, 'back 32 gauge pinprick':6, 'Yes Back Pain': 6, 'Back YES Pain prick':6, 
                      # highest intensity pain stimulus (BACK)
                      'Med Pain Back': 7, 'Med Back Pain': 7, 'Back Medium Pain prick' : 7 , # intermediate intensity pain stimulus (BACK)
                      'plastic back': 8, 'back plastic':8, 'No Pain Back': 8, 'BACK PLASTIC': 8, 'Trigger#6': 8, 'back plastic filament': 8 , 'No Back Pain': 8, 'Back No Pain': 8,
                      # sensory stimulus, no pain (BACK)
                      'stop': 9, 'Stop': 9, 'STOP': 9, # stop
                      '1000001': 10, '100160': 10, '100480': 10, '1000000': 10, # lesser weight pen tip down
                      '1000010': 11, '100048': 11, # lesser weight pen tip up
                      '1100001': 12, '100320': 12, '1000000': 12, # greater weight pen tip down
                      '1100010': 13, # greater weight pen tip up
                      }
    
    # conversion factor for converting from given MS to SAMPLES
    MS_TO_SAMP = 400/1000 # e.g. 300 ms * (400 Hz / 1000 ms) = 120 samples
    SAMPS_TO_MS = 1000/400
    
    # Notes:
    # C5 has a glitch where the 100048 keys_from_annot are all at time 0, so there are 6 key press keys_from_annot missing PP because they are followed by a 100048 instead of a 100480.
    # C5 epoch correction sequence is (after 1 for starting correction): 1 0 1 1 1 1.

    # subject ID
    custom_mapping = {'eyes closed': 1, 'Trigger#1': 1, 'EYES CLOSED': 1, # eyes closed
                      'eyes open': 2, 'eyes opened': 2, 'Trigger#2': 2, 'EYES OPEN': 2, 'eyes openned':2, # eyes open
                      'pinprick hand': 3, 'hand pinprick': 3, 'Yes Pain Hand': 3, 'Trigger#3': 3, 'HAND PINPRICK': 3, 'hand 32 gauge pinprick': 3, 'Yes Hand Pain': 3, 'Hand YES Pain prick' : 3,
                      # highest intensity pain stimulus
                      'Med Pain Hand': 4 , 'Med Hand Pain': 4 , 'Hand Medium Pain prick' : 4, # intermediate intensity pain stimulus (HAND)
                      'No Pain Hand': 5, 'hand plastic':5, 'plastic hand': 5, 'Trigger#4': 5,  'HAND PLASTIC': 5, 'hand plastic filament':5, 'No Hand Pain': 5, 'Hand NO Pain' :5 ,
                      # sensory stimulus, no pain 
                      'pinprick back': 6, 'back pinprick':6, 'Yes Pain Back': 6, 'BACK  PINPRICK': 6, 'BACK PINPRICK': 6, 'Trigger#5': 6, 'back 32 gauge pinprick':6, 'Yes Back Pain': 6, 'Back YES Pain prick':6, 
                      # highest intensity pain stimulus (BACK)
                      'Med Pain Back': 7, 'Med Back Pain': 7, 'Back Medium Pain prick' : 7 , # intermediate intensity pain stimulus (BACK)
                      'plastic back': 8, 'back plastic':8, 'No Pain Back': 8, 'BACK PLASTIC': 8, 'Trigger#6': 8, 'back plastic filament': 8 , 'No Back Pain': 8, 'Back No Pain': 8,
                      # sensory stimulus, no pain (BACK)
                      'stop': 9, 'Stop': 9, 'STOP': 9, # stop
                      '1000001': 10, '100160': 10, '100480': 10, '1000000': 10, # lesser weight pen tip down
                      '1000010': 11, '100048': 11, # lesser weight pen tip up
                      '1100001': 12, '100320': 12, '1000000': 12, # greater weight pen tip down
                      '1100010': 13, # greater weight pen tip up
                      }

    # raw from arguments
    (events_from_annot, event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
    
    # get key and val lists from event_dict
    key_list = list(event_dict.keys())
    val_list = list(event_dict.values())
    
    raw
    
    
    # #### **ARE THERE ANY EPOCHS SHARING A SAMPLE INDEX WITH A KEYPRESS?**
    # #### *IF so, delete the issue event prior/after key-press before instantiating Epochs object*
    
    
    merged_flag = 0
    events_from_annot_new = events_from_annot.copy()
    for i in range(0,len(events_from_annot)-1):
        # if any consecutive events occur at the same sample, delete the one thats not 
        if events_from_annot[i][0] == events_from_annot[i+1][0] and events_from_annot[i][2] < 10 :
            merged_flag = 1
            print(f"Found merged epochs with labels {events_from_annot[i][2]} and {events_from_annot[i+1][2]}. Deleting epoch at index {i+1}.")
            print(f"{i}: {events_from_annot[i]}\n{i+1}: {events_from_annot [i+1]}")
            events_from_annot_new = np.delete(events_from_annot_new,i+1,axis=0)
            
        elif events_from_annot[i][0] == events_from_annot [i+1][0] and events_from_annot[i+1][2] < 10 :
            merged_flag = 1
            print(f"Found merged epochs with labels {events_from_annot[i][2]} and {events_from_annot[i+1][2]}. Deleting epoch at index {i}.")
            print(f"{i}: {events_from_annot[i]}\n{i+1}: {events_from_annot [i+1]}")
            events_from_annot_new = np.delete(events_from_annot_new,i,axis=0)
    
    if merged_flag:
        events_from_annot = events_from_annot_new
    
    
    # #### **ARE THERE ANY REPEATED KEYPRESS keys_from_annot WITHIN THE SAME SECOND OR TWO?**
    # #### *IF so, delete all prior issue events and just keep the last, then instantiate Epochs object*
    
    # conversion factor for converting from given MS to SAMPLES
    MS_TO_SAMP = 400/1000 # e.g. 300 ms * (400 Hz / 1000 ms) = 120 samples
    SAMPS_TO_MS = 1000/400
    repeated_flag = 0; repeated_count = 0
    events_from_annot_new = events_from_annot.copy()
    for i in range(0,len(events_from_annot_new)-1):
        # if any consecutive events occur at the same sample, delete the one thats not 
        if events_from_annot[i][2] in range(3,8) and events_from_annot[i+1][2] in range(3,8):
            # if the current and previous events have the same epoch and are less than a second apart BUT the following epoch is not less than a second apart:
            if ( ((events_from_annot[i+1][0] - events_from_annot[i][0]) < 1000*MS_TO_SAMP) and (events_from_annot[i][2] == events_from_annot[i+1][2]) and \
                (events_from_annot[i+2][2] == events_from_annot[i+1][2]) ): # and not \
                # ( (events_from_annot[i+2][0] - events_from_annot[i+1][0]) > 1000*MS_TO_SAMP ) ):
                
                repeated_flag = 1
                print(f"Found repeated key press ({events_from_annot[i][2]}) at index {i}. Deleting epoch at index {i} and keeping the following epoch.")
                print(f"{np.round(events_from_annot[i][0]*SAMPS_TO_MS/1000,2)}: {events_from_annot[i]}")
                print(f"{np.round(events_from_annot[i+1][0]*SAMPS_TO_MS/1000,2)}: {events_from_annot[i+1]}")
                # events_from_annot_new = np.delete(events_from_annot_new, i+1, axis=0)
                repeated_count += 1
                
    #         # else if 
    #         elif ( (events_from_annot[i+1][0] - events_from_annot[i][0]) < 1000*MS_TO_SAMP and \
    #                (events_from_annot[i][2] == events_from_annot[i+1][2]) and not \
    #                (events_from_annot[i+2][0] - events_from_annot[i+1][0]) > 1000*MS_TO_SAMP and \
    #                (events_from_annot[i+2][2] != events_from_annot[i+1][2]) ):
                
    if repeated_flag:
        events_from_annot = events_from_annot_new
        print(f"\nRemoved {repeated_count} extra keys_from_annot.")
    
    
    # Create initial epochs object with available keys_from_annot
    
    # create events to epoch-ize data
    
    # get key and val lists from event_dict
    key_list = list(event_dict.keys())
    val_list = list(event_dict.values())
    
    # create epochs object differently depending on paradigm
    if (10 in event_dict.values() or 12 in event_dict.values()):
        print(f"{sub_id}\nCreating epochs WITH key presses\n")
        epochs = mne.Epochs(raw, events_from_annot, event_dict, tmin=-0.2, tmax=0.8, proj=True, preload=True,
                            event_repeated='merge', baseline=(0,0))
    else:    
        # when we don't have key presses, let's assume that the key press is 200 ms before the pinprick, as the tmin for the first case ^
        print(f"{sub_id}\nCreating epochs WITHOUT key presses\n")
        epochs = mne.Epochs(raw, events_from_annot, event_dict, tmin=0.0, tmax=1.0, proj=True, preload=True,
                            event_repeated='merge', baseline=(0,0))
        
    # clear_display()
    
    epochs
    
    # del raw # clear memory
    

    
    # adjust events_from_annot for repeated events that are dropped by MNE
    print(f"{sub_id}\nRemoving repeated epochs from annotations...")
    epo_drop_arr = np.array(epochs.drop_log, dtype=object)
    repeated_ids = np.argwhere(epo_drop_arr)
    events_from_annot_drop_repeats_arr = np.delete(events_from_annot, repeated_ids, 0)
    events_from_annot_drop_repeats_list = events_from_annot_drop_repeats_arr.tolist()
    print(f"\nDropped {len(events_from_annot) - len(events_from_annot_drop_repeats_arr)} repeated epochs")
    # clear_display()
    
    
    # ##### Find stimulus events, labels, missing labels/samples, and PP/Stimulus Lags
    

    # find epochs only from stim events
    print(f"{sub_id}\nfinding the 60 pin-prick epochs...")
    
    # get lists for keys and values of event_dict
    key_list = list(event_dict.keys())
    val_list = list(event_dict.values())
    
    # intialize lists for epoch indices and labels
    stim_labels = []
    StimOn_ids  = []
    key_wo_pp_ids = []
    key_wo_pp_lbls = []
    key_wo_pp_samps_to_ms = []
    key_to_pp_lag = []
    pp_updown_dur = []
    ITI_stim_gap = [] # uncertain whether this is calculated well enough to output it
    
    # save only stimulus epochs
    stim_labels, StimOn_ids, key_wo_pp_ids, key_wo_pp_lbls, key_wo_pp_samps_to_ms, key_to_pp_lag, pp_updown_dur, _ = get_stim_epochs(epochs,val_list,key_list,events_from_annot_drop_repeats_list)
    
    stim_epochs = epochs[StimOn_ids]; del epochs
    
    if 3 not in val_list:
        print('LAGS BETWEEN PINPRICK UP AND DOWN:')
        print(pp_updown_dur)
        print(len(pp_updown_dur))
    
        if ITI_stim_gap:
            print('LAGS BETWEEN STIMULUS EVENTS:')
            print(ITI_stim_gap)
            print(len(ITI_stim_gap))
    
    stim_epochs
    
    

    # evoked = stim_epochs.average()
    # evoked.plot()
    
    
    # ##### Create label lists for each stimulus type

    # define labels in separate lists
    custom_mapping.keys()
    
    # HAND
    yes_hand_pain_list = list(custom_mapping.keys())[8:16]
    # print(yes_hand_pain_list)
    med_hand_pain_list = list(custom_mapping.keys())[16:19]
    # print(med_hand_pain_list)
    no_hand_pain_list = list(custom_mapping.keys())[19:27]
    # print(no_hand_pain_list)
    
    
    # BACK
    yes_back_pain_list = list(custom_mapping.keys())[27:36]
    # print(yes_back_pain_list)
    med_back_pain_list = list(custom_mapping.keys())[36:39]
    # print(med_back_pain_list)
    no_back_pain_list = list(custom_mapping.keys())[39:47]
    # print(no_back_pain_list)
    
    
    # ##### Create label array from annotations for comparison to ground truth (from Excel file)
    

    
    # change labels to keys
    
    keys_from_annot = labels_to_keys(stim_labels, val_list)
    
    print('STIMULUS INDICES FROM ALL EVENTS:')
    print(StimOn_ids)
    print('\n\tLENGTH:',len(StimOn_ids))
    
    
    print('\nCONVERTED KEYS FROM LABELS:')
    print(keys_from_annot)
    print('\n\tLENGTH:',len(keys_from_annot))
    
    if key_to_pp_lag:
        # print('\nLAGS BETWEEN KEY PRESSES AND PINPRICKS:')
        # print(key_to_pp_lag)
        from statistics import mean, stdev
        key_to_pp_lag_mean = mean(key_to_pp_lag)
        key_to_pp_lag_stdev = stdev(key_to_pp_lag)
    
        print(f'\n\tMean: {np.round(key_to_pp_lag_mean)} ms,  St Dev: {np.round(key_to_pp_lag_stdev)} ms')
    
    print('\n\tLENGTH:',len(key_to_pp_lag))
    
    print('\nLABELS OF KEY PRESS WITHOUT PINPRICKS:')
    print(key_wo_pp_lbls)
    
    print('\nINDICES OF KEY PRESS WITHOUT PINPRICKS:')
    print(key_wo_pp_ids)
    
    print('\nTIME STAMPS OF KEY PRESS WITHOUT PINPRICKS (in seconds):')
    print([ np.round((i/1000),1) for i in key_wo_pp_samps_to_ms])
    print('\n\tLENGTH:',len(key_wo_pp_samps_to_ms))
    
    
    # #### **IF there are missing pinpricks, correct here**

    if key_wo_pp_ids:
        key_wo_check = input("0 or 1: Is at least one of the missing PP(s) actually missing (instead of just a wrong/extra keypress)?")
        if key_wo_check == '1':
            print("Events with missing pinprick keys_from_annot:")
            for i in range(0, max(1,len(key_wo_pp_ids))):
                print(events_from_annot_drop_repeats_list[key_wo_pp_ids[i]] )
                needs_pp_adjustment = input("0 or 1: Does this key w/o pp event need correction?")
                if needs_pp_adjustment == '1':
                    # add 200ms*SAMPS_TO_MS to event time since PP marker missing
                    events_from_annot_drop_repeats_list[key_wo_pp_ids[i]][0] = int(events_from_annot_drop_repeats_list[key_wo_pp_ids[i]][0] + 200*MS_TO_SAMP)
                    print('New event after 200ms adjustment for missing pinprick:')
                    print( events_from_annot_drop_repeats_list[key_wo_pp_ids[i]] )
    
                    # append missing ID to StimOn_ids to be counted in stim_epochs
                    StimOn_ids.append(key_wo_pp_ids[i])
                # else:
                    # continue
            # replace contents of array with that of updated list
            events_from_annot_drop_repeats_arr = np.array(events_from_annot_drop_repeats_list)
            events_from_annot = events_from_annot_drop_repeats_arr
            # del epochs and make new object including events from missing PP
            del stim_epochs
            stim_epochs = mne.Epochs(raw, events_from_annot_drop_repeats_arr, event_dict, tmin=-0.2, tmax=0.8, 
                                        proj=True, preload=True, event_repeated='drop')#, baseline=(None,0))
    
            StimOn_ids.sort()
            stim_epochs = stim_epochs[StimOn_ids]
    
            stim_epochs
    else: print("\nNo need to account for missing PP's.")
    
    
    # ##### IF correcting missing PP's, add label(s) to keys_from_annot
    

    if key_wo_pp_ids:
        if key_wo_check == '1':
            events_with_key_wo_fixed = stim_epochs.events.tolist()
            events_with_key_wo_fixed.sort() # sort chronologically by samples
    
            # find added indices (those that are not pinprick keys_from_annot)
            idx_el_key_wo_tup = [(i,events_with_key_wo_fixed[i][2]) for i, el in enumerate(events_with_key_wo_fixed) if events_with_key_wo_fixed[i][2] < 10]
    
            for i in range(0,len(idx_el_key_wo_tup)):
                keys_from_annot.insert(idx_el_key_wo_tup[i][0], idx_el_key_wo_tup[i][1])
    
            print('Corrected keys_from_annot!\n')
            print(keys_from_annot)
            print('LENGTH:',len(keys_from_annot))
    
    
    # #### **Import stimulus and pain report information for the subject (from excel)**
    
    # import pain ratings to compare to annotations
    import pandas as pd
    
    for file in os.listdir(data_path):
        if file.startswith(sub_id):
            edf_dir = file
    
    xfname=""
    for file in os.listdir(data_path+edf_dir):
        if file.endswith(".xlsx"):
            xfname = file
    
    df = pd.read_excel((data_path+edf_dir+'/'+xfname), sheet_name=0) 
    
    lower_back_flag = 0
    try:
        if isinstance(df['Unnamed: 2'].tolist().index('LOWER BACK '), int):
            lower_back_flag = 1
            column_back_idx = df['Unnamed: 2'].tolist().index('LOWER BACK ')
            ground_truth_hand = df['PIN PRICK PAIN SCALE '][3:column_back_idx].tolist()
            
            pain_ratings_hand = df['Unnamed: 1'][3:column_back_idx].tolist()
            pain_ratings_back = df['Unnamed: 1'][column_back_idx:].tolist()
    
            if ' 32 guage (3) ' in ground_truth_hand or 'PM (4)' in ground_truth_hand:
                # index where rows switch to back
                ground_truth_hand_new = []
                for idx, el in enumerate(ground_truth_hand):
                    if el == ' 32 guage (3) ':
                        ground_truth_hand_new.append(3)
                    elif el == 'PM (4)':
                        ground_truth_hand_new.append(4)
                # back rows
                ground_truth_back = df['PIN PRICK PAIN SCALE '][column_back_idx:].tolist()
                ground_truth_back_new = []
                for idx, el in enumerate(ground_truth_back):
                    if el == ' 32 guage (3) ':
                        ground_truth_back_new.append(6)
                    elif el == 'PM (4)':
                        ground_truth_back_new.append(8)
        
    except: pass
    
    if lower_back_flag:
        # concatenate lists
        ground_truth = ground_truth_hand_new + ground_truth_back_new
        pain_ratings_lst = pain_ratings_hand + pain_ratings_back
    else:
        # concatenate lists
        ground_truth = df['PIN PRICK PAIN SCALE '][3:].tolist() 
        pain_ratings_lst = df['Unnamed: 1'][3:].tolist()
    
    # check if ground_truth contains nans which happens for some reason
    if np.any(np.isnan(ground_truth)):
        excel_nans = np.where(np.isnan(ground_truth))
        excel_nans = excel_nans[0].tolist()
        delete_multiple_elements(ground_truth,excel_nans)
        
        pain_nans = np.where(np.isnan(pain_ratings_lst))
        pain_nans = pain_nans[0].tolist()
        delete_multiple_elements(pain_ratings_lst,pain_nans)
        
    print(f"Loaded '{xfname}'!")
    

    
    print('FROM ANNOTATIONS:')
    print(keys_from_annot)
    print('LENGTH:',len(keys_from_annot))
    
    print('\nGROUND TRUTH STIMULUS KEYS:')
    print(ground_truth)
    print('LENGTH:',len(ground_truth))
    
    print('\nPAIN RATINGS:')
    print(pain_ratings_lst)
    print('LENGTH:',len(pain_ratings_lst))
    
    print('\nDo the lists match?')
    mtch_ans = 'Yes.' if ground_truth == keys_from_annot else 'No!'
    print(mtch_ans)
    
    
    # #### *IF missing back labels adjust here:*
    
    
    # adjustment for Trigger# keys_from_annot that require changing 3 and 4 to 5 and 6 for back keys_from_annot
    if {5,6} & set(keys_from_annot) == []:#[-5:-1]:
        back_switch_id = input("Enter the index at which pinpricks switch to the lower back: ")
        keys_from_annot_new = keys_from_annot.copy()
        for i in range(int(back_switch_id), len(keys_from_annot)):
            if keys_from_annot[i] == 3:
                keys_from_annot_new[i] = 6
            elif keys_from_annot[i] == 4:
                keys_from_annot_new[i] = 8
    
        # validate first,
        print(keys_from_annot_new)
        print(len(keys_from_annot_new))
    
        # then uncomment and allow overwrite
        keys_from_annot = keys_from_annot_new
    
    
    # #### Update old labeling convention
    
    # using loop for simplicity; if using masks, need to work in reverse from 6 to 4
    if not {7} & set(keys_from_annot):#[-5:-1]: # the absence of 8 confirms no LS
        for i,lbl in enumerate(keys_from_annot):
            if lbl==4: # 4 should be 5 (hand NS)
                keys_from_annot[i] = 5
            if lbl==5: # 5 should be 6 (back HS)
                keys_from_annot[i] = 6
            if lbl==6: # 6 should be 8 (back NS)
                keys_from_annot[i] = 8       
    
    # also do it for the ground_truth
    if not {7} & set(ground_truth):#[-5:-1]: # the absence of 8 confirms no LS
        for i,lbl in enumerate(ground_truth):
            if lbl==4: # 4 should be 5 (hand NS)
                ground_truth[i] = 5
            if lbl==5: # 5 should be 6 (back HS)
                ground_truth[i] = 6
            if lbl==6: # 6 should be 8 (back NS)
                ground_truth[i] = 8  
    
    print('FROM ANNOTATIONS:')
    print(keys_from_annot)
    print('LENGTH:',len(keys_from_annot))
    
    print('\nGROUND TRUTH STIMULUS KEYS:')
    print(ground_truth)
    print('LENGTH:',len(ground_truth))
    
    
    # #### ***IF lists don't match, check for missing, extra, or point errors:***
    # #### * *ONLY WORKS FOR SINGLE ERRORS* *
    # #### * *If more errors exist after an 'Unknown Error' code, they will not be reported* *
    

    # mismatch_list = []
    iss_ids = []
    
    if mtch_ans == 'No!':
        simple_issue_check = input("0 or 1: Does the issue appear to consist of single mismatches ONLY?\n")
        if simple_issue_check == '1':
            for i, el in enumerate(keys_from_annot):
                # find mismatch
                try:
                    mismatch = next( (idx, x, y) for idx, (x, y) in enumerate(zip(keys_from_annot, ground_truth)) if x!=y )
                except: 
                    print('No (more) mismatches found. Exiting loop.')
                    break
                else:
                    iss_i = mismatch[0]
                    if iss_i in iss_ids: continue
                    else:  
                        iss_ids.append( iss_i ); print(iss_i)
                        if keys_from_annot[iss_i+1:iss_i+1+2] == ground_truth[iss_i+1:iss_i+1+2]:
                            keys_from_annot[iss_i] = mismatch[2]
                            print(f'Point error mismatch: changed label {mismatch[1]} to {mismatch[2]} in annotations\n')
                        elif keys_from_annot[iss_i+1:iss_i+1+2] == ground_truth[iss_i:iss_i+2]:
                            del keys_from_annot[iss_i]; del StimOn_ids[iss_i]; stim_epochs.drop(iss_i)
                            print(f'Extra label mismatch: expected {mismatch[2]}, got {mismatch[1]}. Deleted label from annotations.\n')
                        elif keys_from_annot[iss_i+1:iss_i+1+2] == ground_truth[iss_i+1+1:iss_i+1+1+2]:
                            del ground_truth[iss_i]; del pain_ratings_lst[iss_i]
                            print(f'Missing label mismatch: expected {mismatch[2]} in keys, deleted trial {iss_i} from stim_labels and pain_ratings_lst.\n')
                        else:
                            print(f'Unknown error, check manually. **May be more than one marker missing/extra.\n')
                            # continue
                            # break
    
            print('\nAFTER CORRECTION:\n')
    
            print('FROM ANNOTATIONS:')
            print(keys_from_annot)
            print(len(keys_from_annot))
          
            print('GROUND TRUTH:')
            print(ground_truth)
            print(len(ground_truth))
            
            print('PAIN RATINGS:')
            print(pain_ratings_lst)
            print(len(pain_ratings_lst))
            
            
            print('\nDo the lists match now?')
            mtch_ans = 'Yes.' if ground_truth == keys_from_annot else 'No!'
            print(mtch_ans)
        else:
            print('\nPlease correct the issues manually using SigViewer, deleting epochs if necessary.')
    else: 
        print('Labels already match.')
    
    print("index\tkey-press\texcel-sheet")
    [print(f"{i} | {keys_from_annot[i]} {ground_truth[i]}   Y") if keys_from_annot[i]==ground_truth[i] else print(f"{i} | {keys_from_annot[i]} {ground_truth[i]}") for i in range(min(len(ground_truth),len(keys_from_annot))) ];

    print('\nDo the lists match now?')
    mtch_ans = 'Yes.' if ground_truth == keys_from_annot else 'No!'
    print(mtch_ans)
    
    ### Use this cell as a workspace if need to manually delete any epochs from stim_epochs and keys_from_annot:
    
    while mtch_ans!='Yes.':
        # what to delete
        drop_start = input('annotations drop START: ')
        if '.' not in drop_start and '/' not in drop_start and 'done' not in drop_start:
            drop_end = input('annotations drop END: ')
            drop_list = [*range(int(drop_start),int(drop_end)+1)]
        elif 'done' in drop_start:
            break
        elif '/' in drop_start:
            drop_list = [int(el) for el in drop_start.split('/')]
        elif drop_start == '.':
            drop_list = []
        
        # where to delete
        stim_epochs.drop(drop_list)
        delete_multiple_elements(StimOn_ids, drop_list)
        delete_multiple_elements(keys_from_annot, drop_list)
        
        print('len(epo_times):\t\t', len(StimOn_ids) )
        print('len(stim_labels):\t',len(keys_from_annot) )
        print('\nDo the lists match now?')
        mtch_ans = 'Yes.' if ground_truth == keys_from_annot else 'No!'
        print(mtch_ans)
        # if len(StimOn_ids) != len(keys_from_annot): raise
        
    clear_display()
        
    ###########################################
    # DROPPING FOR STIM LABELS AND PAIN RATINGS
    ###########################################
    while mtch_ans!='Yes.':
        gt_drop_start = input('gt & pain ratings drop START: ')
        if '.' not in gt_drop_start and '/' not in gt_drop_start and 'done' not in gt_drop_start:
            gt_drop_end = input('gt & pain ratings drop END: ')
            gt_drop = [*range(int(gt_drop_start),int(gt_drop_end)+1)]
        elif '/' in gt_drop_start:
            gt_drop = [int(el) for el in gt_drop_start.split('/')]
        elif gt_drop_start == '.':
            gt_drop = []
        elif 'done' in drop_start:
            break
   
        delete_multiple_elements(ground_truth, gt_drop)
        delete_multiple_elements(pain_ratings_lst, gt_drop)
        print('len(ground_truth):\t',len(ground_truth) )
    
        print('\nDo the lists match now?')
        mtch_ans = 'Yes.' if ground_truth == keys_from_annot else 'No!'
        print(mtch_ans)
        
    if mtch_ans!= 'Yes.':
        raise Exception("Lists do not match")
    
    
    # ## Complete preprocessing and save

    # verify stim_epochs object looks correct
    print('FINAL EPOCH COUNT:', len(stim_epochs))
    stim_epochs
    
    
    # #### Implement AutoReject, Save and Export

    # Final check
    epo_times = events_from_annot[StimOn_ids]
    
    print('len(stim_epochs):\t', len(stim_epochs))
    
    print("\nlen(ground_truth):\t",len(ground_truth))
    
    print("\nlen(pain_ratings_lst):\t",len(pain_ratings_lst))
    
    print('\nlen(epo_times):\t\t', len(epo_times))
    
    
    # stim_epochs.drop_channels('Fp1')
    

    # use autoreject package to automatically clean epochs 
    print(f"{sub_id}\nFinding epochs to clean...")
    ar = AutoReject(random_state=42)
    _, reject_log = ar.fit_transform(stim_epochs, return_log=True)
    print(reject_log)
    # clear_display()
    
    # drop rejected epochs
    bad_epochs_bool = reject_log.bad_epochs.tolist()
    dropped_epochs_list = [i for i, val in enumerate(bad_epochs_bool) if val]
    print(f'Dropped {len(dropped_epochs_list)} epochs: ', dropped_epochs_list)
    
    # save processed epochs
    print("\nSaving processed epochs...")
    
    save_fname = sub_id[:3] + '_preprocessed-epo'
    
    stim_epochs.drop(dropped_epochs_list)
    epo_times = np.delete(epo_times,dropped_epochs_list,axis=0)
    delete_multiple_elements(ground_truth,dropped_epochs_list)
    delete_multiple_elements(keys_from_annot,dropped_epochs_list)
    delete_multiple_elements(pain_ratings_lst,dropped_epochs_list)
    

    # Final check
    print('len(dropped_epochs_list):\t', len(dropped_epochs_list))
    print('\nlen(stim_epochs):\t', len(stim_epochs))
    print('\nlen(epo_times):\t\t', len(epo_times))
    print("\nlen(ground_truth):\t",len(ground_truth))
    print("\nlen(pain_ratings_lst):\t",len(pain_ratings_lst))
    
    
    # Complete the saves
    stim_epochs.save(save_path + save_fname + '.fif',
                     verbose=True,
                     overwrite=True)
    
    # save drop log
    print("\nSaving drop_log as mat file...")
    mdic = {"drop_log": dropped_epochs_list}
    scio.savemat(save_path+sub_id[:3]+'_drop_log.mat', mdic)
    
    # save epo_times
    print("\nSaving epoch_times as mat file...")
    mdic = {"epo_times": epo_times}
    scio.savemat(save_path+sub_id[:3]+'_epo_times.mat', mdic)
    
    # save stim labels
    print("\nSaving stim_labels as mat file...")
    mdic = {"stim_labels": ground_truth}
    # mdic = {"stim_labels": keys_from_annot}
    scio.savemat(save_path+sub_id[:3]+'_stim_labels.mat', mdic)
    
    # save pain ratings
    print("\nSaving pain_ratings as mat file...\n")
    mdic = {"pain_ratings": pain_ratings_lst}
    scio.savemat(save_path+sub_id[:3]+'_pain_ratings.mat', mdic)
    
    print("Done.")
    # clear_display()
        
    # verify stim_epochs object looks correct
    print('FINAL EPOCH COUNT:', len(stim_epochs))
    stim_epochs
    
