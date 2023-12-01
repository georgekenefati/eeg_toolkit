from ._utils import * 
from ._preprocess import *
mne.set_log_level('WARNING')
from mne.datasets import fetch_fsaverage
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.beamformer import make_lcmv, apply_lcmv_raw
import scipy.io as scio
get_ipython().run_line_magic('matplotlib', 'inline')

# Source the fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subject = 'fsaverage'
trans='fsaverage'
subjects_dir=os.path.dirname(fs_dir)
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # surface for dSPM
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
model_fname = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem.fif')

# epoch time min and max
def get_time_window():
    bmax=0.
    t_win = float(input("Please enter the peri-stimulus time window."+
    "\nEx: '0 (default)' = [-0.2,0.8], '2' = [-1.0,1.0], etc...\n\n>> "))
    if t_win==0.:
        tmin,tmax = -0.2,0.8
        time_win_path=''
    else:
        tmin,tmax = -t_win/2,t_win/2
    print(f"[{tmin},{bmax},{tmax}]")
    time_win_path=f'{int(t_win)}_sec_time_window/'
    # print(time_win_path)
    return tmin,bmax,tmax,time_win_path

# apply inverse
snr=3.
apply_inverse_raw_kwargs = dict(
    lambda2 = 1. / snr ** 2, # regularizer parameter (λ²)
    verbose=True)

def to_source(raw,data_path,save_path_cont,
              save_path_zepo,
              selected_labels,noise_cov_win,
              include_zepochs=True,
              average_dipoles=True):
    """
    Perform source localization on Raw object
    using fsaverage for certain selected labels.
    raw: Raw object
    selected_labels: list of label names
    noise_cov_win = (rest_min,rest_max). Crop raw during eyes-open resting condition
    include_zepochs = whether to also export z-scored epochs, default True.
    average_dipoles = whether to average source points in each ROI, default True.
    """
    
    ############################## Make STC folders per subject ##############################
    time_win_path_continuous =  os.path.join(save_path_continuous,sub_id)
    if not os.path.exists(os.path.join(save_path_continuous,sub_id)): # continuous
        os.mkdir(time_win_path_continuous)
    time_win_path_zepochs =  os.path.join(save_path_zepochs,sub_id)
    if not os.path.exists(os.path.join(save_path_zepochs,sub_id)): # zepochs
        os.mkdir(time_win_path_zepochs)
        
    ##################### Check whether STC folder already contains expected number of files #########################
    if len(os.listdir(time_win_path_continuous))<len(roi_names) or len(os.listdir(time_win_path_zepochs))<len(roi_names):
        sub_raw_fname = f'{sub_id}_preprocessed-raw.fif'
        sub_epo_fname = f'{sub_id}_preprocessed-epo.fif'
        print(sub_raw_fname)
        print(sub_epo_fname)
    else:
        continue
        
    #################################### Read Raw and Epochs & Set montage ###########################################
    raw_path=os.path.join(data_info_path,sub_raw_fname)
    raw = mne.io.read_raw_fif(raw_path,preload=True)
    raw.set_eeg_reference('average',projection=True)
    
    if "FP1" in raw.ch_names: # wrong 64ch montage, has 4 channels dropped (subjects C24, 055, 056, 047)
        custom_montage = mne.channels.read_custom_montage('../montages/Hydro_Neo_Net_64_xyz_cms_Caps.sfp') 
    else:
        custom_montage = mne.channels.read_custom_montage('./montages/Hydro_Neo_Net_64_xyz_cms.sfp') 
    elif sub_id in subs_32ch:
        custom_montage = mne.channels.read_custom_montage('../montages/Hydro_Neo_Net_32_xyz_cms_No_Fp1.sfp') 

    set_montage(raw,custom_montage)
    # raw.plot_sensors(kind='3d',show_names=True); # optional to plot
    
    epochs_path=os.path.join(epo_time_win_path,sub_epo_fname)
    epochs=mne.read_epochs(epochs_path)
    set_montage(epochs,custom_montage)
    epochs.set_eeg_reference('average',projection=True)

    ##################################### Z-score Epochs then convert to STC ############################################
    data_epo = epochs.get_data()
    data_zepo = np.zeros_like(data_epo)
    base_data = epochs.get_data(tmin=tmin, tmax=bmax)
    
    for i in range(data_epo.shape[0]): # for each epoch
        for j in range(data_epo.shape[1]): # for each channel
            base_mean_tmp = np.mean(base_data[i,j,:])
            base_std_tmp = np.std(base_data[i,j,:])
            data_zepo[i,j,:] = (data_epo[i,j,:] - base_mean_tmp) / base_std_tmp
    
    zepochs = mne.EpochsArray(data_zepo,
                              info=epochs.info, 
                              tmin=tmin, 
                              event_id=epochs.event_id,
                              events=epochs.events,
                              )
    
    ##################################### Compute noise & data covariance ############################################
    raw_crop = raw.copy().crop(tmin=60*rest_min,tmax=60*rest_max) 
    noise_cov = mne.compute_raw_covariance(raw_crop, verbose=True)
    
    ################################### Regularize the covariance matrices ##########################################
    noise_cov = mne.cov.regularize(noise_cov, raw_crop.info,
                                   eeg=0.1, verbose=True)
    
    #################################### Compute the forward solution ###############################################
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                    meg=False, eeg=True, n_jobs=-1,
                                    verbose=True)
    clear_display()   
    
    ###################################### Make the inverse operator ###############################################
    inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, 
                                                 noise_cov, verbose=True)
    
    inverse_operator_zepo = mne.minimum_norm.make_inverse_operator(zepochs.info, fwd, 
                                                 noise_cov, verbose=True)     
    clear_display()    
    ################################# Save source time courses for each label #######################################   
    if len(os.listdir(time_win_path_continuous))<len(roi_names):
        for i in range(int(len(selected_labels)/2)): # half because mne automatically saves both hemispheres
            print(f"************\t{i}\t!! CONT. !! {selected_labels[i].name}\t\t************\n")
            print(sub_raw_fname)
    
            stc_dSPM_tmp = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, label=selected_labels[i], 
                                                              method='dSPM',**apply_inverse_raw_kwargs)
            # Save the continuous STC file
            stc_dSPM_tmp.save(os.path.join(save_path_continuous,sub_id,f"{selected_labels[i].name[:-3]}"),overwrite=True)
            clear_display()

    ################################# Apply inverse to epochs ####################################### 
    if include_zepochs=True:
        if len(os.listdir(time_win_path_zepochs))<len(roi_names):
            for i in range(len(selected_labels)):
                print(f"%%%%%%%%%%%%%%\t{i}\t!! ZEPOCHS !! {selected_labels[i].name}\t\t%%%%%%%%%%%%%%\n")
                print(sub_epo_fname)
                
                zepochs_stc = mne.minimum_norm.apply_inverse_epochs(zepochs, inverse_operator_zepo,label=selected_labels[i], 
                                                                      method='dSPM',**apply_inverse_raw_kwargs)
                # Extract data
                zepochs_stc_data = [el.data for el in zepochs_stc]
                # Turn into 3D array
                zepochs_stc_arr = np.array(zepochs_stc_data)
        
                # Save STC Zepochs per region
                mdict = {"data": zepochs_stc_arr}
                scio.savemat(os.path.join(save_path_zepochs,sub_id,f"{selected_labels[i].name}_stc_zepo.mat"), mdict)
                print(f"Saving file: {selected_labels[i].name[:-3]}_stc_zepo.mat")
                clear_display()
        



