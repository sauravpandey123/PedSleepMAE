import mne
import os
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import timezone
from scipy import interpolate
import sleep_study as ss  # Ensure this is the correct module
from utils.channel_info import * 


def load_study(name, preload=False, exclude=[], verbose='CRITICAL'):
    path = os.path.join(ss.data_dir, 'Sleep_Data', name + '.edf')
    path = os.path.abspath(path)
    raw = mne.io.read_raw_edf(input_fname=path, exclude=exclude, preload=preload,
                                verbose=verbose)

    patient_id, study_id = name.split('_')

    tmp = ss.info.SLEEP_STUDY
    date = tmp[(tmp.STUDY_PAT_ID == int(patient_id))
             & (tmp.SLEEP_STUDY_ID == int(study_id))] \
                     .SLEEP_STUDY_START_DATETIME.values[0] \
                     .split()[0]

    time = str(raw.info['meas_date']).split()[1][:-6]

    new_datetime = parser.parse(date + ' ' + time + ' UTC') \
                         .replace(tzinfo=timezone.utc)

    raw.set_meas_date(new_datetime)
    # raw._raw_extras[0]['meas_date'] = new_datetime

    annotation_path = os.path.join(ss.data_dir, 'Sleep_Data', name + '.tsv')
    df = pd.read_csv(annotation_path, sep='\t')
    annotations = mne.Annotations(df.onset, df.duration, df.description,
                                  orig_time=new_datetime)

    raw.set_annotations(annotations)
    raw.rename_channels({name: name.upper() for name in raw.info['ch_names']})
    return raw




def get_signals_and_stages(name, verbose=False, downsample=True):
    raw = load_study(name, exclude=['Patient Event'])
    freq = int(raw.info['sfreq']) # 256, 400, 512
    channels = raw.ch_names  #These would be the channels within the file, like EEG, C-Pressure, RESP etc and will vary from file to file.
    n_samples = raw.n_times
    if verbose:
        print('sampling rate:', freq, 'Hz')
        print('channel names:', raw.info['ch_names'])
        print( )
        sleep_stage_stats = ss.data.sleep_stage_stats([study])
        print( )
    
    EVENT_DICT = {
        'Sleep stage W' : 0,
        'Sleep stage N1': 1,
        'Sleep stage N2': 2,
        'Sleep stage N3': 3,
        'Sleep stage R' : 4,
        }
    
    events, event_id = mne.events_from_annotations(raw, event_id = EVENT_DICT, verbose=verbose)
    labels = []
    data = []
        
    for event in events:
        label, onset = event[[2, 0]]
        # get 30 seconds of data corresponding to the label
        indices = [onset, onset + ss.info.INTERVAL*freq]
        
        if indices[1] <= n_samples:
            interval_data = raw.get_data(channels, start=indices[0], stop=indices[1]) 
            data.append(interval_data) 
            labels.append(label)
            # sometimes the last interval seems to go over the length of the data and cause problems.
            # it's probably okay to just skip those for now.
    labels = np.array(labels)
    data = np.array(data)
    
    # Downsample to 128Hz
    if downsample:
        if freq % ss.info.REFERENCE_FREQ == 0:
            k = freq//ss.info.REFERENCE_FREQ
            data = data[:,:,::k]

        elif freq != ss.info.REFERENCE_FREQ:
            x = np.linspace(0, ss.info.INTERVAL, num=ss.info.INTERVAL*freq)
            new_x = np.linspace(0, ss.info.INTERVAL, num=ss.info.INTERVAL*ss.info.REFERENCE_FREQ)
            f = interpolate.interp1d(x, data, kind='linear', axis= -1, assume_sorted=True)
            data = f(new_x)   
    
    # data is (num events) by (num channels) by (30s x ss.info.REFERENCE_FREQ) 
    return np.array(data), labels, channels



def filter_and_standardize_channels(signals, channel_names): 
    channel_names = list(channel_names)
    channel_indices = np.array([channel_names.index(ch) for ch in filter_channels])  #filter channels comes from CHANNEL_INFO file. 
    selected_data = signals[:, channel_indices, :]  
    standardized_data = (selected_data - mean_array[:, np.newaxis]) / sd_array[:, np.newaxis]   
    return standardized_data
