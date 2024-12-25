import sys

def get_class_weights(labels):
    labels = labels.lower()
    if (labels == "a"):
        apnea_labels_combined_weights = {0:1, 1:121}
        return apnea_labels_combined_weights
    elif (labels == "h"):
        hypop_labels_combined_weights = {0:1, 1:50}
        return hypop_labels_combined_weights
    elif (labels == "ah"): 
        apnea_hypopnea_combined_weights = {0:1, 1:37} 
        return apnea_hypopnea_combined_weights
    elif (labels == "e"):
        eeg_labels_weights = {0: 0.5245960633523447, 1: 10.664228170121712}
        return eeg_labels_weights
    elif (labels == "d"):
        desat_labels_weights =  {0: 0.5483390312492801, 1: 5.671804099895436}
        return desat_labels_weights
    elif (labels == "s"):
        sleep_labels_weights = {0: 0.9, 1: 5, 2: 0.9, 3: 0.9, 4: 0.9}
        return sleep_labels_weights
 