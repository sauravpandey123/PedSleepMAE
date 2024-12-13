from math import ceil
import os
import pandas as pd

def findLabel(description):
    EVENT_DICT = {
        'oxygen desaturation':1,
        'eeg arousal':1,
        'central apnea':1,
        'obstructive apnea':2,
        'mixed apnea':3,
        'obstructive hypopnea':1,
        'hypopnea':2
        }
    if description in EVENT_DICT:
        return EVENT_DICT[description]    

def getCorrespondingLabel(tsvfile, term):      
    
    directory = "/nas/longleaf/home/srpandey/Desktop/leelab/projects/NCH_Sleep_Data/Sleep_Data/"
    edf_files = [f.replace(".edf","") for f in os.listdir(directory) if f.endswith('.edf')]
    tsv_file = directory + tsvfile + ".tsv"
    df = pd.read_csv(tsv_file,sep='\t')
    labelsList = []

    correspondingInterval = [] # the thirty second intervals corresponding to the label
    indices = []
    sleep_stages = ['Sleep stage W','Sleep stage N1','Sleep stage N2','Sleep stage N3','Sleep stage R']
    
    if term == "apnea": searchFor = ["central apnea", "mixed apnea", "obstructive apnea"]
    if term == "desat": searchFor = ["oxygen desaturation"]
    if term == "eeg": searchFor = ["eeg arousal"]
    if term == "hypop": searchFor = ["obstructive hypopnea", "hypopnea"]
    
    for i in range(len(df)):
        description = df.at[i, 'description'].strip()
        if description in sleep_stages:
            labelsList.append(i)
 
            
    acceptable_digits = [findLabel(item) for item in searchFor]
    for tracker in range(len(labelsList)):
        if (labelsList[tracker] in acceptable_digits): #skip those that have a 1 since you have already scanned them
            continue
        current_tracker = tracker 
        next_tracker = current_tracker + 1 
        current_number = labelsList[current_tracker]  #current corresponding index
        if (next_tracker == len(labelsList)):  #handle the last element
            next_number = current_number+1
        else:
            next_number = labelsList[next_tracker] #next corresponding index
        for i in range(current_number, next_number,1):  #for the first time, it would be between 12 and 17  (current, next + 1)
            description = df.at[i, 'description'].strip()  #check if this is oxygen desat
            if description.lower() in searchFor:  #we found oxygen deset
                searchLabel = findLabel(description.lower())
                labelsList[current_tracker] = searchLabel
                duration = df.at[i, 'duration']
                event_onset = df.at[i, 'onset']
                start_sleep_stage = df.at[current_number, 'onset']
                available = ceil(start_sleep_stage + 30 - event_onset) #some issues with 29.999995
                if (available < duration):
                    # print (duration, start_sleep_stage, available)
                    remaining = duration - available
                    slots = ceil(remaining/30)
                    for j in range(1, slots + 1):
                        if (current_tracker + j < len(labelsList)):
                            labelsList[current_tracker + j] = searchLabel
                        else:
                            print ("Overflow detected at the end in file:", tsv_file)
    for i in range(len(labelsList)):
        if (labelsList[i] not in acceptable_digits):
            labelsList[i] = 0

    return (labelsList)
