from tensorboard.backend.event_processing import event_accumulator

from heapq import nsmallest

from os import listdir, makedirs
from os.path import isfile, join, exists, isdir

from shutil import copy2
import re

weights_dir = 'weights/'

def main(model_name, save_freq, lastmdls):
    global lowestval_weight_dir, scl
    
    tensordir = f'logs/{model_name}/'
    lowestval_weight_dir = join(tensordir, "lowestvals")
    
    try: latest_file = next((f for f in listdir(tensordir) if f.endswith('.0')), None)
    except FileNotFoundError: print("Couldn't find your model!"); return
    
    try: tfile = join(tensordir, latest_file)
    except TypeError: print("Couldn't find a valid tensorboard file!"); return
    
    ea = event_accumulator.EventAccumulator(tfile,
        size_guidance={ # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    })

    ea.Reload()
    ea.Tags()

    scl = ea.Scalars('loss/g/total')

    listwstep = {
        float(val.value): (val.step // save_freq) * save_freq
        for val in scl
        if (val.step // save_freq) * save_freq in [val.step for val in scl]
    }

    lowest_vals = sorted(listwstep.keys())[:lastmdls]

    listwstep = {
        value: step
        for value, step in listwstep.items()
        if value in lowest_vals
    }
    
    sorted_dict = dict(sorted(listwstep.items(), key=None, reverse=False))
    
    return sorted_dict
    

def selectweights(model_name, file_dict, weights_dir, lowestval_weight_dir):
    try: makedirs(lowestval_weight_dir)
    except FileExistsError: pass
    logdir = []
    file_dict = eval(file_dict)

    for key, value in file_dict.items():
        pattern = fr"_s{value}\.pth$"
        matching_weights = [f for f in listdir(weights_dir) if re.search(pattern, f)]
        for weight in matching_weights:
            source_path = join(weights_dir, weight)
            destination_path = join(lowestval_weight_dir, weight)
            
            copy2(source_path, destination_path)
            logdir.append(f"File = {weight} Value: {key}, Step: {value}\n")
            print(f"File = {weight} Value: {key}, Step: {value}")
    
    result = ''.join(logdir)
    
    return result

if __name__ == "__main__":
    model = str(input("Enter the name of the model: "))
    sav_freq = int(input("Enter save frequency of the model: "))
    ds = main(model, sav_freq)
    
    if ds: selectweights(model, ds, weights_dir, lowestval_weight_dir)
    