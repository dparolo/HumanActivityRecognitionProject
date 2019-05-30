import scipy.io
import numpy as np

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datasets", required = True, help = "dataset file(s) name(s)")
    args = vars(ap.parse_args())
    
    data_names = args["datasets"].split("___")
    data = {}
    for name in data_names:
        d = scipy.io.loadmat(name, squeeze_me=True, struct_as_record=True)
        for key in list(d.keys())[3:]:
            data[name + " " + key] = d[key]
    keys = list(data.keys())
    lengths = np.array([data[key][0].shape[0] for key in keys])
    subjects = [""]*len(keys)
    for i,k in enumerate(keys):
        subjects[i] = k.split(" ")[1].split("_")[1]
    subjects = np.array(subjects)
    unique_subjects = np.unique(subjects)
    total_times = np.empty(unique_subjects.shape, dtype = int)
    for i,subject in enumerate(unique_subjects):
        indices = np.arange(len(subjects))[subjects == subject]
        total_times[i] = np.sum(lengths[indices])
    relative_times = total_times/np.sum(total_times)
    indices = np.argsort(-relative_times)
    sorted_relative_times = relative_times[indices]
    relative_cumsum = np.cumsum(sorted_relative_times)
    rule = relative_cumsum < 0.75
    subjects_train = unique_subjects[indices][rule]
