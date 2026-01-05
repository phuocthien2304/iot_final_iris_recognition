import os
import random
from collections import defaultdict

def parse_casia_interval_filename(filename):
    """

    :param filename: Filename of the Casia-Iris-Interval image
    :return:
    identifier -> unique ID of the subject
    side -> L or R for left or right eye - unique per person
    index -> index of the image for this class - there are multiple pictures per eye
    """
    filename = filename.replace(".png", "").replace(".jpg", "")
    index = int(filename[-2:])
    side = filename[-3:-2]
    identifier = int(filename[-6:-3])
    return identifier, side, index

def parse_casia_thousand_filename(filename):
    """

    :param filename: Filename of the Casia-Iris-Interval image
    :return:
    identifier -> unique ID of the subject
    side -> L or R for left or right eye - unique per person
    index -> index of the image for this class - there are multiple pictures per eye
    """
    filename = filename.replace(".jpg", "").replace(".png", "")
    index = int(filename[-2:])
    side = filename[-3:-2]
    identifier = int(filename[-6:-3])
    return identifier, side, index



def get_files_walk(dir):
    """
    Generate all filenames in a given dir
    :param dir:
    :return:
    """
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            yield os.path.join(dirpath, file)


def casia_train_val_test_split(dir, parse_func=parse_casia_interval_filename, from_=0, to=1500, random_seed = 42):
    identities = defaultdict(list)
    train = defaultdict(list)
    validation = defaultdict(list)
    test = defaultdict(list)

    for file in get_files_walk(dir):
        if "_mask" in file: continue
        try:
            identifier, side, index = parse_func(file)
            identities[(identifier, side)].append(file)
        except (ValueError, IndexError):
            continue

    # Filter out identities with only a single picture of iris -> can't do recognition with only a single image per class??
    # Filter by ID value (not by index)
    identities_filtered = { key : sorted(identities[key], key=lambda x: x[0]) for key in identities.keys() if from_ <= key[0] < to and len(identities[key]) >= 2 }
    # Split identities
    for key in identities_filtered.keys():
        test[key].append(identities_filtered[key][1]) # add the second picture
        validation[key].append(identities_filtered[key][4])
        validation[key].append(identities_filtered[key][6])
        train[key] += [identities_filtered[key][0],
                       identities_filtered[key][2],
                       identities_filtered[key][3],
                       identities_filtered[key][5]] + identities_filtered[key][7:]

    return train, validation, test

def casia_enrollment_test_split(dir, parse_func=parse_casia_interval_filename, from_=0, to=1500):
    """
    Split for open-set recognition: enrollment (7 files) and test (3 files)
    enrollment: indices 0,2,3,5,7,8,9
    test: indices 1,4,6
    """
    identities = defaultdict(list)
    enrollment = defaultdict(list)
    test = defaultdict(list)

    for file in get_files_walk(dir):
        if "_mask" in file: continue
        try:
            identifier, side, index = parse_func(file)
            identities[(identifier, side)].append(file)
        except (ValueError, IndexError):
            continue

    # Filter by ID value and require at least 2 files
    identities_filtered = { key : sorted(identities[key], key=lambda x: x[0]) for key in identities.keys() if from_ <= key[0] < to and len(identities[key]) >= 2 }
    
    # Split identities for open-set
    for key in identities_filtered.keys():
        files = identities_filtered[key]
        # test: indices 1,4,6
        if len(files) > 1:
            test[key].append(files[1])
        if len(files) > 4:
            test[key].append(files[4])
        if len(files) > 6:
            test[key].append(files[6])
        
        # enrollment: indices 0,2,3,5,7,8,9+
        if len(files) > 0:
            enrollment[key].append(files[0])
        if len(files) > 2:
            enrollment[key].append(files[2])
        if len(files) > 3:
            enrollment[key].append(files[3])
        if len(files) > 5:
            enrollment[key].append(files[5])
        if len(files) > 7:
            enrollment[key] += files[7:]

    return enrollment, test

def load_train_test(dir, filename_parse_func=parse_casia_thousand_filename):
    train_files = [os.path.join(dir, "train", file) for file in os.listdir(os.path.join(dir, "train")) if "_mask" not in file]
    test_files = [os.path.join(dir, "test", file) for file in os.listdir(os.path.join(dir, "test")) if "_mask" not in file]

    train_dict = defaultdict(list)
    for file in train_files:
        identifier, side, _ = filename_parse_func(file)
        train_dict[(identifier, side)].append(file)

    test_dict = defaultdict(list)
    for file in test_files:
        identifier, side, _ = filename_parse_func(file)
        test_dict[(identifier, side)].append(file)

    return train_dict, test_dict
