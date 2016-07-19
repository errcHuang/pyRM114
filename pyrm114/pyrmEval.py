import random
import sys
#slices a list into n nearly-equal-length partitions
#returns list of lists
def random_partition(lst, n, shuffle=True):
    if shuffle is True:
        random.shuffle(lst) #optional
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i+1)))] for i in xrange(n) ]

#Randomly resamples labeled datasets into comprehensive training set and test set
#reshuffles data and returns training/test sets as list of dictionaries
#the arguments should be lists that represent classes

#output: tuple
def make_training_and_test_set(trainProportion, shuffle=True, **kwargs):
    if trainProportion > 1.0 or trainProportion < 0.0:
        raise ValueError('trainPartition can only be between 0 and 1.0')
    training_data = []
    test_data = []
    for key, class_list in kwargs.iteritems():
        if shuffle is True:
            random.shuffle(class_list) #optional
        trainIndex = trainProportion * len(class_list) #calculates index for end of training set
        trainIndex = int(round(trainIndex))
        training_data.append( {key : class_list[:trainIndex]} ) #partitions training set from start to random index
        test_data.append( {key : class_list[trainIndex:]} ) #partition test set from random index to end
    return (training_data, test_data)

#perform n-fold cross validation
#returns list of lists of dictionaries
def create_cross_validate_set(n, shuffle=True, **kwargs):
    dataset = []
    for key, value_list in kwargs.items():
        for values in value_list:
            dataset.append({ key : values })
    #dataset, filler = make_training_and_test_set(1.0, True, **kwargs)
    subpartitions = random_partition(dataset, n)
    return subpartitions
