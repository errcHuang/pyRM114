
#slices a list into n nearly-equal-length partitions
#returns list of lists
def random_partition(lst, n):
    random.shuffle(lst) #optional
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i+1)))] for i in xrange(n) ]

#Randomly resamples labeled datasets into comprehensive training set and test set
#reshuffles data and returns training/test sets at
#the list 'dataset' should have lists that represent classes
def make_training_and_test_set(trainProportion, dataset):
    training_data = []
    test_data = []
    for data in dataset:
        random.shuffle(data) #optional
        trainIndex = trainProportion * len(data) #calculates index for end of training set
        trainIndex = int(round(trainIndex))
        training_data.append(data[:trainIndex]) #partitions training set from start to random index
        test_data.append(data[trainIndex:]) #partition test set from random index to end
    return (training_data, test_data)


