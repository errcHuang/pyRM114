import random
#slices a list into n nearly-equal-length partitions
#returns list of lists
def random_partition(lst, n):
    random.shuffle(lst) #optional
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i+1)))] for i in xrange(n) ]

#Randomly resamples labeled datasets into comprehensive training set and test set
#reshuffles data and returns training/test sets as list of dictionaries
#the arguments should be lists that represent classes
def make_training_and_test_set(trainProportion, shuffle=True, **kwargs):
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


