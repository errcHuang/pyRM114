from tweepy import OAuthHandler
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import argparse
import tweepy
import random
import re
try:
    import cPickle as pickle
except:
    import pickle

__directory__ = '' #placeholder for directory
def main(argv):
    #Create top level parser
    parser = argparse.ArgumentParser(
            description='Machine learning classifier that learns people\'s speech patterns via their tweets.',
            prog='realpolitalk.py')
    subparsers = parser.add_subparsers(help='Use one of the following three commands:\n' \
                                            '\ttrain --help\n' \
                                            '\tclassify --help\n' \
                                            '\treset --help\n')

    #create parser for 'train' command
    parser_train = subparsers.add_parser('train', help='given twitter handles, train the classifier')
    parser_train.add_argument('screen_names', nargs='+',
            help = 'twitter handles for those whose tweets you want to use to train the classifier')
    parser_train.add_argument('--trainpartition', '-tp', nargs='?', default='.8', type=float,
            help = 'portion of tweets allocated for training. rest is for testing')
    parser_train.add_argument('--algorithm', '-a',  nargs='?', type=str,  default='osb unique microgroom',
            help = 'type of algorithm for crm114. e.g. \'%(default)s\'')
    parser_train.add_argument('--directory', '-d', nargs='?', type=str, default = os.path.dirname(os.path.abspath(__file__)),
            help = 'directory that all program files should go into')
    parser_train.add_argument('--eval', nargs='?', type=argparse.FileType('w'), const = sys.stdout,
            help = 'evaluate effectiveness of algorithm by separating tweets into training/test sets and printing model evaluation statistics')
    parser_train.add_argument('--trainmethod', '-tm', nargs='*',  
            help = 'change method of training (TOE, SSTTT, DSTTT, DSTTTR, TTE, TUNE). only works with the --eval flag')
    parser_train.set_defaults(func=train_command)
    #realpolitalk specific
    parser_train.add_argument('--offline', action='store_true', help = 'use offline saved tweets')

    #create parser for 'reset' command
    parser_reset = subparsers.add_parser('reset',
         help = 'commands to delete saved files (corpus, tweets, crm).')
    parser_reset.add_argument('--corpus', action='store_true', help = 'deletes all trained corpuses')
    parser_reset.add_argument('--crm', action='store_true', help = 'remove .crm files')
    parser_reset.add_argument('--all', action='store_true', help = 'deletes corpuses, tweets, and crm files')
    #realpolitalk specific
    parser_reset.add_argument('--tweets', action='store_true', help = 'deletes all saved offline tweets')
    parser_reset.set_defaults(func = reset_command)

    #-classify - UNDER CONSTRUCTION
    parser_classify = subparsers.add_parser('classify', help='classify textfile(s) based on trained corpuses')
    parser_classify.add_argument('textfiles', nargs='+', type=str, help='textfiles for classifying. E.g. test.txt')
    parser_classify.set_defaults(func = classify_command)

    #parse the args and call whichever function was selected (func=...)
    args = parser.parse_args()
    args.func(args)

#basically training and test set partitioning is outside scope of class,
#though there will be an evaluator function
class pyRM114:
    PROB_LIST = namedtuple('PROB_LIST', ['match', 'probability', 'pr'])

    def __init__(self, *args, directory='.', algorithm='osb unique microgroom'):
        self.categories = args #list of categories to classify to
        self.directory = directory #directory to save all files
        self.algorithm = algorithm
        _create_crm_files(categories, algorithm)

        #its user responsibility to separate training files into small and separate textfiles
    
    def train(self, category, training_string):
        #write to text file
        file_dir = os.path.join(self.directory, 'train.tmp')
        with open(file_dir, 'w') as f:
            f.write(training_string)
        _train(file_dir, category)    
        subprocess.Popen(['rm', '-rf',file_dir])

    #by default will split training files into lines and train each line
    def train_textfile(self, category, *args, train_method='TET', delimiter='\n'):
        #check if category exists
        if category is not in self.categories:
            raise IndexError('Category does not exist!')
            return

        #global vars
        file_names = args #screen names for training

        #train classifier
        print 'training...'
        for fName in file_names:
            with open(fName, 'r') as f:
                entire_file = f.read()
                file_sections = entire_file.split(delimiter)

                for section in file_sections:
                    #write section to temporary train file
                    with open('train.tmp', 'w') as t:
                        t.write(section)
                    #attempt to classify and then train based on results
                    best_match, prob_list = _classify('train.tmp')
                    _smart_train(best_match, prob_list, category, train_method, 'train.tmp')
                    #delete temporary file
                    subprocess.Popen('rm -rf train.tmp', shell=True)

    def classify(self, string):
        #write to text file
        file_dir = os.path.join(self.directory, 'classify.tmp')
        with open(file_dir, 'w') as f:
            f.write(string)
        bestMatch, probList = _classify(file_dir)
            print 'best match: ' + bestMatch
            print 'probabilities:'
            for tup in probList:
                print '\t' + str(tup[0]) + ': ' + str(tup[1])
    
    #takes filename as args
    def classify_textfiles(self, *args):
        for textfile in args:
            bestMatch, probList = _classify(str(textfile))
            print 'best match: ' + bestMatch
            print 'probabilities:'
            for tup in probList:
                print '\t' + str(tup[0]) + ': ' + str(tup[1])
            print
        #clean_workspace()

    def reset(corpus=True, crm=True):
        if (corpus is True):
            subprocess.call('rm -f *.css', shell=True) #remove all corpus type files

        if (crm is True):
            subprocess.call('rm -f *.crm', shell=True)

    def evaluate(self):
        #if --eval flag is used, then classify test set and print statistics
        if (args.eval is not None):
            print 'evaluating algorithm...'


            temp_stats_file = open(os.path.join(self.directory, 'prob_distribution.txt'), 'w')

            y_true = []
            y_pred = []
            print >>temp_stats_file,'------ BEST MATCH AND PROBABILITIES ------'


                for t in tweets:
                    trueAuthor = t.author.screen_name
                    tweetFileName = write_tweets_to_file([t], self.directory, trueAuthor + '.txt')
                    bestMatch, probList = _classify(tweetFileName)

                    #Write best match and probabilities either to std.out or to file
                    print >>temp_stats_file, 'best match: ' + bestMatch[0]
                    print >>temp_stats_file, 'probabilities:'
                    for tup in probList:
                        print >>temp_stats_file, '\t' + str(tup[0]) + ': ' + str(tup[1])
                    print >>temp_stats_file, ''

                    y_true.append(trueAuthor)
                    y_pred.append(bestMatch[0])
            temp_stats_file.close()

            print >>args.eval,'------ EVALUATION STATS ------'
            #Compute Accuracy Score
            print >>args.eval, 'Accuracy score (normalized):', accuracy_score(y_true, y_pred)
            print >>args.eval

            #Confusion Matrix
            cm = confusion_matrix(y_true, y_pred, labels=screen_names)
            print >>args.eval, 'Confusion matrix:\n', cm
            print >>args.eval

            #Classification report
            print >>args.eval, classification_report(y_true, y_pred, target_names=screen_names)
            args.eval.close()

            #show confusion matrix
            plt.figure()
            plot_confusion_matrix(cm, screen_names)
            plt.show()

        clean_workspace(screen_names)
    
    #match: (name_of_match, prob_of_match, pr_of_match)
    #probList = [ (name1, probability1, pR1) (name2, probability2, pR2) ...]
    #train_method: (name_of_method, pR_threshold)
    def _smart_train(match, probList, truth_match_name, train_method, textfilename,
            prThreshold = 10.0):
        name_of_match = match[0]
        pr = match[2]
        if train_method == 'TOE': #train on error
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                _train(textfilename, truth_match_name)
            else:
                #make sure that all the probabilities are different
                #else it's basically the same as random guessing
                previousProb = probList[0][1]
                isEqual = True
                for probTuple in probList:
                    if probTuple[2] != previousProb:
                        isEqual = False
                        break
                    previousProb = probTuple[2]
                
                if isEqual is True:
                    _train(textfilename, truth_match_name)
        elif train_method == 'SSTTT':
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                _train(textfilename, truth_match_name)
            elif pr < prThreshold: #if correct, but match PR is less than PR threshold
                _train(textfilename, truth_match_name)
        elif train_method == 'DSTTT':
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                _train(textfilename, truth_match_name)
            elif pr < prThreshold: #if correct, but match PR is less than PR threshold
                _train(textfilename, truth_match_name)

                #if not sure if others were incorrect, then untrain out 
                for tuples in probList:
                    name = tuples[0]
                    if name != truth_match_name:
                        prWrong = abs(tuples[2])
                        if prWrong < prThreshold:
                            _untrain(textfilename, name)
        elif train_method == 'DSTTTR':
            oldPR = match[2]
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                _train(textfilename, truth_match_name)
            elif pr < prThreshold: #if correctly matched, but match PR is less than PR threshold
                _train(textfilename, truth_match_name)

                #reclassify text
                bestMatch, probList = _classify(textfilename)
                newMatchName = bestMatch[0]
                newPR = bestMatch[2]

                #if improvement not good enough, then untrain out of incorrect classes
                if (newPR < prThreshold) or abs(newPR-oldPR) > 3:
                    for tuples in probList:
                        name = tuples[0]
                        if name != truth_match_name:
                            _untrain(textfilename, name)
        elif train_method == 'TTE':
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                _train(textfilename, truth_match_name)

                #reclassify text
                bestMatch, probList = _classify(textfilename)
                newMatchName = bestMatch[0]
                newPR = bestMatch[2]

                #keep training until pr threshold improves to satisfactory level
                loopCount = 0
                while (newPR < prThreshold and loopCount is not 5):
                    _train(textfilename, truth_match_name)

                    #reclassify text
                    bestMatch, probList = _classify(textfilename)
                    newMatchName = bestMatch[0]
                    newPR = bestMatch[2]
        elif train_method == 'TUNE':
            raise Exception('TUNE not implemented yet')
        else: #by default train everything (TET)
            _train(textfilename, truth_match_name)



    def _train(trainingTxtFile, corpus_name):
        subprocess.call('crm ' + os.path.join(__directory__, 'learn.crm') +
                ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True)
    def _untrain(trainingTxtFile, corpus_name):
        subprocess.call('crm ' + os.path.join(__directory__, 'unlearn.crm') +
                ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True)

    # classifies textfile and returns best match and probabilities
    # bestMatch = tuple(bestMatch bestProb, bestPR)
    # probList = [ (twitterHandle1, probability1, pR1) (twitterHandle2, probability2, pR2) ...]
    def _classify(textFileName):
        output =  subprocess.check_output('crm ' + os.path.join(__directory__, 'classify.crm') + ' < ' + textFileName, shell=True) #string output from crm114
        outList = output.split()
        bestMatch = ( str(outList[0]), float(outList[1]), float(outList[2]) ) #(match, prob, pR)
        outList = outList[3:]

        probList = []
        it = iter(outList)
        for x in it:
            x.rstrip(':')
            p = PROB_LIST(x, float(next(it)), float(next(it)))
            probList.append(p)
            #probList.append((x, float(next(it)), float(next(it)) ))
        
        #probList: (match, probability, pR)

        return (bestMatch, tuple(probList)) 

    def _create_crm_files(file_names, classification_type):
        #classification_type = classification_type.rstrip('>').strip('<')
        CLASSIFY_EXT = '.css'    #create files if they don't exist
        UNLEARN_CMD = "{ learn <%s refute> (:*:_arg2:) }"
        LEARN_CMD = "{ learn <%s> (:*:_arg2:) }"
        CLASSIFY_CMD = "{ isolate (:stats:);" \
                " classify <%s> ( %s ) (:stats:);" \
                " match [:stats:] (:: :best: :prob: :pr:)" \
                " /Best match to file #. \\(([[:graph:]]+)\\) [[:graph:]]+: ([0-9\\.]+)[[:space:]]+pR:[[:space:]]+([[:graph:]]+)/;" \
                " %s " \
                " match [:best:] (:: :best_match:) /([[:graph:]]+).css/;" \
                " output /:*:best_match: :*:prob: :*:pr: \\n%s\\n/ }" # %output_list
        MATCH_VAR = 'match [:stats:] (:: :%s_temp:)' \
                        ' /\\(%s\\): (.*?)\\\\n/;' \
                    ' match [:%s_temp:] (:: :%s_prob: :%s_pr:)' \
                    ' /prob: ([[:graph:]]+), pR:[[:space:]]+([[:graph:]]+)/;'
        #create learn.crm
        learnCRM = open(os.path.join(__directory__,'learn.crm'), 'w')
        learnCRM.write(LEARN_CMD % classification_type)
        learnCRM.close()

        #create unlearn.crm
        unlearnCRM = open(os.path.join(__directory__, 'unlearn.crm'), 'w')
        unlearnCRM.write(UNLEARN_CMD % classification_type)
        unlearnCRM.close()

        #create classify.crm
        classifyCRM = open('classify.crm', 'w')
        name_list = [name + CLASSIFY_EXT for name in file_names]
        match_list = [MATCH_VAR % (name, name, name, name, name) for name in name_list] #create list of MATCH_VARs based on screen name
        output_list = ['%s: :*:%s_prob: :*:%s_pr:' % (name.split('.')[0], name, name) for name in name_list] #create list for output

        classifyCRM.write(CLASSIFY_CMD % (classification_type,
                                          ' '.join(name_list),
                                          ' '.join(match_list),
                                          ' '.join(output_list)
                                         ))
        classifyCRM.close()

        CRM_BINARY = 'crm'
        CLASSIFICATION_EXT = '.css'

        #create corpus files
        for n in file_names:
            pipe = os.popen(('crm '+ os.path.join(__directory__) + 'learn.crm ' +  str(n + CLASSIFICATION_EXT)), 'w')
            pipe.close()







def crm_files_exist(screen_names):
    #check if files exit already
    allFilesExist = True
    for name in screen_names:
        complete_directory = os.path.join(__directory__, name + '.css')
        if(os.path.isfile(complete_directory) is False):
            allFilesExist = False
            break
    return allFilesExist

#slices a list into n nearly-equal-length partitions
#returns list of lists
def random_partition(lst, n):
    random.shuffle(lst)
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i+1)))] for i in xrange(n) ]

#Randomly resamples labeled datasets into comprehensive training set and test set
#reshuffles data and returns training/test sets at
#the list 'dataset' should have lists that represent classes
def get_training_and_test_set(trainProportion, dataset):
    training_data = []
    test_data = []
    for data in dataset:
        random.shuffle(data)
        trainIndex = trainProportion * len(data) #calculates index for end of training set
        trainIndex = int(round(trainIndex))
        training_data.append(data[:trainIndex]) #partitions training set from start to random index
        test_data.append(data[trainIndex:]) #partition test set from random index to end
    return (training_data, test_data)




def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def clean_workspace(screen_names):
    for name in screen_names:
        subprocess.call('rm -f ' + os.path.join(__directory__, str(name)+ '.txt'), shell=True)

if __name__ == '__main__':
    main(sys.argv[1:])


