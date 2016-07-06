from sklearn.metrics import *
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import argparse
import random
try:
    import cPickle as pickle
except:
    import pickle

#basically training and test set partitioning is outside scope of class,
#though there will be an evaluator function
class pyrmClassifier:
    PROB_LIST = namedtuple('PROB_LIST', ['match', 'probability', 'pr'])

    def __init__(self, list_of_categories, 
            directory=os.getcwd(), 
            algorithm='osb unique microgroom',
            word_pattern='[[:graph:]]+'):
        self.categories = list_of_categories #list of categories to classify to
        self.directory = directory #directory to save all files
        self.algorithm = algorithm
        self.reset()
        self._create_crm_files(self.categories, algorithm, word_pattern)

    
    def train(self, category, training_string):
        #write to text file
        file_dir = os.path.join(self.directory, 'train.tmp')
        with open(file_dir, 'w') as f:
            f.write(training_string)
        self._train(file_dir, category)
        os.remove(file_dir)

    #by default will split training files into lines and train each line
    def train_textfile(self, category, train_method='TET', delimiter='\n', *args):
        #check if category exists
        if category not in self.categories:
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
                    best_match, prob_list = self._classify('train.tmp')
                    self._smart_train(best_match, prob_list, category, train_method, 'train.tmp')
                    #delete temporary file
                    subprocess.Popen('rm -rf train.tmp', shell=True)

    def classify(self, string):
        #write to text file
        file_dir = os.path.join(self.directory, 'classify.tmp')
        with open(file_dir, 'w') as f:
            f.write(string)
        bestMatch, probList = self._classify(file_dir)
        self._print_classify(bestMatch, probList)
        subprocess.Popen(['rm', file_dir])
    
    #takes filename as args
    def classify_textfiles(self, *args):
        for textfile in args:
            bestMatch, probList = self._classify(str(textfile))
            self._print_classify(bestMatch, probList)
        #clean_workspace()

    def reset(self, corpus=True, crm=False):
        if (corpus is True):
            subprocess.call('rm -f *.css', shell=True) #remove all corpus type files

        if (crm is True):
            subprocess.call('rm -f *.crm', shell=True)

    def evaluate(self, y_true, y_pred, output=sys.stdout):
        #if --eval flag is used, then classify test set and print statistics
        print 'evaluating algorithm...'

        temp_stats_file = open(os.path.join(self.directory, 'prob_distribution.txt'), 'w')

        print >>output,'------ EVALUATION STATS ------'
        #Compute Accuracy Score
        print >>output, 'Accuracy score (normalized):', accuracy_score(y_true, y_pred)
        print >>output

        #Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.categories)
        print >>output, 'Confusion matrix:\n', cm
        print >>output

        #Classification report
        print >>output, classification_report(y_true, y_pred, target_names=self.categories)
        output.close()

        #show confusion matrix
        plt.figure()
        self._plot_confusion_matrix(cm, self.categories)
        plt.show()
     

    def crm_files_exist(self):
        #check if crm files exist already
        allFilesExist = True
        for category in self.categories:
            complete_directory = os.path.join(self.directory, category + '.css')
            if(os.path.isfile(complete_directory) is False):
                allFilesExist = False
                break
        return allFilesExist

    def _print_classify(self, bestMatch, probList):
        print 'best match: ' + bestMatch[0]
        print 'match, probability, pr:'
        for tup in probList:
            print '\t', str(tup[0]), str(tup[1]), str(tup[2]) #prints probability and pR

    
    def _plot_confusion_matrix(self, cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    #match: (name_of_match, prob_of_match, pr_of_match)
    #probList = [ (name1, probability1, pR1) (name2, probability2, pR2) ...]
    #train_method: (name_of_method, pR_threshold)
    def _smart_train(self, match, probList, truth_match_name, train_method, textfilename,
            prThreshold = 10.0):
        name_of_match = match[0]
        pr = match[2]
        if train_method == 'TOE': #train on error
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                self._train(textfilename, truth_match_name)
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
                    self._train(textfilename, truth_match_name)
        elif train_method == 'SSTTT':
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                self._train(textfilename, truth_match_name)
            elif pr < prThreshold: #if correct, but match PR is less than PR threshold
                self._train(textfilename, truth_match_name)
        elif train_method == 'DSTTT':
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                self._train(textfilename, truth_match_name)
            elif pr < prThreshold: #if correct, but match PR is less than PR threshold
                self._train(textfilename, truth_match_name)

                #if not sure if others were incorrect, then untrain out 
                for tuples in probList:
                    name = tuples[0]
                    if name != truth_match_name:
                        prWrong = abs(tuples[2])
                        if prWrong < prThreshold:
                            self._untrain(textfilename, name)
        elif train_method == 'DSTTTR':
            oldPR = match[2]
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                self._train(textfilename, truth_match_name)
            elif pr < prThreshold: #if correctly matched, but match PR is less than PR threshold
                self._train(textfilename, truth_match_name)

                #reclassify text
                bestMatch, probList = _classify(textfilename)
                newMatchName = bestMatch[0]
                newPR = bestMatch[2]

                #if improvement not good enough, then untrain out of incorrect classes
                if (newPR < prThreshold) or abs(newPR-oldPR) > 3:
                    for tuples in probList:
                        name = tuples[0]
                        if name != truth_match_name:
                            self._untrain(textfilename, name)
        elif train_method == 'TTE':
            if truth_match_name != name_of_match: #if classifier incorrectly predicts 
                self._train(textfilename, truth_match_name)

                #reclassify text
                bestMatch, probList = _classify(textfilename)
                newMatchName = bestMatch[0]
                newPR = bestMatch[2]

                #keep training until pr threshold improves to satisfactory level
                loopCount = 0
                while (newPR < prThreshold and loopCount is not 5):
                    self._train(textfilename, truth_match_name)

                    #reclassify text
                    bestMatch, probList = _classify(textfilename)
                    newMatchName = bestMatch[0]
                    newPR = bestMatch[2]
        elif train_method == 'TUNE':
            raise Exception('TUNE not implemented yet')
        else: #by default train everything (TET)
            self._train(textfilename, truth_match_name)



    def _train(self, trainingTxtFile, corpus_name):
        if corpus_name not in self.categories:
            raise ValueError('Category doesn\'t exist!')
        subprocess.call('crm ' + os.path.join(self.directory, 'learn.crm') +
                ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True)
    def _untrain(self, trainingTxtFile, corpus_name):
        subprocess.call('crm ' + os.path.join(self.directory, 'unlearn.crm') +
                ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True)

    # classifies textfile and returns best match and probabilities
    # bestMatch = tuple(bestMatch bestProb, bestPR)
    # probList = [ (twitterHandle1, probability1, pR1) (twitterHandle2, probability2, pR2) ...]
    def _classify(self, textFileName):
        output =  subprocess.check_output('crm ' + os.path.join(self.directory, 'classify.crm') + ' < ' + textFileName, shell=True) #string output from crm114
        outList = output.split()
        bestMatch = ( str(outList[0]), float(outList[1]), float(outList[2]) ) #(match, prob, pR)
        outList = outList[3:]

        probList = []
        it = iter(outList)
        for x in it:
            x.rstrip(':')
            p = (x, float(next(it)), float(next(it)))
            probList.append(p)
            #probList.append((x, float(next(it)), float(next(it)) ))
        
        #probList: (match, probability, pR)

        return (bestMatch, tuple(probList)) 

    def _create_crm_files(self, file_names, classification_type, word_pat):
        #classification_type = classification_type.rstrip('>').strip('<')
        CLASSIFY_EXT = '.css'    #create files if they don't exist
        UNLEARN_CMD = "{ learn <%s refute> (:*:_arg2:) }"
        LEARN_CMD = "{ learn <%s> (:*:_arg2:) }"
        CLASSIFY_CMD = "{ isolate (:stats:);" \
                " classify <%s> ( %s ) (:stats:) /%s/;" \
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
        learnCRM = open(os.path.join(self.directory,'learn.crm'), 'w')
        learnCRM.write(LEARN_CMD % classification_type)
        learnCRM.close()
        #print 'created learn.crm'

        #create unlearn.crm
        unlearnCRM = open(os.path.join(self.directory, 'unlearn.crm'), 'w')
        unlearnCRM.write(UNLEARN_CMD % classification_type)
        unlearnCRM.close()
        #print 'created unlearn.crm'

        #create classify.crm
        classifyCRM = open('classify.crm', 'w')
        name_list = [name + CLASSIFY_EXT for name in file_names]
        match_list = [MATCH_VAR % (name, name, name, name, name) for name in name_list] #create list of MATCH_VARs based on screen name
        output_list = ['%s: :*:%s_prob: :*:%s_pr:' % (name.split('.')[0], name, name) for name in name_list] #create list for output

        classifyCRM.write(CLASSIFY_CMD % (classification_type,
                                          ' '.join(name_list),
                                          word_pat,
                                          ' '.join(match_list),
                                          ' '.join(output_list)
                                         ))
        classifyCRM.close()
        #print 'created classify.crm'

        CRM_BINARY = 'crm'
        CLASSIFICATION_EXT = '.css'

        #create corpus files
        for n in file_names:
            pipe = os.popen(('crm '+ os.path.join(self.directory,'learn.crm') + ' ' +  str(n + CLASSIFICATION_EXT)), 'w')
            pipe.close()
        #print 'created corpus files'

# def main(argv):
#     #Create top level parser
#     parser = argparse.ArgumentParser(
#             description='Machine learning classifier that learns people\'s speech patterns via their tweets.',
#             prog='pyrm114.py')
#     subparsers = parser.add_subparsers(help='Use one of the following three commands:\n' \
#                                             '\ttrain --help\n' \
#                                             '\tclassify --help\n' \
#                                             '\treset --help\n')
# 
#     #create parser for 'train' command
#     parser_train = subparsers.add_parser('train', help='given twitter handles, train the classifier')
#     parser_train.add_argument('--trainstring', '-ts', nargs=2, type=str, 
#             help = 'train a string')
#     parser_train.add_argument('screen_names', nargs='+',
#             help = 'twitter handles for those whose tweets you want to use to train the classifier')
#     parser_train.add_argument('--trainpartition', '-tp', nargs='?', default='.8', type=float,
#             help = 'portion of tweets allocated for training. rest is for testing')
#     parser_train.add_argument('--algorithm', '-a',  nargs='?', type=str,  default='osb unique microgroom',
#             help = 'type of algorithm for crm114. e.g. \'%(default)s\'')
#     parser_train.add_argument('--directory', '-d', nargs='?', type=str, default = os.path.dirname(os.path.abspath(__file__)),
#             help = 'directory that all program files should go into')
#     parser_train.add_argument('--eval', nargs='?', type=argparse.FileType('w'), const = sys.stdout,
#             help = 'evaluate effectiveness of algorithm by separating tweets into training/test sets and printing model evaluation statistics')
#     parser_train.add_argument('--trainmethod', '-tm', nargs='*',  
#             help = 'change method of training (TOE, SSTTT, DSTTT, DSTTTR, TTE, TUNE). only works with the --eval flag')
#     parser_train.set_defaults(func=train_command)
#     #realpolitalk specific
#     parser_train.add_argument('--offline', action='store_true', help = 'use offline saved tweets')
# 
#     #create parser for 'reset' command
#     parser_reset = subparsers.add_parser('reset',
#          help = 'commands to delete saved files (corpus, tweets, crm).')
#     parser_reset.add_argument('--corpus', action='store_true', help = 'deletes all trained corpuses')
#     parser_reset.add_argument('--crm', action='store_true', help = 'remove .crm files')
#     parser_reset.add_argument('--all', action='store_true', help = 'deletes corpuses, tweets, and crm files')
#     #realpolitalk specific
#     parser_reset.add_argument('--tweets', action='store_true', help = 'deletes all saved offline tweets')
#     parser_reset.set_defaults(func = reset_command)
# 
#     #-classify - UNDER CONSTRUCTION
#     parser_classify = subparsers.add_parser('classify', help='classify textfile(s) based on trained corpuses')
#     parser_classify.add_argument('textfiles', nargs='+', type=str, help='textfiles for classifying. E.g. test.txt')
#     parser_classify.set_defaults(func = classify_command)
# 
#     #parse the args and call whichever function was selected (func=...)
#     args = parser.parse_args()
#     args.func(args)


#if __name__ == '__main__':
#    main(sys.argv[1:])


