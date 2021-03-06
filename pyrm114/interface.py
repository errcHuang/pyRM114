from sklearn.metrics import *
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import argparse
import random
import ntpath

#TO-DO: make training echo like classify is
#basically training and test set partitioning is outside scope of class,
#though there will be an evaluator function
class pyrmClassifier:
    PROB_LIST = namedtuple('PROB_LIST', ['match', 'probability', 'pr'])

    def __init__(self, list_of_categories, 
            directory=os.getcwd(), 
            algorithm='osb unique microgroom',
            word_pattern='[[:graph:]]+',
            reset=True):
        self.categories = list_of_categories #list of categories to classify to
        self.directory = self._create_directory(directory) #directory to save all files
        self.algorithm = algorithm
        if reset is True:
            self.reset()
        self._create_crm_files(self.categories, algorithm, word_pattern)

    def train(self, category, training_string, train_method='TET', pr=10.0):
        #write training string to textfile so can be processed by CRM114
        #r = random.randint(0, len(training_string))
        random_file_name = 'train.tmp'
        file_dir = os.path.join(self.directory, random_file_name)
        with open(file_dir, 'w') as f:
            f.write(training_string)
        bestMatch, probList = self.classify(training_string) #classify
        self._smart_train(bestMatch, probList, category, train_method, file_dir, pr)
        os.remove(file_dir)

    def untrain(self, category, string):
        file_dir = os.path.join(self.directory, 'untrain.tmp')
        with open(file_dir, 'w') as f:
            f.write(string)
        self._untrain(file_dir, category)
        os.remove(file_dir)
    
    # classifies textfile and returns best match and probabilities
    # bestMatch = tuple(bestMatch bestProb, bestPR)
    # probList = [ (twitterHandle1, probability1, pR1) (twitterHandle2, probability2, pR2) ...]
    def classify(self, text):
        output = subprocess.check_output(
                'echo \'' + text + '\'' + ' | ' + 'crm ' + 'classify.crm',
                shell=True, cwd=self.directory)
        #output =  subprocess.check_output('crm ' + os.path.join(self.directory, 'classify.crm') + ' < ' + textFileName, shell=True) #string output from crm114
        #os.remove(textFileName)
        outList = output.split()
        bestMatch = ( str(outList[0]), float(outList[1]), float(outList[2]) ) #(match, prob, pR)
        outList = outList[3:]

        probList = []
        it = iter(outList)
        for x in it:
            x = x.rstrip(':')
            p = (x, float(next(it)), float(next(it)))
            probList.append(p)
            #probList.append((x, float(next(it)), float(next(it)) ))
        
        #probList: (match, probability, pR)
        return (bestMatch, list(probList)) 
    
    def reset(self, corpus=True, crm=False):
        if (corpus is True):
            subprocess.call('rm -f *.css', shell=True) #remove all corpus type files

        if (crm is True):
            subprocess.call('rm -f *.crm', shell=True)

    #note you're going to have to close output yourself
    #http://scikit-learn.org/stable/modules/model_evaluation.html
    #Output: dictionary 
    def evaluate(self, y_true, y_pred, output=sys.stdout, **kwargs): 
        accuracy_score_bool = kwargs.get('accuracy_score', False)
        confusion_matrix_bool = kwargs.get('confusion_matrix', False)
        plot_confusion_matrix = kwargs.get('plot_confusion_matrix', False)
        precision_recall_fscore_support_bool = kwargs.get('precision_recall_fscore_support', False)
        classification_report_bool = kwargs.get('classification_report', False)

        out_dict = {}

        print >>output,'------ EVALUATION STATS ------'
        if accuracy_score_bool:
            #Compute Accuracy Score
            score = accuracy_score(y_true, y_pred)
            out_dict['accuracy_score'] = score
            print >>output, 'Accuracy score (normalized):', accuracy_score(y_true, y_pred)
        print >>output

        if confusion_matrix_bool:
            #Confusion Matrix
            cm = confusion_matrix(y_true, y_pred, labels=self.categories)
            out_dict['confusion_matrix'] = cm
            print >>output, 'Confusion matrix:\n', cm
            print >>output
            if plot_confusion_matrix:
                plt.figure()
                self._plot_confusion_matrix(cm, self.categories)
                plt.show()

        if precision_recall_fscore_support_bool:
            #(precision, recall, f1b, support)
            prfs = precision_recall_fscore_support(y_true, y_pred, labels=self.categories)
            out_dict['precision_recall_fscore_support'] = prfs
            #print >>output, prfs
            #print >>output

        if classification_report_bool:
            #Classification report
            print >>output, classification_report(y_true, y_pred, target_names=self.categories)
        return out_dict

    def print_classify(self, bestMatch, probList, print_location=sys.stdout):
        print >>print_location, 'best match: ' + bestMatch[0]
        print >>print_location, 'match, probability, pr:'
        for tup in probList:
            print >>print_location, '\t', str(tup[0]), str(tup[1]), str(tup[2]) #prints probability and pR

    def crm_files_exist(self):
        #check if crm files exist already
        allFilesExist = True
        for category in self.categories:
            complete_directory = os.path.join(self.directory, category + '.css')
            if(os.path.isfile(complete_directory) is False):
                allFilesExist = False
                break
        return allFilesExist

    #DEPRECATED
    #by default will split training files into lines and train each line
    def _train_textfile(self, category, train_method='TET', delimiter='\n', *args):
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
    
    #DEPRECATED
    #takes filename as args
    def _classify_textfiles(self, *args):
        for textfile in args:
            bestMatch, probList = self._classify(str(textfile))
            self._print_classify(bestMatch, probList)
        #clean_workspace()

    def _create_directory(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
        
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
    def _smart_train(self, match, probList, truth_match_name, train_method, textfilename, prThreshold = 10.0):
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
        trainingTxtFile = ntpath.basename(trainingTxtFile)
        if corpus_name not in self.categories:
            raise ValueError('Category doesn\'t exist!')
        subprocess.call('crm ' + 'learn.crm' + ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True, cwd=self.directory)
    def _untrain(self, trainingTxtFile, corpus_name):
        trainingTxtFile = ntpath.basename(trainingTxtFile)
        subprocess.call('crm ' + 'unlearn.crm' +
                ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True, cwd=self.directory)

    def _create_crm_files(self, file_names, classification_type, word_pat):
        #classification_type = classification_type.rstrip('>').strip('<')
        CLASSIFY_EXT = '.css'    #create files if they don't exist
        UNLEARN_CMD = "{ learn <%s refute> (:*:_arg2:) /%s/}"
        LEARN_CMD = "{ learn <%s> (:*:_arg2:) /%s/}"
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
        learnCRM.write(LEARN_CMD % (classification_type, word_pat))
        learnCRM.close()
        #print 'created learn.crm'

        #create unlearn.crm
        unlearnCRM = open(os.path.join(self.directory, 'unlearn.crm'), 'w')
        unlearnCRM.write(UNLEARN_CMD % (classification_type, word_pat))
        unlearnCRM.close()
        #print 'created unlearn.crm'

        #create classify.crm
        classifyCRM = open(os.path.join(self.directory, 'classify.crm'), 'w')
        name_list = [ntpath.basename(name) + CLASSIFY_EXT for name in file_names]
        match_list = [MATCH_VAR % (name, name, name, name, name) for name in name_list] #create list of MATCH_VARs based on screen name
        output_list = ['%s: :*:%s_prob: :*:%s_pr:' % (os.path.splitext(name)[0], name, name) for name in name_list] #create list for output

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
            cmd = ('crm '+ 'learn.crm' + ' ' +  str(n + CLASSIFICATION_EXT))
            #print cmd
            pipe = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, cwd=self.directory).stdin
            #pipe = os.popen(('crm '+ os.path.join(self.directory,'learn.crm') + ' ' +  str(n + CLASSIFICATION_EXT)), 'w')
            pipe.close()
        #print 'created corpus files'
