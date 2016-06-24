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
try:
    import cPickle as pickle
except:
    import pickle

#authentication stuff for twitter
__consumer_key__ = 'SDSxLoUOU5eNAQEOkvvEwTFKi'
__consumer_secret__ = 'D1ikExbbdX6xZ13IDQrDnXUqTEh1VG5vxZtp7pJ1zg6Hmteent'
__access_token__ = '76215622-BmkW83GMTl1ZB9ctNgVB22TSBo9saoUMByrxp7mW0'
__access_token_secret__ = 'FeUDn5Sdn1yVxLh30Wlg9pgREvc2AQtcq5uSKKUE892EV'

__auth__ = tweepy.OAuthHandler(__consumer_key__, __consumer_secret__)
__auth__.set_access_token(__access_token__, __access_token_secret__)

__api__ = tweepy.API(__auth__)


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

def train_command(args):
    #global vars
    screen_names = args.screen_names #screen names for training
    all_tweets = []
    trainmethod = args.trainmethod

    try:
        all_tweets = grab_tweets(screen_names, args.offline) #grab alltweets
    except tweepy.TweepError as e:
        if e.message[0]['code'] is 34:
            print
            print 'one or more of those twitter handles don\'t exist!'
            print '(tip: check capitalization)'
            sys.exit()
        else:
            raise e
    print 'retrieved all tweets'

    #---------  check flags ----------


    #--trainpartition, check that trainpartion is between 0 and 1
    train_partition = args.trainpartition
    if (not 0.0 <= train_partition <= 1.0):
        sys.exit('--trainpartition must be between 0.0 and 1.0')

    #--algorithm
    create_crm_files(screen_names, args.algorithm)

    #--directory
    __directory__ = args.directory

    training_tweets = []
    test_tweets = []

    #if --eval flag used, divide tweets into training and partitioning sets
    if (args.eval is not None):
        #structure of lists:
        #   training_tweets = [ [list of clintonTweets], [list of sandersTweets], ...]
        #   test_tweets = [ [list of clintonTweets], [list of sandersTweets], ...]
        training_tweets, test_tweets = get_training_and_test_set(train_partition, all_tweets)
    else:
        training_tweets = all_tweets
    #--trainmethod
    if (args.trainmethod is None):
        trainmethod = ['TET', 0] #default Train Every Thing - not recommended...


    #train classifier
    for someones_tweets in training_tweets:
        screen_name = str(someones_tweets[0].author.screen_name)
        someones_tweets_file = write_tweets_to_file(someones_tweets, __directory__, screen_name + '.txt')
        bestMatch, probList = classify(someones_tweets_file)
        smart_train(bestMatch, probList, screen_name, trainmethod, someones_tweets_file) 
    print 'trained classifier.'

    #if --eval flag is used, then classify test set and print statistics
    if (args.eval is not None):
        print 'evaluating algorithm...'


        temp_stats_file = open(os.path.join(__directory__, 'prob_distribution.txt'), 'w')

        y_true = []
        y_pred = []
        print >>temp_stats_file,'------ BEST MATCH AND PROBABILITIES ------'
        for tweets in test_tweets:
            for t in tweets:
                trueAuthor = t.author.screen_name
                tweetFileName = write_tweets_to_file([t], __directory__, trueAuthor + '.txt')
                bestMatch, probList = classify(tweetFileName)

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

def classify_command(args):
    for files in args.textfiles:
        bestMatch, probList = classify(str(files))
        print 'best match: ' + bestMatch
        print 'probabilities:'
        for tup in probList:
            print '\t' + str(tup[0]) + ': ' + str(tup[1])
        print

    #clean_workspace()


def reset_command(args):
   #--corpus/--resetall
    if (args.corpus is True or args.all is True):
        subprocess.call('rm -f *.css', shell=True) #remove all corpus type files

    #--tweets/--all
    if (args.tweets is True or args.all is True):
        subprocess.call('rm -f *.tweets', shell=True)

    if (args.crm is True or args.all is True):
        subprocess.call('rm -f *.crm', shell=True)

    if (args.all is True):
        subprocess.call('rm prob_distribution.txt', shell=True)



def crm_files_exist(screen_names):
    #check if files exit already
    allFilesExist = True
    for name in screen_names:
        complete_directory = os.path.join(__directory__, name + '.css')
        if(os.path.isfile(complete_directory) is False):
            allFilesExist = False
            break
    return allFilesExist


def create_crm_files(screen_names, classification_type):
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
    name_list = [names + CLASSIFY_EXT for names in screen_names]
    match_list = [MATCH_VAR % (names, names, names, names, names) for names in name_list] #create list of MATCH_VARs based on screen name
    output_list = ['%s: :*:%s_prob: :*:%s_pr:' % (names.split('.')[0], names, names) for names in name_list] #create list for output

    classifyCRM.write(CLASSIFY_CMD % (classification_type,
                                      ' '.join(name_list),
                                      ' '.join(match_list),
                                      ' '.join(output_list)
                                     ))
    classifyCRM.close()

    CRM_BINARY = 'crm'
    CLASSIFICATION_EXT = '.css'

    #create corpus files
    for n in screen_names:
        pipe = os.popen(('crm '+ os.path.join(__directory__) + 'learn.crm ' +  str(n + CLASSIFICATION_EXT)), 'w')
        pipe.close()

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

def train(trainingTxtFile, corpus_name):
    subprocess.call('crm ' + os.path.join(__directory__, 'learn.crm') +
            ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True)
def untrain(trainingTxtFile, corpus_name):
    subprocess.call('crm ' + os.path.join(__directory__, 'unlearn.crm') +
            ' ' + (corpus_name+'.css') + ' < '+ trainingTxtFile, shell=True)

#match: (name_of_match, prob_of_match, pr_of_match)
#probList = [ (name1, probability1, pR1) (name2, probability2, pR2) ...]
#train_method: (name_of_method, pR_threshold)
def smart_train(match, probList, truth_match_name, train_method, textfilename):
    prThreshold = train_method[1]
    name_of_match = match[0]
    pr = match[2]
    if train_method[0] == 'TOE': #train on error
        if truth_match_name != name_of_match: #if classifier incorrectly predicts 
            train(textfilename, truth_match_name)
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
                train(textfilename, truth_match_name)
    elif train_method[0] == 'SSTTT':
        if truth_match_name != name_of_match: #if classifier incorrectly predicts 
            train(textfilename, truth_match_name)
        elif pr < prThreshold: #if correct, but match PR is less than PR threshold
            train(textfilename, truth_match_name)
    elif train_method[0] == 'DSTTT':
        if truth_match_name != name_of_match: #if classifier incorrectly predicts 
            train(textfilename, truth_match_name)
        elif pr < prThreshold: #if correct, but match PR is less than PR threshold
            train(textfilename, truth_match_name)

            #if not sure if others were incorrect, then untrain out 
            for tuples in probList:
                name = tuples[0]
                if name != truth_match_name:
                    prWrong = abs(tuples[2])
                    if prWrong < prThreshold:
                        untrain(textfilename, name)
    elif train_method[0] == 'DSTTTR':
        oldPR = match[2]
        if truth_match_name != name_of_match: #if classifier incorrectly predicts 
            train(textfilename, truth_match_name)
        elif pr < prThreshold: #if correctly matched, but match PR is less than PR threshold
            train(textfilename, truth_match_name)

            #reclassify text
            bestMatch, probList = classify(textfilename)
            newMatchName = bestMatch[0]
            newPR = bestMatch[2]

            #if improvement not good enough, then untrain out of incorrect classes
            if (newPR < prThreshold) or abs(newPR-oldPR) > 3:
                for tuples in probList:
                    name = tuples[0]
                    if name != truth_match_name:
                        untrain(textfilename, name)
    elif train_method[0] == 'TTE':
        if truth_match_name != name_of_match: #if classifier incorrectly predicts 
            train(textfilename, truth_match_name)

            #reclassify text
            bestMatch, probList = classify(textfilename)
            newMatchName = bestMatch[0]
            newPR = bestMatch[2]

            #keep training until pr threshold improves to satisfactory level
            loopCount = 0
            while (newPR < prThreshold and loopCount is not 5):
                train(textfilename, truth_match_name)

                #reclassify text
                bestMatch, probList = classify(textfilename)
                newMatchName = bestMatch[0]
                newPR = bestMatch[2]
    elif train_method[0] == 'TUNE':
        print 'nice meme son'
    else: #by default train everything (TET)
        train(textfilename, truth_match_name)



# classifies textfile and returns best match and probabilities
# bestMatch = tuple(bestMatch bestProb, bestPR)
# probList = [ (twitterHandle1, probability1, pR1) (twitterHandle2, probability2, pR2) ...]
def classify(textFileName):
    output =  subprocess.check_output('crm ' + os.path.join(__directory__, 'classify.crm') + ' < ' + textFileName, shell=True) #string output from crm114
    outList = output.split()
    bestMatch = ( str(outList[0]), float(outList[1]), float(outList[2]) ) #(match, prob, pR)
    outList = outList[3:]

    probList = []
    it = iter(outList)
    for x in it:
        x.rstrip(':')
        probList.append((x, float(next(it)), float(next(it)) ))

    #probList: (match, probability, pR)

    return (bestMatch, tuple(probList)) 

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

"""TWITTER RELATED FUNCTIONS"""


#grabs all tweets in list of screen_names and returns one list with lists of tweets
def grab_tweets(screen_names, use_offline = True):
    LOCAL_FILE_EXT = '.tweets'

    all_tweets = []


    for name in screen_names:
        complete_directory = os.path.join(__directory__, name + LOCAL_FILE_EXT)
        if(use_offline and os.path.isfile(complete_directory)):
            tweetFile = open(complete_directory, 'rb')

            print 'loading %s\'s tweets from file...' % name
            tweets = pickle.load(tweetFile)
            tweetFile.close()

            all_tweets.append(tweets) #add that person's tweetlist to big list
        else:
            print 'retrieving %s\'s tweets from twitter' % name
            tweets = get_all_tweets(name, False)

            tf = open(complete_directory, 'wb')

            print 'saving %s\'s tweets to file...' % name
            pickle.dump(tweets, tf)
            tf.close()

            all_tweets.append(tweets) #add person's tweetlist to biglist
    return all_tweets

#source: gist.github.com/yanofsky/5436496
def get_all_tweets(screen_name, include_retweets = True):
    #initialize a list to hold all the tweepy tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = __api__.user_timeline(screen_name = screen_name,count=200)


    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
            print "getting tweets before %s" % (oldest)

            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets =__api__.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

            #save most recent tweets
            alltweets.extend(new_tweets)

            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1

            print "...%s tweets downloaded so far" % (len(alltweets))
    #parse out t.co links - TO-DO
    #alltweets = [for t in alltweets t.text.find('https://t.co')

    if (include_retweets is False): #filter out retweets
        print 'parsing out retweets...'
        alltweets = [t for t in alltweets if (t.text.startswith('RT') is False)
                    and t.text.startswith('"') is False]
    return alltweets

#saves tweets to text file under firstnamelastname.txt by default
#returns filename (string)
#note: make sure tweets isn't empty or else errors
def write_tweets_to_file(tweets, directory, nameoffile = 'lmao.txt'):
    writefile = open(os.path.join(directory,  nameoffile), 'w')
    for t in tweets:
        writefile.write(t.text.encode('ascii', 'ignore') + '\n')
    writefile.close()
    return nameoffile

if __name__ == '__main__':
    main(sys.argv[1:])


