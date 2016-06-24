# realpolitalk: a machine learning tweet-based approach to understanding human speech 
realpolitalk takes tweets from Twitter users and machine learns them. 

The vision behind the project is to be able to take someone's tweets, learn how they write/talk from them, and be able to classify their speech patterns in any medium, whether an essay, book, speech, etc.

##Requirements:
- Python 2.7 
- CRM-114 (installation instructions below)

###Debian/Ubuntu

`sudo apt-get install crm114`

###Red Hat/Fedora

`sudo dnf install crm114`

###Everyone Else

```
## If you do not yet have libtre and its headers:
curl -O http://crm114.sourceforge.net/tarballs/tre-0.7.5.tar.gz
tar -zxf tre-*.tar.gz
cd tre-*
./configure --enable-static
make
sudo make install
cd ..

curl -O http://crm114.sourceforge.net/tarballs/crm114-20100106-BlameMichelson.src.tar.gz
tar -zxf crm114-*.tar.gz
cd crm114*.src
make
sudo make install
cd ..
```

- Twitter API access (consumer key, consumer secret, access token, [excellent tutorial here on accessing Twitter API](http://pythoncentral.io/introduction-to-tweepy-twitter-for-python/)
- Other requirements (tweepy, numpy, scikit-learn, matplotlib, etc.) can be installed with this command.

`pip install -r requirements.txt`

##Usage:
The basic usage pipeline for realpolitalk is to _train_ then _classify_ (and _reset_) as needed.

`python realpolitalk.py {train, classify, reset}`

###Training
The most basic usage is to enter the 'train' command and then enter _any_ number of screen names that you want to train the algorithm on.

`python realpolitalk.py train HillaryClinton realDonaldTrump BernieSanders PRyan`

###Classifying 
After training realpolitalk on tweets, you can enter the 'classify' command and enter _any_ number of textfiles you want classified. 

The result is a best match (whose speech pattern the textfile most resembles) and the distribution of probabilities for the other choices.

`python realpolitalk.py classify goldmansachs_transcript.txt make_america_great_again.txt`
####Example output (of the above command)

```
best match: HillaryClinton
probabilities:
  HillaryClinton:: 1.0
  realDonaldTrump:: 5.81e-31
  BernieSanders:: 1.44e-22
  PRyan:: 1.64e-56

best match: realDonaldTrump
probabilities:
  HillaryClinton:: 1.47e-36
  realDonaldTrump:: 1.0
  BernieSanders:: 2.94e-12
  PRyan:: 1.7e-86
```

###Resetting
If you want to start fresh, maybe train with different users, you can use the 'reset' command to delete all your trained corpuses (the already trained algorithm in .css file format) and tweets.

`python realpolitalk.py reset --all`

##Advanced Usage
###Evaluation statistics generation 

The most important flag is the **'--eval'** flag, which generates a number of statistics (confusion matrix, precision recall) and either prints them to stdout or to a file 

Usage is as follows:

```python
python realpolitalk.py train HillaryClinton realDonaldTrump --eval #no arguments to print to stdout

python realpolitalk.py train HillaryClinton realDonaldTrump --eval statistics.txt #file_to_write_stats_to.txt
```

(note: best match/probability distribution will always be put in 'prob_distribution.txt' on each run). 

###Changing algorithm options
By default, realpolitalk uses an OSB classifier (osb unique microgroom). The classifier type can be changed via the '-a' flag.

Example usage is as follows:
```python
#use Entropy-type classifier instead of OSB, and print evaluation stats to stdout
python realpolitalk.py train HillaryClinton realDonaldTrump -a 'entropy unique crosslink' --eval
```

The full list of CRM114 classifiers can be found [here](http://i.imgur.com/okAhS8l.png).


###Other options
A complete list of options can be found via:
```python
python realpolitalk.py train -h
python realpolitalk.py classify -h
python realpolitalk.py reset -h
```


##Misc.
###CRM114
[CRM114](crm114.sourceforge.net) is basically a programming language/engine that is centered entirely around parsing and learning/classifying text streams. 

Originally used for spam classification, CRM114 is super fast (written in C) and wildly accurate (>99.9%). You can basically plug-and-play with different algorithms (Hidden Markov Model, OSB, winnow, bit entropy, etc.) with relative ease.


###Origin of name
Realpolitalk was originally designed to only analyze the tweets of politicians hence the [reference in the name](https://en.wikipedia.org/wiki/Realpolitik).

