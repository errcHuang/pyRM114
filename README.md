# pyRM114: python wrapper for CRM114 

Makes CRM114 beautiful and easy to use via a python wrapper interface

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

##Installation
`pip install pyrm114`

##Usage:
The basic usage pipeline for realpolitalk is to _train_ then _classify_ (and _reset_) as needed

```python
from pyrm114 import pyrmClassifier

p = pyrmClassifier(['Barack_Obama', 'Donald_Trump']) #initialize by specifying classifier categories

#training with strings
p.train('Barack_Obama', 'Change we can believe in')
p.train('Donald_Trump', 'Make America great again')

#classifying
p.classify('change we can')

#best match: Barack_Obama
#match, probability, pr:
#	Barack_Obama: 0.789 0.57
#	Donald_Trump: 0.211 -0.57


#resetting (deleting the trained classifier)
p.reset()
```
##Advanced Usage 

To be added...

##Misc.
###CRM114
[CRM114](crm114.sourceforge.net) is basically a programming language/engine that is centered entirely around parsing and learning/classifying text streams. 

Originally used for spam classification, CRM114 is super fast (written in C) and wildly accurate (>99.9%). You can basically plug-and-play with different algorithms (Hidden Markov Model, OSB, winnow, bit entropy, etc.) with relative ease.


