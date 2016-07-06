import subprocess
from pyrm114 import *

#
p = pyrmClassifier(['a','b'])
p.train('a', 'ay lmao')
p.train('b', 'b rip')
p.classify('lmao')
with open('rip.txt', 'w') as f:
    p.classify('lmao', f)
subprocess.call('cat rip.txt')
os.remove('rip.txt')
print p.crm_files_exist()
p.reset()
print p.crm_files_exist()
