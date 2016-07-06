from pyrm114 import pyrm114

#
p = pyrm114(['a','b'])
p.train('a', 'ay lmao')
p.train('b', 'b rip')
p.classify('lmao')
print p.crm_files_exist()
p.reset()
print p.crm_files_exist()
