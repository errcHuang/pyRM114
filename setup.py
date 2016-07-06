import pyrm114
from distutils.core import setup

VERSION = pyrm114.version

'''
roll with:

git tag VERSION
git push --tags
python setup.py sdist upload -r pypi
'''

setup(
  name = 'pyrm114',
  packages = ['pyrm114'], # this must be the same as the name above
  version = VERSION,
  description = 'Python wrapper for CRM114 classifier',
  author = 'Eric Huang',
  author_email = 'eric.huanghg@gmail.com',
  url = 'https://github.com/errcHuang/pyrm114', # use the URL to the github repo
  download_url = 'https://github.com/errcHuang/pyrm114/tarball/0.1', # I'll explain this in a second
  keywords = ['machine', 'learning', 'crm114', 'pyrm114', 'python', 'wrapper'], # arbitrary keywords
  classifiers = [],
  setup_requires=[
	'requests',
	'numpy',
  ],
  install_requires=[
	'matplotlib>=1.4.3',
	'numpy>=1.11.1',
	'scikit-learn>=0.17.1',
	'requests',
	'jsonschema',
	'json-spec'
  ]
	
)
