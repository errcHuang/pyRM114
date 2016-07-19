version = '0.6'
from interface import pyrmClassifier

def check_version():
    """
    Tells you if you have an old version of pyrm114.
    """
    import requests
    r = requests.get('https://pypi.python.org/pypi/pyrm114/json').json()
    r = r['info']['version']
    if r != version:
        print("A newer version of pyrm114 is available. " +
              "'pip install -U pyrm114' to update.")
    return r

__license__ = '(c) 2016 Eric Huang. ' \
              'MIT License. See LICENSE file for details.'
__all__ = ['pyrmClassifier', 'pyrmEval']
