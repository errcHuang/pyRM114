version = '0.31'

def check_version():
    """
    Tells you if you have an old version of ndio.
    """
    import requests
    r = requests.get('https://pypi.python.org/pypi/pyrm114/json').json()
    r = r['info']['version']
    if r != version:
        print("A newer version of ndio is available. " +
              "'pip install -U ndio' to update.")
    return r

#__license__ = '(c) 2016 Eric Huang. ' \
#              'MIT License. See LICENSE file for details.'
#__all__ = ['pyrm114']
