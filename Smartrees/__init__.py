from os.path import isfile
from os.path import dirname

version_file = '{}/version.txt'.format(dirname(__file__))
if isfile(version_file):
    with open(version_file) as version_file:
        __version__ = version_file.read().strip()

print(__file__)

from os.path import join
from dotenv import load_dotenv
# ...
env_path = join(dirname(dirname(__file__)),'.env') # ../.env
load_dotenv(dotenv_path=env_path)
