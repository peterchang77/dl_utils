import argparse
from dl_utils import datasets

def download(name):

    if name == 'bet':
        datasets.download(name='bet', path='.')

    if name == 'jars':
        datasets.download(name='jars', path='.')

if __name__ == '__main__':

    description = 'Download test data'
    usage = 'python download.py [-h] name'

    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('name', metavar='name', type=str, help='name of dataset (bet, etc...)')
    args = parser.parse_args()

    download(args.name)
