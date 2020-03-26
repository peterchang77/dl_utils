import argparse
from dl_utils import datasets

def download(name):

    datasets.download(name=name, path='.')

if __name__ == '__main__':

    description = 'Download test data'
    usage = 'python download.py [-h] name'

    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('name', metavar='name', type=str, help='name of dataset (ct/bet-demo, etc...)')
    args = parser.parse_args()

    download(args.name)
