import os, glob, yaml, requests
from zipfile import ZipFile
from dl_utils.general import *
from dl_utils.db import DB

def download(name, path='/data/raw', overwrite=False):
    """
    Method to download prepared dataset

    """
    URLS = {
        'bet': 'https://www.dropbox.com/s/051khxpy3s1vmxs/bet.zip?dl=1',
        'jars': 'https://www.dropbox.com/s/21itp32v9ht0czh/jars.zip?dl=1',
        'brats': 'https://www.dropbox.com/s/wuady574manrwew/brats.zip?dl=1',
        'niigzs': 'https://www.dropbox.com/s/c1gprnesbcmg5c6/niigzs.zip?dl=1',
        'cxr_ett': 'https://www.dropbox.com/s/9dmmmgvzcbwy7ww/cxr_ett.zip?dl=1'}

    if name not in URLS:
        printd('ERROR provided dataset name is not recognized')
    
    path = '{}/{}'.format(path, name)
    retrieve(URLS[name], path, overwrite)

def retrieve(url, path, overwrite):
    """
    Method to download and unzip remote data archive

    """
    # --- Download
    dst= '{}/zips/raw.zip'.format(path)

    if not os.path.exists(dst) or overwrite:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        pull(url, dst)

    # --- Unzip
    unzip(dst, path, overwrite)

    # --- Set paths
    set_paths(path)
    
def pull(url, dst):
    
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 32768
    dload_size = 0

    with open(dst, 'wb') as f:
        for data in r.iter_content(block_size):
            dload_size += len(data)
            printp('Downloading dataset to {} ({:0.3f} MB / {:0.3f} MB)'.format(
                dst, dload_size / 1e6, total_size / 1e6), dload_size / total_size)
            f.write(data)

    printp('Completed dataset download to {} ({:0.3f} MB / {:0.3f} MB)'.format(
        dst, dload_size / 1e6, total_size / 1e6), dload_size / total_size)
            
def unzip(dst, path, overwrite):
    
    zf = ZipFile(dst)

    fnames = zf.infolist()
    total_size = sum([f.file_size for f in fnames])
    unzip_size = 0

    for f in fnames:
        if f.filename[-1] != '/':
            unzip_size += f.file_size
            printp('Extracting zip archive ({:0.3f} MB / {:0.3f} MB)'.format(
                unzip_size / 1e6, total_size / 1e6), unzip_size / total_size)

            if not os.path.exists('{}/{}'.format(path, f.filename)) or overwrite:
                zf.extract(f, path)

    printp('Completed archive extraction ({:0.3f} MB / {:0.3f} MB)'.format(
        unzip_size / 1e6, total_size / 1e6), unzip_size / total_size)

def set_paths(path):

    # --- Set db path
    ymls = glob.glob('{}/ymls/db*.yml'.format(path))
    for yml in ymls:
        db = DB(yml=yml)
        db.set_paths({'data': path, 'code': path})
        db.to_yml(to_csv=False)
    
    # --- Set client path
    client = '{}/ymls/client.yml'.format(path)
    if os.path.exists(client):
        c = yaml.load(open(client, 'r'), Loader=yaml.FullLoader)
        c['_db'] = '{}/ymls/db.yml'.format(path)
        yaml.dump(c, open(client, 'w'))
