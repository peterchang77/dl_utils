import os, yaml, requests, tarfile
from dl_utils.general import printp, printd, tools as jtools

def download(name, path=None, overwrite=False):
    """
    Method to download prepared dataset

    """
    URLS = {
        'ct/bet-demo': 'https://www.dropbox.com/s/siqr7hbj640sckq/bet_demo.tar?dl=1',
        'mr/brats': 'https://www.dropbox.com/s/1rj559ygzows2ry/brats.tar?dl=1',
        'mr/niigzs-demo': 'https://www.dropbox.com/s/l9sdwb8j0hhztzn/niigzs_demo.tar?dl=1',
        'xr/ett-demo': 'https://www.dropbox.com/s/eawf9l1p7la27ky/cxr_ett_demo.tar?dl=1'}

    # --- Check if dataset name exists
    if name not in URLS:
        printd('ERROR provided dataset name is not recognized')
        return
    
    # --- Download
    paths = jtools.get_paths(name)
    if not os.path.exists(paths['data']) or overwrite:
        path = path or '/data/raw/{}'.format(name.replace('/', '_').replace('-', '_'))
        retrieve(URLS[name], path, overwrite)
        jtools.set_paths(path, name)

def retrieve(url, path, overwrite):
    """
    Method to download and unzip remote data archive

    """
    tar = '{}/tars/data.tar'.format(path)

    # --- Download
    if not os.path.exists(tar) or overwrite:
        os.makedirs(os.path.dirname(tar), exist_ok=True)
        pull(url, tar)

    # --- Unarchive 
    if not os.path.exists('{}/proc'.format(path)) or overwrite:
        jtools.unarchive(tar, path)

def pull(url, dst):
    """
    Method to perform pull blocks of data from remote URL

    """
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

# ===============================================================================
# DEPRECATED 
# ===============================================================================

def _unzip(dst, path, overwrite):
    """
    Method to unzip dst *.zip file into path

    """
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

def _set_paths(path):

    # --- Set db path
    ymls = sorted(glob.glob('{}/ymls/db*.yml'.format(path)))
    for yml in ymls:
        db = DB(yml=yml)
        db.set_paths({'data': path, 'code': path})
        db.to_yml(to_csv=False)

    # --- Set _db path
    _db = [] 
    for suffix in ['/db-sum', '/db.', '']:
        _db = [y for y in ymls if suffix in y]
        if len(_db) > 0:
            _db = _db[0]
            break
    
    # --- Set client path
    client = '{}/ymls/client.yml'.format(path)
    if os.path.exists(client) and len(_db) > 0:
        c = yaml.load(open(client, 'r'), Loader=yaml.FullLoader)
        c['_db'] = _db 
        yaml.dump(c, open(client, 'w'))
