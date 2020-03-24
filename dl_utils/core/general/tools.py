import os, yaml, tarfile
from .general import printp

# ===============================================================================
# PATH MANIPULATION 
# ===============================================================================

def load_configs(name, dirname='.jarvis'):
    """
    Method to load Jarvis configuration file

    """
    fname = '{}/{}/{}'.format(os.environ.get('HOME', '.'), dirname, name)

    configs = {}
    if os.path.exists(fname):
        with open(fname, 'r') as y:
            configs = yaml.load(y, Loader=yaml.FullLoader)

    return configs

def save_configs(configs, name, dirname='.jarvis'):
    """
    Method to save Jarvis configuration file

    """
    fname = '{}/{}/{}'.format(os.environ.get('HOME', '.'), dirname, name)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as y:
        yaml.dump(configs.to_dict(), y, sort_keys=False)

def set_paths(context_id, paths):
    """
    Method to update global $HOME/.jarvis/paths.yml with new paths

    :params

      (str) context_id
      (str) paths

    """
    # --- Load
    configs = load_configs('paths')

    # --- Configure paths
    if type(paths) is str:
        paths = {'code': paths, 'data': paths}

    configs['context_id'] = {**{'code': None, 'data': None}, **paths}

    # --- Save
    save_configs(configs, 'paths')

def get_paths(context_id):
    """
    Method to read global $HOME/.jarvis/paths.yml

    :params

      (str) context_id : if None, return all paths

    """
    # --- Load
    configs = load_configs('paths')

    return {**{'code': None, 'data': None}, **configs.get(context_id, {})}

# ===============================================================================
# TAR TOOLS 
# ===============================================================================

def unarchive(tar, path='.'):
    """
    Method to unpack *.tar(.gz) archive (and sort into appropriate folders)

    """
    with tarfile.open(tar, 'r:*') as tf:
        N = len(tf.getnames())
        for n, t in enumerate(tf):
            printp('Extracting archive ({:07d} / {:07d})'.format(n + 1, N), (n + 1) / N)
            tf.extract(t, path)
