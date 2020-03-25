import os, yaml, tarfile
from .printer import printp

# ===============================================================================
# PATH MANIPULATION 
# ===============================================================================

def load_configs(name, dirname='.jarvis'):
    """
    Method to load Jarvis configuration file

    """
    fname = '{}/{}/{}.yml'.format(os.environ.get('HOME', '.'), dirname, name)

    configs = {}
    if os.path.exists(fname):
        with open(fname, 'r') as y:
            configs = yaml.load(y, Loader=yaml.FullLoader)

    return configs

def save_configs(configs, name, dirname='.jarvis'):
    """
    Method to save Jarvis configuration file

    """
    fname = '{}/{}/{}.yml'.format(os.environ.get('HOME', '.'), dirname, name)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as y:
        yaml.dump(configs, y)

def set_paths(project_id, paths):
    """
    Method to update global $HOME/.jarvis/paths.yml with new paths

    :params

      (str) project_id 
      (str) paths

    """
    # --- Load
    configs = load_configs('paths')

    # --- Configure paths
    if type(paths) is str:
        paths = {'code': paths, 'data': paths}

    assert type(paths) is dict

    remove_slash = lambda x : x[:-1] if x[-1] == '/' else x
    paths = {**{'code': '', 'data': ''}, **paths}
    paths = {k: remove_slash(v) if v != '' else v for k, v in paths.items()}

    configs[project_id] = paths

    # --- Save
    save_configs(configs, 'paths')

    return configs[project_id] 

def get_paths(project_id):
    """
    Method to read global $HOME/.jarvis/paths.yml

    :params

      (str) project_id : if None, return all paths

    """
    # --- Load
    configs = load_configs('paths')

    # --- TODO: determine project_id based on ENV var

    return {**{'code': None, 'data': None}, **configs.get(project_id, {})}

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
