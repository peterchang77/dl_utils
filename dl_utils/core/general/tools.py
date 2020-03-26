import os, glob, yaml, tarfile
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

def set_paths(paths, project_id, version_id=None):
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
    paths = {k: remove_slash(v or '/') for k, v in paths.items()}

    # --- Version
    paths['code'] = path_sub_version(paths['code'], version_id)

    configs[project_id] = paths

    # --- Save
    save_configs(configs, 'paths')

    return configs[project_id] 

def get_paths(project_id, version_id=None, configs=None):
    """
    Method to read global $HOME/.jarvis/paths.yml

    NOTE: if version_id is provided (or set as ENV), paths['code'] will be updated

    """
    # --- Load
    configs = configs or load_configs('paths')
    paths = {**{'code': '', 'data': ''}, **configs.get(project_id, {})}

    # --- Add version if available
    version_id = version_id or os.environ.get('JARVIS_VERSION_ID', None)
    if version_id is not None and paths['code'] != '':
        paths['code'] = path_add_version(paths['code'], version_id)

    return paths 

def path_add_version(code, version_id):
    """
    Method to intergrate version_id into code path

    """
    if version_id is None:
        return code

    if code != '':
        if not os.path.exists(code):
            code = '{}/{}/data'.format(code[:-5], version_id) if code[-5:] == '/data' else \
                '{}/{}'.format(code, version_id)

    return code

def path_sub_version(code, version_id):
    """
    Method to remove version_id from code path

    """
    if version_id is None:
        return code

    if version_id in code:
        code = code.replace('/{}'.format(version_id), '')

    return code

def autodetect(path='', pattern=None):
    """
    Method to autodetect project_id, version_id, paths and files

    For version detection, the following dir structure is assumed:

      * full path ==> {root}/[v][0-9]/data (real)
      * code path ==> {root}/data (stored in ~/.jarvis/path.yml)

    """
    # --- Read defaults based on ENV variables
    project_id = os.environ.get('JARVIS_PROJECT_ID', None) 
    version_id = os.environ.get('JARVIS_VERSION_ID', None) 
    paths = get_paths(project_id, version_id)

    # --- Attempt extraction at active path for inference
    path = autodetect_active_path(path, pattern, paths, project_id, version_id)

    trim = lambda x : x[:-5] if x[-5:] == '/data' else x
    is_subpath = lambda x : (x in path) and (x != '')

    # --- Attempt extraction of project_id 
    if project_id is None:

        # --- Loop through available paths
        configs = load_configs('paths')
        for pid in configs:

            # --- Get expanded paths
            p = get_paths(pid, version_id=version_id, configs=configs)

            if is_subpath(trim(p['code'])) or is_subpath(p['data']):
                project_id = pid
                paths = p

                break

    # --- Attempt extraction of version_id
    if version_id is None and project_id is not None:

        if is_subpath(trim(paths['code'])) and trim(paths['code']) != path:

            suffix = path.split(trim(paths['code']))[1][1:]
            if '/' in suffix:
                suffix = suffix.split('/')[0]
            if len(suffix) > 1:
                if suffix[0] == 'v' and suffix[1].isnumeric():
                    version_id = suffix
                    paths['code'] = path_add_version(paths['code'], version_id)

    # --- Attempt extraction of files
    files = autodetect_files(pattern, paths, project_id, version_id)

    trim = lambda x : os.path.abspath(x).replace(paths['code'], '') if x is not None else None
    files['yml'] = trim(files['yml']) 
    files['csv'] = trim(files['csv']) 

    return project_id, version_id, paths, files

def autodetect_active_path(path, pattern, paths, project_id=None, version_id=None):
    """
    Method to determine active path to use for autodetect(...) function

    Inference is based off of the following priority:

      (1) Provided path (if exists)
      (2) JARVIS_{*}_FILE ENV variable
      (3) Current working directory

    """
    if os.path.exists(path):
        return path

    # --- Check ENV
    for ext in ['yml', 'csv']:
        fname = autodetect_filepath(ext, pattern, paths, project_id, version_id)
        if os.path.exists(fname or ''):
            return fname

    # --- Check CWD
    return os.getcwd()

def autodetect_files(pattern=None, paths=None, project_id=None, version_id=None):
    """
    Method to autodetect relative files based on ENV variables and provided patterns

    """
    # --- Infer files fully from yml if possible
    yml = autodetect_filepath('yml', pattern, paths, project_id, version_id)
    if os.path.exists(yml or ''):
        y = yaml.load(open(yml, 'r'), Loader=yaml.FullLoader)
        return y.get('files', {'csv': None, 'yml': yml})

    # --- Infer csv alone from autodetect
    csv = autodetect_filepath('csv', pattern, paths, project_id, version_id)
    return {'yml': None, 'csv': csv}

def autodetect_filepath(ext='yml', pattern=None, paths=None, project_id=None, version_id=None):
    """
    Method to autodetect full file path based on ENV variables and provided pattern

    Inference is based off of the following priority:

      (1) JARVIS_{*}_FILE ENV variable - use as full path if exists
      (2) JARVIS_{*}_FILE ENV variable - use as pattern if nonempty 
      (3) pattern = ... (provided kwarg) 
      (4) pattern = '*' 

    :params

      (bool) remove_root : if True, attempt to remove root to path

      NOTE: remove_root is NOT guaranteed to work (e.g. if JARVIS_{}_FILE is set)

    """
    fname = os.environ.get('JARVIS_{}_FILE'.format(ext.upper()), None)
    if os.path.exists(fname or ''):
        return fname 

    # --- Determine code path 
    paths = paths or {}
    paths = {**{'code': None, 'data': None}, **paths}

    if paths['code'] is None:
        paths = get_paths(project_id, version_id)

    if not os.path.exists(paths['code']):
        return None 

    # --- Search
    pattern = fname or pattern or '*'
    matches = glob.glob('{}/{}s/{}.{}'.format(paths['code'], ext, pattern, ext))
    fname = sorted(matches)[0] if len(matches) > 0 else '' 

    return fname

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
