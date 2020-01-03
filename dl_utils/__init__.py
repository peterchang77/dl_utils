import os, glob, importlib
from types import ModuleType

def load_modules(repo='dl_utils'):
    """
    Method to load all submodules (with __init__.py) relative to package root

    """
    ROOT = os.environ.get('{}_ROOT'.format(repo).upper(), '.')
    libs = ['core', 'edge']

    is_hidden = lambda x : x.startswith('_')
    is_module = lambda x, module : isinstance(getattr(module, x), ModuleType)

    for lib in libs:

        inits = glob.glob('{}/{}/{}/*/__init__.py'.format(ROOT, repo, lib))
        inits = [i.replace(ROOT, '') for i in inits]

        for init in inits:

            # --- Get path to module
            path = '.'.join(init.split('/')[1:-1])

            # --- Import module
            module = importlib.import_module(path)
            names = [n for n in module.__dict__ if not is_hidden(n) and not is_module(n, module)]
            globals().update({n: getattr(module, n) for n in names})

# --- Load modules
load_modules()

# --- Remove imports from namespace
for name in ['os', 'glob', 'importlib', 'ModuleType', 'load_modules', 'name']:
    globals().pop(name)
