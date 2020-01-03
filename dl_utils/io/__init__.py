import os, glob, importlib
from types import ModuleType

def load_modules(repo='dl_utils', submodule='io', allows=['hdf5', 'dicom', 'wrapper']):
    """
    Method to compile submodule from all available libraries

    """
    ROOT = os.environ.get('{}_ROOT'.format(repo).upper(), '.')

    is_hidden = lambda x : x.startswith('_')
    is_module = lambda x, module : isinstance(getattr(module, x), ModuleType) and x not in allows 

    inits = glob.glob('{}/{}/*/{}/__init__.py'.format(ROOT, repo, submodule))
    inits = [i.replace(ROOT, '') for i in inits]

    LOAD_FUNCS = {}
    SAVE_FUNCS = {}

    for init in inits:

        # --- Get path to module
        path = '.'.join(init.split('/')[1:-1])

        # --- Import module
        module = importlib.import_module(path)
        names = [n for n in module.__dict__ if not is_hidden(n) and not is_module(n, module)]
        globals().update({n: getattr(module, n) for n in names})

        # --- Record LOAD_FUNCS / SAVE_FUNCS
        for n in module.__dict__:

            if hasattr(globals().get(n, None), 'LOAD_FUNCS'):
                LOAD_FUNCS.update(getattr(globals()[n], 'LOAD_FUNCS'))

            if hasattr(globals().get(n, None), 'SAVE_FUNCS'):
                SAVE_FUNCS.update(getattr(globals()[n], 'SAVE_FUNCS'))

    # ==========================================================
    # FILEIO
    # ==========================================================

    # --- Load module
    fileio = importlib.import_module('dl_utils.core.io.fileio')

    # --- Register load funcs
    fileio.LOAD_FUNCS.update(LOAD_FUNCS)
    fileio.SAVE_FUNCS.update(SAVE_FUNCS)

    # --- Update io namespace
    globals()['load'] = fileio.load
    globals()['save'] = fileio.save
    globals()['LOAD_FUNCS'] = fileio.LOAD_FUNCS
    globals()['SAVE_FUNCS'] = fileio.SAVE_FUNCS

# --- Load modules
load_modules()

# --- Remove imports from namespace
for name in ['os', 'glob', 'importlib', 'ModuleType', 'load_modules', 'name']:
    globals().pop(name)
