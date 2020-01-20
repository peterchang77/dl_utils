import numpy as np

def init(funcs_def='mr_train', **kwargs):
    """
    Method to to prepare **kwargs for db.apply(...)
    
    :params

      (str)  funcs_def = 'mr_train', 'ct_train', ... OR

      (list) funcs_def = [{

        'func': 'coord', 'stats', ... OR lambda function,
        'kwargs': {
            kwargs_0: colkey_0,
            kwargs_1: colkey_1,
            ...},

        }]

        kwargs_0 ==> named argument of lambda function
        colkey_0 ==> column name to feed into kwargs

    """
    if type(funcs_def) is str:
        funcs_def = get_default_funcs_def(funcs_def, **kwargs)

    assert type(funcs_def) is list

    # --- Update funcs
    update(kwargs.get('FUNCS', {}))

    # --- Parse apply kwargs
    return {
        'mask': kwargs.get('mask', None),
        'load': kwargs.get('load', None),
        'funcs': [FUNCS.get(d['func'], d['func']) for d in funcs_def],
        'kwargs': [d['kwargs'] for d in funcs_def]}

def update(funcs):
    """
    Method to update FUNCS with dict of additional methods

    """
    FUNCS.update(funcs)

def calculate_coord(arr):
    """
    Method to calculate normalized coord positions

    """
    z = arr.shape[0]

    return {'coord': np.arange(z) / (z - 1)}

def calculate_stats(arr, name, axis=(0, 1, 2, 3)):
    """
    Method to calculate image statistics across channels: mu, sd

    """
    mu = arr.mean(axis=tuple(axis))
    sd = arr.std(axis=tuple(axis))

    mu = {'{}-mu'.format(name[1:]): mu} 
    sd = {'{}-sd'.format(name[1:]): sd} 

    return {**mu, **sd}

def calculate_label(arr, name, classes, axis=(1, 2, 3)):
    """
    Method to calculate if label class is present 

    """
    is_present = lambda c : np.sum(arr == c, axis=tuple(axis)) > 0

    return {'{}-{:02d}'.format(name[1:], c): is_present(c) for c in range(classes)}

def calculate_slices(arr):
    """
    Method to calculate total number of slices in volume

    """
    return {'slices': arr.shape[0]}

# ============================================================
# REGISTERED FUNCTIONS
# ============================================================

FUNCS = {
    'coord': calculate_coord,
    'stats': calculate_stats,
    'label': calculate_label,
    'slices': calculate_slices}

# ============================================================
# DEFAULT FUNCS_DEF
# ============================================================

def get_default_funcs_def(func_def, dats=['dat'], lbls=['lbl'], classes=2, **kwargs):

    if func_def not in ['mr_train', 'xr_train', 'ct_train']:
        return []

    # --- Create generic label-derived stats
    fdef = [{
        'func': 'coord',
        'kwargs': {'arr': lbls[0]}}]

    fdef += [{
        'func': 'label',
        'kwargs': {'arr': l, 'name': '!' + l, 'classes': classes}
        } for l in lbls]

    # --- Add data-specific stats (if needed)
    if func_def in ['mr_train', 'xr_train']:

        fdef = fdef[0:1] + [{
            'func': 'stats',
            'kwargs': {'arr': d, 'name': '!' + d}
            } for d in dats] + fdef[1:]

    return fdef

# ============================================================
