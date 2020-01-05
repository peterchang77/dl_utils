import numpy as np

def init(funcs_def='mr_train', **kwargs):
    """
    Method to to prepare **kwargs for db.apply(...)

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
        'funcs': [FUNCS[d['func']] for d in funcs_def],
        'kwargs': [d['kwargs'] for d in funcs_def]}

def update(funcs):
    """
    Method to update FUNCS with dict of additional methods

    """
    FUNCS.update(funcs)

def calculate_coord(lbl):
    """
    Method to calculate normalized coord positions

    """
    z = lbl.shape[0]

    return {'coord': np.arange(z) / (z - 1)}

def calculate_stats(dat):
    """
    Method to calculate image statistics across channels: mu, sd

    """
    mu = dat.mean(axis=(0, 1, 2))
    sd = dat.std(axis=(0, 1, 2))

    mu = {'mu-{:02d}'.format(c): m for c, m in enumerate(mu)} 
    sd = {'sd-{:02d}'.format(c): s for c, s in enumerate(sd)} 

    return {**mu, **sd}

def calculate_label(lbl, classes):
    """
    Method to calculate if label class is present 

    """
    is_present = lambda c : np.sum(lbl == c, axis=(1, 2, 3)) > 0

    return {'lbl-{:02d}'.format(c): is_present(c) for c in range(classes + 1)}

# ============================================================
# REGISTERED FUNCTIONS
# ============================================================

FUNCS = {
    'coord': calculate_coord,
    'stats': calculate_stats,
    'label': calculate_label}

# ============================================================
# DEFAULT FUNCS_DEF
# ============================================================

def get_default_funcs_def(func_def, **kwargs):

    # --- Get defaults
    mapping = kwargs.get('mapping', {'dat': 'dat', 'lbl': 'lbl'})
    classes = kwargs.get('classes', 2)

    if func_def in ['mr_train', 'xr_train']:

        return [{

            'func': 'coord',
            'kwargs': {'lbl': mapping['lbl']}}, {

            'func': 'stats',
            'kwargs': {'dat': mapping['dat']}}, {

            'func': 'label',
            'kwargs': {'lbl': mapping['lbl'], 'classes': classes}

        }]

    if func_def in ['ct_train']:

        return [{

            'func': 'coord',
            'kwargs': {'lbl': mapping['lbl']}}, {

            'func': 'label',
            'kwargs': {'lbl': mapping['lbl'], 'classes': classes}

        }]

    return []

# ============================================================
