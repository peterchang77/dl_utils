import numpy as np

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

# =============================================
# REGISTERD FUNCTIONS
# =============================================
FUNCS = {
    'coord': calculate_coord,
    'stats': calculate_stats,
    'label': calculate_label}
# =============================================
