import os, json, numpy as np
from ..general import printd

def parse_ext(fname):
    """
    Method to parse file extension

    """
    file_type = 'unknown'

    # --- (1) If fname is a path string
    if type(fname) is str:

        file_type = fname.split('.')[-1]

        if file_type[-1] == '/':
            file_type = 'dcm'

    # --- (2) If fname is a list of DICOM files 
    elif type(fname) is list:
        file_type = 'dcm'

    return file_type

def load(fname, json_safe=False, **kwargs):
    """
    Method to load a single file and return in NHWC format.

    :params

      (str) fname: path to file

    :return

      (np.array) data: data in NHWC format
      (JSON obj) meta: metadata object (varies by file format)

    """
    # --- Check if file exists
    if not os.path.exists(fname):
        printd('ERROR file does not exist: %s' % fname)
        return None, None

    # --- Check file ext
    file_type = parse_ext(fname)

    if file_type not in LOAD_FUNCS:
        printd('ERROR file format not recognized: %s' % file_type)
        return None, None

    # --- Load
    data, meta = LOAD_FUNCS[file_type](fname, **kwargs)
    meta = {**{'header': None, 'affine': None, 'unique': None}, **meta}

    # --- Convert np.ndarrays to list if needed
    if json_safe:
        convert_meta_to_json_safe(meta)

    return data, meta

def load_npy(fname, **kwargs):
    """
    Method to load *.npy files

    """
    data = add_axes(np.load(fname))

    # --- Check for corresponding affine.npy or dims.npy file
    dims = load_affine(fname)

    return data, {'affine': np.eye(4)} 

def load_npz(fname, **kwargs):
    """
    Method to load *.npz files

    """
    o = np.load(fname)
    data = add_axes(o[o.files[0]])

    # --- Check for corresponding affine.npy or dims.npy file
    affine = load_affine(fname)

    return data, {'affine': affine} 

def load_affine(fname):
    """
    Method to load corresponding affine.npy or dims.npy file if present

    """
    affine = np.eye(4)

    # --- Check for affine
    fname_affine = '%s/affine.npy' % os.path.dirname(fname)
    if os.path.exists(fname_affine):
        affine = np.load(fname_affine)
        return affine 

    # --- Check for dims 
    fname_dims = '%s/dims.npy' % os.path.dirname(fname)
    if os.path.exists(fname_dims):
        dims = np.load(fname_dims)
        affine[np.arange(3), np.arange(3)] = dims
        return affine 

    return affine

def add_axes(data):
    """
    Method to ensure data is 4D array 

    """
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    if data.ndim == 3:
         data = np.expand_dims(data, axis=-1)

    return data 

def load_json(fname, **kwargs):
    """
    Method to load JSON (meta-only) files

    """
    unique = json.load(open(fname, 'r'))

    return _, {'unique': unique} 

def convert_meta_to_json_safe(meta):

    if 'affine' in meta:
        meta['affine'] = meta['affine'].ravel()[:12].tolist()

    for k, v in meta.items():
        if type(v) is np.ndarray:
            meta[k] = v.tolist()

def save(fname, data, **kwargs):
    """
    Method to save a single file in format implied by file extension 

    :params

      (str) fname: path to file

    """
    # --- Check file ext
    file_type = parse_ext(fname)

    if file_type not in SAVE_FUNCS:
        printd('ERROR file format not recognized: %s' % file_type)

    # --- Save 
    SAVE_FUNCS[file_type](fname, data, **kwargs)

def save_npz(fname, data, **kwargs):

    np.savez_compressed(fname, data)

# ==========================================================
# LOAD / SAVE FUNCTIONS
# ==========================================================
# 
# The following Python dict is used to register file exts
# with implemented load / save functions:
# 
# ==========================================================

LOAD_FUNCS = {
    'npy': load_npy,
    'npz': load_npz,
    'json': load_json}

SAVE_FUNCS = {
    'npz': save_npz}

# ==========================================================
