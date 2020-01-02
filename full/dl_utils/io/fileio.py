import os, numpy as np
from . import dicom, hdf5
from .legacy import mvk, mzl
from ..printer import * 

def load(fname, infos=None, json_safe=False):
    """
    Method to load a single file and return in NHWC format.

    :params

      (str)  fname: path to file
      (dict) infos: infos dictionary 

    :return

      (np.array) data: data in NHWC format
      (dict)     meta: metadata object (varies by file format)

    """
    # --- (1) If fname is a path string
    if type(fname) is str:

        # --- Check  if file exists
        if not os.path.exists(fname):
            printd('Error file does not exist: %s' % fname)
            return None, None

        # --- Check if file type is supported
        file_type = fname.split('.')[-1]
        if file_type[-1] == '/':
            file_type = 'dcm'

    # --- (2) If fname is a list of DICOM files 
    elif type(fname) is list:
        file_type = 'dcm'

    else:
        file_type = 'unknown'


    load_func = {
        'dcm': load_dcm,
        'npy': load_npy,
        'npz': load_npz,
        'mvk': load_mvk,
        'mzl': load_mzl,
        'hdf5': load_hdf}

    if file_type not in load_func:
        printd('Error file format not recognized: %s' % file_type)
        return None, None

    # --- Initialize infos
    if infos is None:
        infos = {}

    # --- Load
    data, meta = load_func[file_type](fname)

    # --- Convert np.ndarrays to list if needed
    if json_safe:
        convert_meta_to_json_safe(meta)

    return data, meta

def load_hdf(fname):
    """
    Method to load *.hdf5 files

    """
    return hdf5.load(fname=fname)

def load_dcm(fname):
    """
    Method to load *.dcm files

    """
    return dicom.load(path=fname)

def load_npy(fname):
    """
    Method to load *.npy files

    """
    data = add_axes(np.load(fname))

    # --- Check for corresponding affine.npy or dims.npy file
    dims = load_affine(fname)

    return data, {'affine': np.eye(4)} 

def load_npz(fname):
    """
    Method to load *.npz files

    """
    o = np.load(fname)
    data = add_axes(o[o.files[0]])

    # --- Check for corresponding affine.npy or dims.npy file
    affine = load_affine(fname)

    return data, {'affine': affine} 

def load_mvk(fname):

    data, obj = mvk.load(fname, infos={'full_res': True})

    affine = np.eye(4)
    affine[np.arange(3), np.arange(3)] = obj['voxel_size'][1:]

    return data, {'affine': affine} 

def load_mzl(fname):

    data = mzl.decompress(fname, method='full')

    return data, {'affine': np.eye(4)} 

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

def save_hdf(fname, data, meta={}, chunks=None, compression='gzip'):

    hdf5.save(fname, data=data, meta=meta, chunks=chunks, compression=compression)

def save_npz(fname, data, meta={}):

    np.savez_compressed(fname, data)

def add_axes(data):
    """
    Method to ensure data is 4D array 

    """
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    if data.ndim == 3:
         data = np.expand_dims(data, axis=-1)

    return data 

def convert_meta_to_json_safe(meta):

    if 'affine' in meta:
        meta['affine'] = meta['affine'].ravel()[:12].tolist()

    for k, v in meta.items():
        if type(v) is np.ndarray:
            meta[k] = v.tolist()
