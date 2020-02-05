import nibabel as nib
from dl_utils import io
from dl_utils.general import *

def to_hdf5(fname):

    img = nib.load(fname)
    aff = img.affine

    # --- Flip axes
    vol = np.swapaxes(img, 0, 2)

    # --- Flip affine
    aff_ = np.eye(4)
    aff_[:3, :3] = aff[2::-1, 2::-1]
    aff_[:3, 3:] = aff[2::-1, 3:]

    # --- Invert yx affine values
    aff_[:2, :3] *= -1

    # --- Align with standard

def from_hdf5():

    pass

def align_with_standard(data, affine):

    if orient == 'AXI':

        pass

def orient_zyx(affine):

    return ['AXI', 'COR', 'SAG'][np.argmax(np.abs(affine[:3, 0]))]

def orient_xyz(affine):

    return ['SAG', 'COR', 'AXI'][np.argmax(np.abs(affine[:3, 2]))]
