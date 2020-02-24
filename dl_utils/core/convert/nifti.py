import numpy as np
import nibabel as nib
from dl_utils import io

def from_nifti(src, dst=None):
    """
    Method to convert *.nii.gz to alternate format (e.g. *.hdf5)

    Note, by default the affine matrices loaded by Nibabel have x- and y- direction consines inverted 

    Thus, algorithm proceeds as follows:

      (1) Flip x- and y- direction cosines 

          * img_raw is used to track pixel data 
          * img_aff is used to track affine matrix

      (2) Align with standard spaces
      (3) Invert axes

    """
    img_raw = nib.load(src)

    # --- Flip xy direction cosines 
    img_aff = flip_xy(img_raw)

    # --- Align with standard
    img_raw, img_aff = align_with_standard(img_raw, img_aff)

    # --- Invert axes
    data = img_raw.get_fdata()
    data = np.swapaxes(data, 0, 2)

    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)

    # --- Invert affine
    meta = {'affine': invert_affine(img_aff.affine)}

    # --- Save
    if dst is not None:
        io.save(dst, data, meta)

    return data, meta

def to_nifti(src, dst=None):
    """
    Method to convert alternate format to *.nii.gz

    """
    pass

def flip_xy(img):
    """
    Method to flip x- and y- direction cosines

    Note that the axes denoting the x- and y- direction cosines differ based on orientation

    """
    orient = orient_xyz(img.affine)

    o = {
        'AXI': (-1, -1, +1),
        'COR': (-1, +1, -1),
        'SAG': (-1, +1, -1)}[orient]

    o = np.stack((np.array([0, 1, 2]), o)).T

    return img.as_reoriented(o)

def align_with_standard(img_raw, img_aff):

    # ================================================================
    # INVERT AXES 
    # ================================================================
    # 
    # This method will flip axes until the following default
    # configurations for xyz-direction cosines are obtained:
    #
    #     AXI = [
    #         [+1,  0,  0],
    #         [ 0, +1,  0],
    #         [ 0,  0, +1]]
    #
    #     COR = [
    #         [+1,  0,  0],
    #         [ 0,  0, +1],
    #         [ 0, -1,  0]]
    #
    #     SAG = [
    #         [ 0,  0, -1],
    #         [+1,  0,  0],
    #         [ 0, -1,  0]] 
    # 
    # ================================================================

    aff = img_aff.affine
    orient = orient_xyz(aff)
    o = np.array([1, 1, 1])

    if orient == 'AXI':

        if aff[0, 0] < 0 : o[0] = -1
        if aff[1, 1] < 0 : o[1] = -1
        if aff[2, 2] < 0 : o[2] = -1

    if orient == 'COR': 

        if aff[0, 0] < 0 : o[0] = -1
        if aff[2, 1] > 0 : o[1] = -1
        if aff[1, 2] < 0 : o[2] = -1

    if orient == 'SAG': 

        if aff[1, 0] < 0 : o[0] = -1
        if aff[2, 1] > 0 : o[1] = -1
        if aff[0, 2] > 0 : o[2] = -1

    o = np.stack((np.array([0, 1, 2]), o)).T

    img_raw = img_raw.as_reoriented(o)
    img_aff = img_aff.as_reoriented(o)

    return img_raw, img_aff

def orient_zyx(affine):

    return ['AXI', 'COR', 'SAG'][np.argmax(np.abs(affine[:3, 0]))]

def orient_xyz(affine):

    return ['SAG', 'COR', 'AXI'][np.argmax(np.abs(affine[:3, 2]))]

def invert_affine(aff):

    aff_ = np.eye(4)
    aff_[:3, :3] = aff[2::-1, 2::-1]
    aff_[:3, 3:] = aff[2::-1, 3:]

    return aff_

if __name__ == '__main__':

    # ===================================================================
    # from dl_tools.utils import show
    # from dl_utils.db import DB
    # db = DB('/home/peter/python/dl_utils/data/niigzs/ymls/db.yml')
    #
    # for sid, fnames, header in db.cursor():
    #     from_nifti(fnames['dat'])
    # ===================================================================

    pass
