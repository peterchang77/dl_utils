import numpy as np
from scipy import ndimage

# =================================================================================
# BLOBS LIBRARY
# =================================================================================

def find_bounds(msk, axes=[True, True, True], padding=[0, 0, 0], aspect_ratio=[None, None, None], dims=[1, 1, 1]):
    """
    Method to find bounds of provided mask 

    As needed, recommend using find_largest(msk, n) to preprocess mask prior to this method.

    :return

      (np.array) bounds = [

          [z0, y0, x0],
          [z1, y1, x1]]

    """
    assert msk.ndim == 4

    bounds = []

    # --- Prepare bounds
    for axis, crop in enumerate(axes):
        if crop:
            a = [0, 1, 2, 3]
            a.pop(axis)
            a = np.nonzero(np.sum(msk, axis=tuple(a)))[0]
            bounds.append([a[0], a[-1]])
        else:
            bounds.append([0, msk.shape[axis] - 1])

    # --- Prepare padding
    padding = calculate_padding(padding, np.diff(bounds, axis=1)) 
    bounds = np.array(bounds) + padding

    # --- Prepare aspect ratio
    if any([a is not None for a in aspect_ratio]):
        bounds = balance_aspect_ratio(bounds, aspect_ratio, dims)

    return bounds.T

def calculate_padding(padding, shape):
    """
    Method to calculate padding:
    
      (1) padding < 1: padding as a function (%) of the shape of blob
      (2) padding > 1: padding as absolute value 

    """
    padding = [int(np.ceil(p * s)) if p < 1 else p for p, s in zip(padding, shape[:3])] 
    padding = np.array([(p * -1, p) for p in padding])

    return padding

def balance_aspect_ratio(bounds, aspect_ratio, dims):
    """
    Method to balance aspect ratio by expanding bounds as needed along each axes

    """
    # --- Find midpoints 
    midpts = np.mean(bounds, axis=1).ravel() 

    # --- Find shapes 
    shapes = np.diff(bounds.astype('float32'), axis=1).ravel()
    shapes_zyx = shapes * np.array(dims[:3])
    shapes_norm = shapes_zyx / min(shapes_zyx)

    none_div = lambda x, y : x / y if x is not None else None
    none_min = lambda it : min([i for i in it if i is not None])

    # --- Find ratios
    min_rs = none_min(aspect_ratio) 
    ratios = [none_div(r, min_rs * s) for r, s in zip(aspect_ratio, shapes_norm)]

    # --- Normalize 
    min_rs = none_min(ratios) 
    ratios = [r / min_rs if r is not None else 1 for r in ratios]

    shapes *= np.array(ratios)

    return np.array([midpts - shapes / 2, midpts + shapes / 2]).T

def find_largest(msk, n=1, min_ratio=None, return_labels=False):
    """
    Method to return largest n blob(s) 
    
    :params

      (int)   n            : n-largest blobs to return
      (float) min_ratio    : return blobs > min_ratio of largest blob
      (bool) return_labels : if True, return labeled matrix (instead of binary)

    """
    if not msk.any():
        return

    labels, _ = ndimage.label(msk > 0)

    counts = np.bincount(labels.ravel())[1:]
    argsrt = np.argsort(counts)[::-1] + 1

    if n == 1 and min_ratio is None:

        if return_labels:
            return labels == argsrt[0], 1

        else:
            return labels == argsrt[0]

    elif min_ratio == 0:

        if return_labels:
            return labels, counts.size

        else:
            return labels > 0


    else:

        # --- Set n based on min_ratio
        if min_ratio is not None:
            counts = counts[argsrt - 1]
            n = np.count_nonzero(counts > (counts[0] * min_ratio))

        # --- Find largest
        if return_labels:

            msk_ = np.zeros(labels.shape, dtype='int16')
            for i, a in enumerate(argsrt[:n]):
                msk_ += (labels == a) * (i + 1)

            return msk_, i + 1

        else:

            msk_ = np.zeros(labels.shape, dtype='bool')
            for a in argsrt[:n]:
                msk_ = msk_ | (labels == a)

            return msk_

def label_largest(msk, n=1, min_ratio=None):
    """
    Method to find larget n blob(s) and return labeled array

    :return

      (np.array) labels where,
        
        labels == 1 : largest blob
        labels == 2 : second largest blob
        labels == 3 : third largest blob, ...

    """
    if not msk.any():
        return

    labels, _ = ndimage.label(msk > 0)

    counts = np.bincount(labels.ravel())[1:]
    argsrt = np.argsort(counts)[::-1] + 1

    if n == 1 and min_ratio is None:
        return (labels == argsrt[0]).astype('int32')

    else:
        if min_ratio is not None:
            counts = counts[argsrt]
            n = np.count_nonzero(counts >= (counts[0] * min_ratio))

        msk_ = np.ones(labels.shape, dtype='int32')
        for i, a in enumerate(argsrt[:n]):
            msk_[labels == a] = i + 1

        return msk_, n

def perim(msk, radius=1):
    """
    Method to create msk perimeter

    """
    return ndimage.binary_dilation(msk, iterations=radius) ^ (msk > 0)

def find_center_of_mass(msk, acceleration=1):

    msk_ = msk[..., 0] if msk.ndim == 4 else msk

    # --- Subsample
    if type(acceleration) is int:
        acceleration = [acceleration] * msk_.ndim 

    msk_ = msk_[::acceleration[0], ::acceleration[1], ::acceleration[2]]
    acceleration = np.array(acceleration).reshape(1, -1)

    center = ndimage.center_of_mass(msk_)

    return np.array(center) * acceleration

def imfill(msk):
    """
    Method to 2D fill holes in blob

    """
    assert msk.ndim == 4

    filled = np.zeros(msk.shape[:3], dtype='bool')
    
    # --- Create edge mask
    edge_mask = np.zeros(shape=msk.shape[1:3], dtype='bool')
    edge_mask[0, :] = True
    edge_mask[:, 0] = True
    edge_mask[-1:, :] = True
    edge_mask[:, -1:] = True

    # --- Loop
    for z, m in enumerate(msk[..., 0]):
        if m.any():

            labels, _ = ndimage.label(m == 0)
            edges = np.unique(labels[edge_mask & (m == 0)])
            for edge in edges:
                filled[z] = filled[z] | (labels == edge)

        else:
            filled[z] = True

    return ~np.expand_dims(filled, axis=-1)

if __name__ == '__main__':

    pass
