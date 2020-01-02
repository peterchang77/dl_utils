import numpy as np
from scipy import ndimage, interpolate, optimize
from . import blobs, norms
from ..hasher import sha1

# ======================================================================================
# COORDINATE SYSTEMS (CREATION + MANAGEMENT)
# ======================================================================================

def create_coords(bounds, shape=[0, 0, 0]):
    """
    Method to create sampling coordinates

    Options for shape argument: [z, y, x] where

      * value >= 1  : resample to this shape
      * value == 0 : do not resample (e.g. preserve cropped shape)

    """
    assert all([s >= 0 for s in shape])

    # --- Prepare bounds
    if len(bounds) != 6:
        bounds = np.array([0, 0, 0, bounds[0] - 1, bounds[1] - 1, bounds[2] - 1]) 

    # --- Round bounds if needed
    bounds = np.array(bounds).reshape(2, 3).T

    # --- Calculate shape
    shape = [np.round(np.diff(b)) + 1 if s == 0 else s for b, s in zip(bounds, shape)]

    # --- Create coords
    indices = [create_indices(b, s) for b, s in zip(bounds, shape)]

    return np.stack(np.meshgrid(*indices, indexing='ij'))

def create_indices(b, s):
    """
    Method to create a total s evenly spaced indices between b[0] and b[1]

    """
    b_ = np.diff(b) + 1

    if b_ % s == 0:
        step = int(b_ / s)
        return np.arange(b[0], b[1] + 1, step)

    if s % b_ == 0:
        step = b_ / s 
        return np.arange(b[0], b[1] + 1, step)

    # --- Use linspace for all other non evenly spaced bounds
    return np.linspace(b[0], b[1], s, endpoint=True)

# ======================================================================================
# RESAMPLE WITH COORDS
# ======================================================================================

def resample(arr, coords, order=1, sigma=0.08, cval=None, force_map=False):

    assert type(arr) is np.ndarray

    if arr.dtype in ['uint8', 'int8'] and order > 0:
        return resample_gaussian(arr, coords, sigma=sigma, force_map=force_map)

    else:
        return resample_interp(arr, coords, order=order, cval=cval, force_map=force_map)

def resample_interp(arr, coords, order=1, cval=None, force_map=False):
    """
    Method to perform resample with standard interpolation

    """
    # --- Determine sigma values
    sigma = prepare_gaussian_sigma(1, coords, arr.shape)

    if np.array(sigma).any() or force_map:

        # =====================================================================
        # OPTION 1: RESAMPLE WITH MAP_COORDS 
        # =====================================================================

        c = arr.shape[3]
        arrs = []

        for c_ in range(c):

            m = np.min(arr[..., c_]) if cval is None else cval
            arrs.append(ndimage.map_coordinates(
                arr[..., c_], coords, order=order, mode='constant', cval=m))

        return np.stack(arrs, axis=-1).astype(arr.dtype)

    else:

        axis = tuple(np.arange(1, coords.ndim).astype('int'))
        z0, y0, x0 = np.round(np.min(coords, axis=axis)).astype('int')
        z1, y1, x1 = np.round(np.max(coords, axis=axis)).astype('int')

        coords_ = coords[:, 
            ((z1 - z0) > 0).astype('int'), 
            ((y1 - y0) > 0).astype('int'), 
            ((x1 - x0) > 0).astype('int')]

        coords_ = np.round(coords_ - coords[:, 0, 0, 0]).astype('int')
        sz, sy, sx = coords_.clip(min=1)

        dz = int(sz / np.abs(sz))
        dy = int(sy / np.abs(sy))
        dx = int(sx / np.abs(sx))

        # --- Calculate padding (if needed)
        z_, y_, x_ = arr.shape[:3]
        pz = np.array([max(0, -z0), max(0, z1 + 1 - z_)])
        py = np.array([max(0, -y0), max(0, y1 + 1 - y_)])
        px = np.array([max(0, -x0), max(0, x1 + 1 - x_)])

        # =====================================================================
        # OPTION 2: RESAMPLE WITH SLICES 
        # =====================================================================
        if ~pz.any() and ~py.any() and ~px.any():

            return arr[z0:z1+dz:sz, y0:y1+dy:sy, x0:x1+dx:sx]

        # =====================================================================
        # OPTION 3: RESAMPLE WITH RANGES +/- PADDING
        # =====================================================================
        z0, z1 = z0 + pz[0], z1 - pz[1]
        y0, y1 = y0 + py[0], y1 - py[1]
        x0, x1 = x0 + px[0], x1 - px[1]
        pad_width = [tuple(np.ceil(p / s).astype('int')) for 
            p, s in zip([pz, py, px], [sz, sy, sx])] + [(0, 0)] * (arr.ndim - 3)
        zs = np.arange(z0, z1 + dz, sz)
        ys = np.arange(y0, y1 + dy, sy)
        xs = np.arange(x0, x1 + dx, sx)

        arr_ = arr[zs]
        arr_ = arr_[:, ys]
        arr_ = arr_[:, :, xs]
        arr_ = np.pad(arr_, pad_width, mode='minimum')

        return arr_.astype(arr.dtype)

def resample_gaussian(arr, coords, sigma=0.08, min_value=0.4, force_map=False):
    """
    Method to perform resample non-float data after Gaussian blur

    :params

      (float) sigma     : % of blob to blur
      (float) min_value : min value after blur; effect "dilation" factor to prevent holes

    """
    # --- Determine sigma values
    sigma = prepare_gaussian_sigma(sigma, coords, arr.shape)

    # =====================================================================
    # OPTION 1: RESAMPLE WITH GAUSSIAN
    # =====================================================================

    if np.array(sigma).any():

        shape = list(coords[0].shape) + [1]
        final = [] 
        sigma = sigma + [0]

        # --- Find unique values 
        uniques = np.unique(arr)

        for u in uniques:

            # arr_ = apply_gaussian_to_mask(arr == u, sigma=sigma)

            percentile = np.count_nonzero(arr == u) / np.prod(arr.shape)
            arr_ = ndimage.filters.gaussian_filter((arr == u).astype('float32'), sigma=sigma)
            arr_ = arr_ * 0.5 / np.percentile(arr_, (1 - percentile) * 100)
            arr_ = resample(arr_, coords, cval=-1)
            final.append(arr_)

        final = np.concatenate(final, axis=-1)

        # --- Convert back to original unique values
        max_values = np.max(final, axis=-1, keepdims=True)
        final = final == max_values
        final = final.astype('uint8') * uniques.reshape(1, 1, 1, -1)
        final = np.sum(final, axis=-1, keepdims=True)
        final[arr_ == -1] = 0

    # =====================================================================
    # OPTION 2: RESAMPLE WITHOUT GAUSSIAN
    # =====================================================================

    else:

        final = resample_interp(arr, coords, order=0, force_map=force_map)

    return final.astype(arr.dtype)

def prepare_gaussian_sigma(sigma, coords, shape, decimals=5):
    """
    Method to prepare gaussian sigma

    """
    if type(sigma) is not list:
        sigma = [sigma] * coords.shape[0] 

    # --- Adjust for integer strides >= 1
    strides = [
        coords[0][:2, 0, 0],
        coords[1][0, :2, 0],
        coords[2][0, 0, :2]]

    strides = [np.around(s, decimals) for s in strides]

    for n, s in enumerate(strides):
        if float(np.diff(s) if len(s) > 1 else 0).is_integer() and float(s[0]).is_integer():
            sigma[n] = 0

    return [g * s for g, s in zip(sigma, shape)]

def apply_gaussian_to_mask(msk, sigma=0.08, min_size=0.05):
    """
    Method for efficient gaussian filter of masks

    Assumptions:

      (1) msk is binarized 
      (2) msk objects less than min_size % of largest blob are ignored

    """
    msk_ = np.zeros(msk.shape, dtype='float32')

    # --- Look for labels
    labels, n = ndimage.label(msk)

    # --- Determine min acceptable blob size
    counts = np.bincount(labels.ravel())
    min_count = max(counts[1:]) * min_size

    for i in range(1, n + 1):
        if counts[i] > min_count:

            m = labels == i 
            bounds = blobs.find_bounds(m)

            # --- Ajust for sigma
            s = np.diff(bounds, axis=0).ravel()
            s = np.array([max(s_ * sigma, 1) for s_ in s])
            padding = np.stack((-s * 3, s * 3), axis=0).T
            bounds = bounds.T.astype('float64') + padding
            bounds = bounds.clip(min=0)
            bounds = np.round(bounds).astype('int')

            # --- Crop
            cropped = m[
                bounds[0, 0]:bounds[0, 1],
                bounds[1, 0]:bounds[1, 1],
                bounds[2, 0]:bounds[2, 1], 0]

            # --- Apply gaussian
            cropped = ndimage.filters.gaussian_filter(cropped.astype('float32'), sigma=s)

            # --- Recreate
            s = cropped.shape
            msk_[
                bounds[0, 0]:bounds[0, 0] + s[0],
                bounds[1, 0]:bounds[1, 0] + s[1],
                bounds[2, 0]:bounds[2, 0] + s[2], 0] = cropped

    return msk_

# ======================================================================================
# TRANSFORM WITH COORDS
# ======================================================================================

def apply_affine_to_coords(affine, coords, origin=None, concat=True):
    """
    Method to apply affine transform to given coords

      (1) Prepare coordinates into np.ndarray
      (2) Flatten and concatenate ones
      (3) Perform transformation
      (4) Reshape if needed

    """
    assert type(coords) is list or type(coords) is np.ndarray

    shape = None

    # --- Prepare list
    if type(coords) is list:
        shape = [len(coords)] + list(coords[0].shape)
        coords_ = np.concatenate([c.reshape(1, -1) for c in coords], axis=0)

    # --- Prepare np.ndarray
    elif type(coords) is np.ndarray:
        coords_ = coords.copy()
        if coords.ndim != 2:
            coords_ = coords_.reshape(coords_.shape[0], -1)
            shape = coords.shape
        else:
            shape = None

    # --- Subtract origin
    if origin is not None:
        origin = np.array(origin).reshape(-1, 1)
        assert origin.shape[0] == coords.shape[0]
        coords_ -= origin

    # --- Concatenate ones if needed
    if concat:
        coords_ = np.concatenate([coords_, np.ones(coords_.shape[1]).reshape(1, -1)])
    assert coords_.shape[0] in [3, 4] 

    # --- Perform transformation
    coords_ = np.matmul(affine[:-1], coords_)

    # --- Add origin
    if origin is not None:
        coords_ += origin

    # --- Reshape if needed
    if shape is not None:
        coords_ = coords_.reshape(*shape)

    return coords_

def convert_coords(coords, affine_src, affine_dst):
    """
    Method to align coords from source affine space to destination affine space 

    :params

      (np.ndarray) : coords, 3 x N or 4 x N array, or
      (list)       : coords, 3-element or 4-element list (which will be flattened)

    """
    # --- Create affine
    affine = np.matmul(np.linalg.inv(affine_dst), affine_src)

    # --- Apply affine
    return apply_affine_to_coords(affine, coords, origin=None)

def rotate_coords(coords, unit_vector, theta, origin, ijk_to_zyx):
    """
    Method to rotate coordinates as defined by 4-element Euler 3D rotation parameterization 

    :params

      (np.ndarray or list) coords
      (np.ndarray or list) ijk_to_zyx : affine matrix mapping ijk coordinates to real zyx coordinates 

    """
    # --- Calculate rotational affine matrix
    affine = norms.convert_rotation_to_affine(unit_vector, theta, origin)

    # --- Calculate forward + reverse ijk-zyx tforms
    zyx_fwd = ijk_to_zyx
    zyx_rev = np.linalg.inv(zyx_fwd)

    affine = np.matmul(affine, zyx_fwd)
    affine = np.matmul(zyx_rev, affine)

    # --- Apply affine
    return apply_affine_to_coords(affine, coords)

def deform_coords(coords, affine, ijk_to_zyx, origin=None):
    """
    Method to apply deformation affine to coordinates 

    Note this is a wrapper method to move coordinates to / back from zyx-system for affine transform

    :params

      (np.ndarray or list) coords
      (np.ndarray or list) ijk_to_zyx : affine matrix mapping ijk coordinates to real zyx coordinates 

    """
    # --- Calculate forward + reverse ijk-zyx tforms
    zyx_fwd = ijk_to_zyx
    zyx_rev = np.linalg.inv(zyx_fwd)

    ori_fwd = np.eye(4)
    ori_rev = np.eye(4)
    if origin is not None:
        ori_fwd[:3, 3] = -origin
        ori_rev[:3, 3] = +origin

    affine = np.matmul(ori_rev, np.matmul(affine, ori_fwd))
    affine = np.matmul(affine, zyx_fwd)
    affine = np.matmul(zyx_rev, affine)

    # --- Apply affine
    return apply_affine_to_coords(affine, coords)

def reorient_coords(coords, affine, dst='COR'):
    """
    Method reorient coordinates to the provided orientation

    :params

      (np.ndarray) coords : 3 x Z x Y x X (x 1) coordinates
      (np.ndarray) affine : native affine matrix

    """
    assert coords.ndim >= 4
    assert coords.shape[0] == 3

    dst = dst.upper()
    assert dst in ['AXI', 'COR', 'SAG']

    # ================================================================
    # INVERT AXES 
    # ================================================================
    # 
    # This method will invert (flip) axes until the following default
    # configurations for direction cosines are obtained:
    #
    #     AXI = [
    #         [+1,  0,  0],
    #         [ 0, +1,  0],
    #         [ 0,  0, +1]]
    #
    #     COR = [
    #         [ 0, -1,  0],
    #         [+1,  0,  0],
    #         [ 0,  0, +1]]
    #
    #     SAG = [
    #         [ 0, -1,  0],
    #         [ 0,  0, +1],
    #         [-1,  0,  0]] 
    # 
    # ================================================================

    SRC = ['AXI', 'COR', 'SAG'][np.argmax(np.abs(affine[:3, 0]))]

    DEFAULTS = {
        'AXI': np.array([+1, +1, +1]),
        'COR': np.array([+1, -1, +1]),
        'SAG': np.array([-1, -1, +1])}

    n = create_normalized_direction_cosines(affine)
    f = DEFAULTS[SRC] / np.sum(n, axis=0) 

    if (f == -1).any():
        coords = coords[:, ::f[0], ::f[1], ::f[2]]

    # ================================================================
    # DEFINE MAPPINGS
    # ================================================================

    LAMBDAS = {}

    LAMBDAS['AXI'] = {
        'AXI': lambda x : x,
        'COR': lambda x : np.rot90(x, -1, axes=(1, 2)),
        'SAG': lambda x : np.rot90(np.rot90(x, +1, axes=(1, 3)), +1, axes=(2, 3))}

    LAMBDAS['COR'] = {
        'COR': lambda x : x,
        'AXI': lambda x : np.rot90(x, +1, axes=(1, 2)),
        'SAG': lambda x : np.rot90(x, +1, axes=(1, 3))} 

    LAMBDAS['SAG'] = {
        'SAG': lambda x : x,
        'COR': lambda x : np.rot90(x, -1, axes=(1, 3)), 
        'AXI': lambda x : np.rot90(np.rot90(x, -1, axes=(2, 3)), -1, axes=(1, 3))}

    return LAMBDAS[SRC][dst](coords)

def create_normalized_direction_cosines(affine):
    """
    Method to create normalized direction cosines from affine matrix

    """
    d = np.abs(affine[:3, :3])
    n = np.floor(np.around(d / np.max(d, axis=0), 5))
    m = affine[:3, :3] * n
    m[m > 0] = +1
    m[m < 0] = -1

    return m.astype('int')

# ======================================================================================
# GENERATING AFFINE MATRICES 
# ======================================================================================

def create_affine_09_dof(a, b):
    """
    Method to create affine matrix from points WITHOUT skew

    The returned affine matrix will map coords in B to coords in A:

      np.matmul(affine, b) = a

    The following algorithm is used:

      (1) Displace until zero-centered with b[1]
      (2) Rotate / scale orthogonalized vectors until two planes are aligned
      (3) Displace until zero-centered with a[1]

    IMPORTANT: remember to convert all points to proper coordinate system (zyx or ijk) 
      prior to calling this method

    """
    # --- (1) Find displacement affine
    affine_dis_b = np.eye(4)
    affine_dis_b[:3, 3] = b[1] * -1

    # --- (2a) Orthogonalize and rotation / scale
    a_ = norms.orthogonalize_vectors(a - a[1], normalize=False)
    b_ = norms.orthogonalize_vectors(b - b[1], normalize=False)

    affine_rot = np.eye(4)
    affine_rot[:3, :3] = np.matmul(a_[1:].T, np.linalg.inv(b_[1:].T))

    # --- (3) Find displacement affine
    affine_dis_a = np.eye(4)
    affine_dis_a[:3, 3] = a[1]

    return np.matmul(affine_dis_a, np.matmul(affine_rot, affine_dis_b))

def create_affine_12_dof(a, b):
    """
    Method to create affine matrix from points (12 degrees of freedom)

    The returned affine matrix will map coords in B to coords in A:

      np.matmul(affine, b) = a

    The following algorithm is used:

      (1) Align the center of masses of a and b
      (2) Perform transform
      (3) Reverse center of mass translations

    IMPORTANT: remember to convert all points to proper coordinate system (zyx or ijk) 
      prior to calling this method

    """
    # --- (0) Reorient to proper 2 x N or 3 x N 
    a = np.squeeze(np.array(a))
    b = np.squeeze(np.array(b))

    assert a.ndim == 2
    assert b.ndim == 2

    aa = a.T if a.shape[1] in [2, 3] else a
    bb = b.T if b.shape[1] in [2, 3] else a

    N = aa.shape[0]
    C = aa.shape[1]

    if C <= 3:
        printd('Warning only %s points provided for affine tform calculation' % C)

    # --- (1) Center points
    am = aa.mean(axis=1).reshape(N, 1)
    bm = bb.mean(axis=1).reshape(N, 1)

    fun = lambda t : np.sum((np.matmul(t.reshape(N, N), (bb - bm)) - (aa - am)) ** 2)
    sol = optimize.minimize(fun, np.eye(N))

    tform = np.eye(N + 1)
    tform[:N, :N] = sol['x'].reshape(N, N)

    tform_rev = np.eye(N + 1)
    tform_fwd = np.eye(N + 1)
    tform_rev[:N, N:] = +am
    tform_fwd[:N, N:] = -bm

    return np.matmul(tform_rev, np.matmul(tform, tform_fwd))

def create_warp_disp(fixed, moving, coords, function='thin-plate'):
    """
    Method to calculate N-dimensional displacement field based on coords 

    :params

      (np.ndarray) fixed  : landmarks for fixed (in ijk- or zyx-coordinates; must match coords)
      (np.ndarray) moving : landmarks for moving (in ijk- or zyx-coordinates; must match coords)
      (np.npdarry) coords : resampling coordinates (must match units for points)
      (str) function : 'thin-plate', 'linear', 'multiquadratic'

    """
    # --- Prepare
    fixed = np.squeeze(fixed)
    moving = np.squeeze(moving)

    assert fixed.shape == moving.shape
    assert fixed.ndim == 2
    assert moving.ndim == 2

    N = fixed.shape[1]

    # ======================================================================================
    # ADD EXTREME POINTS OF COORDINATES FOR ANCHORS
    # ======================================================================================

    if N == 2:

        # --- Determine extreme values
        mm = np.array([
            [coords[0,  0, 0], coords[1, 0,  0]],
            [coords[0, -1, 0], coords[1, 0, -1]]])

        # --- Determine corners
        corners = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]])

    if N == 3:

        # --- Determine extreme values
        mm = np.array([
            [coords[0,  0, 0, 0], coords[1, 0,  0, 0], coords[2, 0, 0,  0]],
            [coords[0, -1, 0, 0], coords[1, 0, -1, 0], coords[2, 0, 0, -1]]])

        # --- Determine coordinate locations
        corners = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]]) 

    # --- Calculate and append corners
    columns = np.tile(np.arange(N).reshape(1, -1), [corners.shape[0], 1])
    corners = mm[corners, columns]

    fixed = np.concatenate([fixed, corners], axis=0)
    moving = np.concatenate([moving, corners], axis=0)

    # ======================================================================================
    # CALCULATE FORMULA TO MODEL DISPLACEMENTS 
    # ======================================================================================
    
    # --- Center the displacements
    offsets = np.mean(fixed - moving, axis=0)
    fixed -= offsets.reshape(1, -1)

    # --- Calculate displacements
    d = fixed - moving 

    # --- Calculate radial basis functions 
    axes = ['z', 'y', 'x'][-N:]
    rbfi = {}

    for n, a in enumerate(axes):
        moving_ = np.concatenate((moving, d[:, n:n+1]), axis=1)
        moving_ = np.unique(moving_, axis=0)
        inputs = [i.ravel() for i in np.split(moving_, N + 1, axis=1)]
        rbfi[a] = interpolate.Rbf(*inputs, function=function)

    # --- Interpolate displacements
    disp = {a: rbfi[a](*coords) for a in axes}
    for n, a in enumerate(axes):
        disp[a] += offsets[n]

    # --- Stack 
    disp = np.stack([disp[a] for a in axes])

    return disp

# ======================================================================================
# XFORMS 
# ======================================================================================

def init_xforms(xforms):
    """
    Method to initialize transforms daemon xforms args dict

    """
    DEFAULTS = {
        'source': None,
        'bounds': None,
        'shapes': [0, 0, 0],
        'affine': {},
        'resamp': {},
        'meanip': None}

    for k, v in DEFAULTS.items():
        if k not in xforms:
            xforms[k] = v

    # --- Initialize bounds 
    if xforms['bounds'] is not None:

        # --- Determine aspect ratio
        aspect_ratio = None
        if xforms['shapes'] is not None:
            aspect_ratio = [1 if s > 0 else None for s in xforms['shapes']]

        DEFAULTS_BOUNDS = {
            'lambda': {'$gte': 2},
            'method': 'default', # usage = dill.dumps(func) 
            'find_largest': None, # usage = {'n': 1, 'min_ratio': None}
            'axes': [True, True, True],
            'aspect_ratio': aspect_ratio,
            'padding': [0, 0, 0]}

        xforms['bounds'] = {k: xforms['bounds'][k] if k in xforms['bounds'] 
            else DEFAULTS_BOUNDS[k] for k in DEFAULTS_BOUNDS}

    # --- Initialize affine 
    if xforms['affine'] is not None:

        DEFAULTS_AFFINE = {
            'unit_vectors': [[1, 0, 0]], # usage = [u1, u2, u3, ...] or {'$rand': ...}
            'thetas': [0], # usage = [t1, t2, t3, ...] or {'$rand': ...}
            'scales': [None], # usage = [s1, s2, s3, ...] or {'$rand': ...}
            'shears': [None], # usage = [s1, s2, s3, ...] or {'$rand': ...}
            'matrices': [None], # usage = [m1, m2, m3] or {'$rand': ...}
            'permute': False #usage = True to use all permutations
        }

        xforms['affine'] = {k: xforms['affine'][k] if k in xforms['affine'] 
            else DEFAULTS_AFFINE[k] for k in DEFAULTS_AFFINE}

    # --- Initialize resamp
    DEFAULTS_RESAMP = {
        'sigma': 0.01,
        'xtags': {}}

    xforms['resamp'] = {k: xforms['resamp'][k] if k in xforms['resamp'] 
        else DEFAULTS_RESAMP[k] for k in DEFAULTS_RESAMP}

    # --- Initialize _hash-xforms
    if '_hash-xforms' not in xforms:
        xforms['_hash-xforms'] = sha1(xforms, truncate=10) 

    return xforms 

if __name__ == '__main__':

    pass
