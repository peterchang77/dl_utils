import numpy as np

# =================================================================================
# NORMALIZED COORDINATES LIBRARY
# =================================================================================

def convert_norm_to_full(data, shape, ints=True, clip=True):
    """
    Method to convert normalized points to full coordinates

    """
    # --- Reshape data
    orig = data.shape
    data = data.reshape(-1, 3)

    # --- Clip 
    if clip:
        data = data.clip(min=0, max=1)

    # --- Reverse normalization
    shape = prepare_shape(shape)
    data = data * shape

    if ints:
        data = np.round(data).astype('int')

    # --- Reshape data
    data = data.reshape(orig)

    return data

def convert_full_to_norm(data, shape):
    """
    Method to convert full coordinates to normalized points

    """
    # --- Reshape data
    orig = data.shape
    data = data.reshape(-1, 3)

    # --- Normalize 
    shape = prepare_shape(shape)
    data = data / shape[:3]

    # --- Reshape data
    data = data.reshape(orig)

    return data

def prepare_shape(shape):
    """
    Method to prepare shape:
    
      (1) Extract first three elements
      (2) Reshape to 1 x 3
      (3) subtract 1 from shape

    """
    return (np.array(shape[:3]).reshape(1, -1) - 1).clip(min=1)

def convert_box_to_msk(data, shape, values=None, fill=False, stroke=1, msk=None):
    """
    Method to normalized pts to msk

    """
    bounds = convert_norm_to_full(data, shape, ints=True, clip=True)
    bounds = bounds.reshape(-1, 6)
    bounds[:, 3:] += 1

    # --- Initialize values
    if values is None:
        values = np.arange(1, bounds.shape[0] + 1).astype('int')

    if type(values) is int:
        values = [values] * bounds.shape[0]

    assert len(values) == bounds.shape[0]

    values = np.array(values).clip(min=1)

    if msk is None:
        msk = np.zeros(shape, dtype='uint8')

    for b, v in zip(bounds, values):

        z1, y1, x1, z2, y2, x2 = b 

        if fill:
            msk[z1:z2, y1:y2, x1:x2] = v 

        else:
            msk[z1:z2, y1:y1 + stroke, x1:x2] = v 
            msk[z1:z2, y2:y2 + stroke, x1:x2] = v 
            msk[z1:z2, y1:y2, x1:x1 + stroke] = v 
            msk[z1:z2, y1:y2, x2:x2 + stroke] = v 

    return msk

def convert_pts_to_msk(data, shape, values=None, r=[1, 3, 3]):
    """
    Method to normalized pts to msk

    """
    msk = np.zeros(shape, dtype='uint8')

    # --- Eliminate points outside of FOV
    data = data.reshape(-1, 3) 
    data = data[np.all((data >= 0) & (data <= 1), axis=1)]

    if data.size == 0:
        return msk

    # --- Convert points
    points = convert_norm_to_full(data, shape, ints=True, clip=True)

    # --- Initialize values
    if values is None:
        values = np.arange(1, points.shape[0] + 1).astype('int')

    if type(values) is int:
        values = [values] * points.shape[0]

    values = values.clip(min=1)

    # --- Initialize radii
    if type(r) is not list:
        r = [r] * 3

    for p, v in zip(points, values):

        msk[
            p[0]-r[0]:p[0]+r[0]+1,
            p[1]-r[1]:p[1]+r[1]+1,
            p[2]-r[2]:p[2]+r[2]+1] = v

    return msk

# =================================================================================
# NORMALIZED ROTATION VECTOR LIBRARY
# =================================================================================

def normalize_vector(vec):

    vec = np.array(vec).ravel()[:3]
    size = np.linalg.norm(vec) 

    assert size != 0, 'Error provided unit vector is size == 0'

    return vec / size 

def orthogonalize_vectors(v, normalize=False):
    """
    Method to orthogonalize vectors

    :params

      (bool) normalize : if False, make length == distance from v[0] to v[1]

    :return

      (np.ndarray) v_, where:
    
        v_[0] = origin (same as v[1])
        v_[1] = axis 0 (same as v[0])
        v_[2] = axis 1 
        v_[3] = axis 2

    """
    v = np.array(v)

    p1 = v[0] - v[1]
    p2 = v[2] - v[1]
    p3 = np.cross(p1, p2)
    p2 = np.cross(p3, p1)

    v_ = np.array([p1, p2, p3])
    v_ = v_ / np.linalg.norm(v_, axis=1).reshape(3, 1)

    if not normalize:
        v_ = v_ * np.linalg.norm(p1)

    v_ += v[1:2]

    return np.concatenate((v[1:2], v_), axis=0)

def init_euler_rotation(unit_vector=None, theta=None, origin=None, p1=None, p2=None):
    """
    Method to initialize 4-element Euler 3D rotation parameterization:

      (1) 3-element unit vector
      (2) 1-element theta

    """
    if unit_vector is None and p1 is None:
        return np.array([1, 0, 0, 0])

    if unit_vector is not None:
        assert theta is not None
        uni = normalize_vector(unit_vector)

    else:
        uni, theta = find_theta_from_points(origin, p1, p2)

    return np.concatenate((uni.ravel(), np.array([theta]).ravel())) 

def convert_rotation_to_affine(unit_vector, theta, origin=None):
    """
    Method to use Euler rotation theorem to reconstruct affine matrix from:

      (1) unit vector (defines plane of rotation)
      (2) theta (defines degrees of rotation about plane)

    Details for rotation matrix formula can be found here:

      https://en.wikipedia.org/wiki/Rotation_matrix

    """
    uz, uy, ux = normalize_vector(unit_vector)
    c = np.cos(float(theta))
    s = np.sin(float(theta))

    a00 = c + uz ** 2 * (1 - c)
    a11 = c + uy ** 2 * (1 - c)
    a22 = c + ux ** 2 * (1 - c)
    a01 = uz * uy * (1 - c) - ux * s
    a10 = uz * uy * (1 - c) + ux * s
    a02 = uz * ux * (1 - c) + uy * s
    a20 = uz * ux * (1 - c) - uy * s
    a12 = ux * uy * (1 - c) - uz * s
    a21 = ux * uy * (1 - c) + uz * s

    affine = np.eye(4)
    affine[:3, :3] = np.array([
        [a00, a01, a02],
        [a10, a11, a12],
        [a20, a21, a22]])

    if origin is not None:

        ori = np.array(origin).ravel()
        fwd = np.eye(4)
        rev = np.eye(4)
        fwd[:3, 3] = -ori
        rev[:3, 3] = +ori

        affine = np.matmul(rev, np.matmul(affine, fwd))

    return affine

def find_theta_from_points(origin, p1, p2):
    """
    Method to find unit_vector and theta from points + origin
    
      * unit_vector : vector penpedicular to rotation
      * theta : angle formed formed by p1 > origin > p2

    """
    if not np.around(p1 - p2, 5).any():
        return np.array([1, 0, 0]), 0

    A = np.array(p1) - np.array(origin)
    B = np.array(p2) - np.array(origin)

    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)

    unit_vector = normalize_vector(np.cross(A, B))
    theta = np.arccos(np.dot(A, B) / (A_norm * B_norm))

    return unit_vector, theta

def find_theta_from_planes(a, b): 
    """
    Method to find unit_vector and theta from two 3D planes

    :params
    
      (np.ndarray) a : [[z0, y0, x0], [z1, y1, x1], [z2, y2, x2]]
      (np.ndarray) b : [[z0, y0, x0], [z1, y1, x1], [z2, y2, x2]]

    """
    a = np.array(a)
    b = np.array(b)

    a_ = a[1:] - a[:-1]
    b_ = b[1:] - b[:-1]

    p1 = np.cross(a_[0], a_[1])
    p2 = np.cross(b_[0], b_[1])

    return find_theta_from_points(origin=[0, 0, 0], p1=p1, p2=p2)
