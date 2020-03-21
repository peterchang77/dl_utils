import numpy as np

def montage(arr, N=None, squeeze=True):
    """
    Method to create a tiled N x N montage of input volume x

    """
    if type(arr) is not np.ndarray:
        return arr

    arr = arr.squeeze().copy()

    # --- Create 4D volume
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=-1)

    assert arr.ndim == 4

    z, y, x, c = arr.shape

    N = N or int(np.ceil(np.sqrt(z)))
    Z = int(np.ceil(z / N ** 2))
    M = np.zeros((Z, N * y, N * x, c), dtype=arr.dtype)

    n = 0
    for z_ in range(Z):
        for y_ in range(N):
            for x_ in range(N):

                yy = y_ * y
                xx = x_ * x
                M[z_, yy:yy + y, xx:xx + x, :] = arr[n]

                n += 1
                if n >= z: 
                    break
            if n >= z: 
                break
        if n >= z: 
            break

    if squeeze:
        M = M[..., 0]

    return M
