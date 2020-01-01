import numpy as np

def montage(x, index=0, z_axis=None):
    """
    Takes an input x and tiles the image.

    :params

      (int) index : index of channel to display; set to None if RGB image
      (str) z_axis : 'first' if (Z x H x W x C) or 'last' if (H x W x C x Z)

    """
    if len(x.shape) == 4 or (len(x.shape) == 3 and x.shape[2] != 3):

        # Attempt to estimate z-axis if none is provided
        if z_axis is None:
            z_axis = 'last' if x.shape[0] == x.shape[1] else 'first'

        # Case (H x W x Z) or (Z x H x W)
        if len(x.shape) == 3:
            if z_axis == 'last': X = np.expand_dims(x, axis=2)
            if z_axis == 'first': X = np.expand_dims(x, axis=3)

        if len(x.shape) == 4:
            X = x

        # All X are now either (H x W x C x Z) or (Z x H x W x C)
        if z_axis == 'last': m, n, c, count = X.shape
        if z_axis == 'first': count, m, n, c = X.shape

        mm = int(np.ceil(np.sqrt(count)))
        nn = mm
        M = np.zeros((mm * m, nn * n, c))

        image_id = 0
        for j in range(mm):
            for k in range(nn):
                if image_id >= count: break
                sliceM, sliceN = j * m, k * n
                if z_axis == 'last': M[sliceM:sliceM + m, sliceN:sliceN + n, :] = X[:, :, :, image_id]
                if z_axis == 'first': M[sliceM:sliceM + m, sliceN:sliceN + n, :] = X[image_id, :, :, :]
                image_id += 1

        if c != 3:
            M = np.squeeze(M[:,:,index])

    else:
        M = x

    return M
