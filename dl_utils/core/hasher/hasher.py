import hashlib, base64, json, io
import numpy as np

convert = {
    str: lambda x : bytes(x, 'utf-8'),
    int: lambda x : bytes(x),
    float: lambda x : bytes(str(x), 'utf-8'),
    dict: lambda x : bytes(json.dumps(x, sort_keys=True), 'utf-8'),
    list: lambda x : bytes(json.dumps(x, sort_keys=True), 'utf-8'),
    np.ndarray: lambda x : x.tobytes(),
    io.BufferedReader: lambda x : x.read()
}

def sha1(obj, truncate=None):
    """
    Method to implement deterministic SHA hash

    """
    assert type(obj) in convert

    m = hashlib.sha1()
    m.update(convert[type(obj)](obj))

    if truncate is None:
        return m.hexdigest()

    else:
        assert type(truncate) is int, 'Error, truncate value must be int'
        return base64.urlsafe_b64encode(m.digest()).decode('ascii')[:truncate]
