def is_equal_trimmed(a, b, decimals=4):
    """
    Method to compare to np.ndarrays after trimming to provided decimal precision 

    """
    return ((a - b) < 10 ** -decimals).all()
