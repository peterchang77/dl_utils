import os, glob
from ..general import printp, printd

def find_matching_files(query, recursive=True, verbose=True):
    """
    Method to find files matching query

    """
    root = query.pop('root')
    keys = sorted(query.keys())

    # --- Record matches and missing 
    matches = {}
    missing = {}

    # --- Search using first key
    q = query.pop(keys[0])
    ms = glob.glob('%s/**/%s' % (root, q), recursive=recursive)

    # --- Prepare remaining query dict
    query = {k: os.path.basename(v) for k, v in query.items()}

    for n, m in enumerate(ms):

        printp('Finding matches...', (n + 1) / len(ms), verbose=verbose)

        d = {keys[0]: m}
        b = os.path.dirname(m)
        e = True

        # --- Find other matches
        for key in keys[1:]:
            ms_ = glob.glob('%s/%s' % (b, query[key]))

            if len(ms_) == 1: 
                d[key] = ms_[0]

            elif len(ms_) == 0:
                d[key] = 'ERROR no match found'
                e = False

            else: 
                d[key] = 'ERROR more than a single match found'
                e = False

        sid = os.path.basename(b)
        d = {k: v.replace(root, '') for k, v in d.items()} 
        if e:
            matches[sid] = d 
        else:
            missing[sid] = d 

    printd('File search complete: %i matches | %i missing' % (len(matches), len(missing)), verbose=verbose)

    return matches, missing
