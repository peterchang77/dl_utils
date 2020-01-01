import multiprocessing as mp
import pydicom

def read_headers(fnames, headers, verbose=False):

    results = {}

    for n, fname in enumerate(fnames):

        d = pydicom.read_file(fname, stop_before_pixels=True)
        results[fname] = {h: getattr(d, h, None) for h in headers}

        if verbose:
            print('Reading DICOM files: %06i' % n, end='\r')

    return results

def read_headers_mp(fnames, headers, processes=4, chunks=100, verbose=False):

    pool = mp.Pool(processes=4)
    results = {}

    for i in range(round(len(fnames) / (chunks * processes) + 0.5)):

        ss = [i * chunks * processes + p * chunks for p in range(processes)]
        fs = [fnames[s:s+chunks] for s in ss]
        rs = [pool.apply_async(read_headers, args=([f, headers])) for f in fs]
        rs = [r.get() for r in rs]

        for r in rs:
            results.update(r)

        if verbose:
            print('Parsed %09i DICOM files' % len(results))

    pool.close()
        
    return results

# =============================
# import pickle, random, time
# f = pickle.load(open('./pkls/dcm_files.pkl', 'rb'))
# random.shuffle(f)
# f = f[:225]
# HEADERS = ['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']
# =============================

# =============================
# start_time = time.time()
# results = read_headers(f, HEADERS, verbose=True)
# print('Elapsed time: %0.2f sec' % (time.time() - start_time))
# =============================

# =============================
# start_time = time.time()
# results = read_headers_mp(f, HEADERS, processes=2, chunks=50, verbose=True)
# print('Elapsed time: %0.2f sec' % (time.time() - start_time))
# =============================
