import os, pickle, multiprocessing as mp
import pydicom

def update_dict(d, keys, value): 
    """
    Method to recursively walk through dict and append value to list

    NOTE: If None is reached for a key, no update is made.

    """
    if keys[0] is None:
        return

    if len(keys) == 1:
        if keys[0] not in d:
            d[keys[0]] = []
        d[keys[0]].append(value)

    else:
        if keys[0] not in d:
            d[keys[0]] = {}

        update_dict(d[keys[0]], keys[1:], value)

def read_headers(fnames, headers, verbose=False, total_count=0):
    """
    Method to read all headers from list of DICOM fnames

    """
    assert type(fnames) is list
    results = {}

    for n, fname in enumerate(fnames):

        try:
            d = pydicom.read_file(fname, stop_before_pixels=True)
            results[fname] = {h: getattr(d, h, None) for h in headers}
        except:
            results[fname] = None

        if verbose:
            print('Reading DICOM files: %09i' % (n + total_count), end='\r')

    return results

class DicomParser():

    def __init__(self, HEADERS=['Modality', 'StudyInstanceUID', 'SeriesInstanceUID'], PATTERN=['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'],
            SAVE_RATE=10000, MP_PROCESSES=2, MP_CHUNKS=100, DIR_JSONS='./jsons', DIR_LINKS='./links'):
        """
        Method to initialize DicomParser object with default values

        All data is parsed into a nested self.collections JSON object:

          * self.collections['fnames'] ==> all requested HEADERS
          * self.collections['parsed'] ==> all data organized by PATTERN
          
        self.collections['fnames'] = {

            '/path/to/file-000': {'Modality': ..., 'StudyInstanceUID': ...},
            '/path/to/file-001': {'Modality': ..., 'StudyInstanceUID': ...},
            '/path/to/file-002': {'Modality': ..., 'StudyInstanceUID': ...},... }

        self.collections['parsed'] = {

            {'PATTERN-00': {'PATTERN-01': {'PATTERN-02': ...}}},
            {'PATTERN-00': {'PATTERN-01': {'PATTERN-02': ...}}}, ...}

        Similarly, if symlinks are created, the same PATTERN is used:

          (self.DIR_LINKS)/(self.PATTERN[0])/(self.PATTERN[1])/.../(self.PATTERN[-1]).dcm

        :params

          (list) HEADERS : list of DICOM headers to extract from each file
          (list) PATTERN : pattern to parse / organize DICOM files 

          (int) SAVE_RATE : rate to serialize / save self.collections JSON object
          (str) DIR_JSONS : directory to serialize JSON 
          (str) DIR_LINKS : directory to save symlinks (if requested)

          (int) MP_PROCESSES : total number of multiprocessing pool objects
          (int) MP_CHUNKS    : total number of DICOMs processed at a time by pool object

        """
        # --- Set default HEADERS + PATTERN
        self.HEADERS = set(HEADERS + PATTERN)
        self.PATTERN = PATTERN

        self.DIR_JSONS = DIR_JSONS
        self.DIR_LINKS = DIR_LINKS

        # --- Load existing JSON if present
        if os.path.exists('%s/collections.pkl' % DIR_JSONS):
            self.collections = pickle.load(open('%s/collections.pkl' % DIR_JSONS, 'rb'))

        else:
            self.collections = {
                'parsed': {},
                'fnames': {}}

        self.SAVE_RATE = SAVE_RATE

        # --- Init pool if needed
        if MP_PROCESSES > 0:
            self.pool = mp.Pool(processes=MP_PROCESSES)
            self.read_headers = self.read_headers_mp
        else:
            self.pool = None
            self.read_headers = read_headers

        self.MP_PROCESSES = MP_PROCESSES
        self.MP_CHUNKS = MP_CHUNKS

    def upsert_docs_parsed(self, docs): 

        for fname, doc in docs.items():
            keys = [doc[p] for p in self.PATTERN[:-1]]
            update_dict(self.collections['parsed'], keys, fname)

    def upsert_docs_fnames(self, docs):

        self.collections['fnames'].update(docs)

    def remove_existing_fnames(self, fnames):

        if len(self.collections['fnames']) > 0:
            fnames = [f for f in fnames if f not in self.collections['fnames']]

        return fnames

    def serialize_docs(self):

        os.makedirs(self.DIR_JSONS, exist_ok=True)
        pickle.dump(self.collections, open('%s/collections.pkl' % self.DIR_JSONS, 'wb'))

    def create_links(self, docs):
        """
        Method to create symlinks based on self.PATTERN

        """
        DST = self.DIR_LINKS  + '/%s' * len(self.PATTERN) + '.dcm'

        for src, doc in docs.items():

            ts = tuple([doc[p] for p in self.PATTERN])
            if not any([t is None for t in ts]):
                dst = DST % ts
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.symlink(src=src, dst=dst)

    def process(self, fnames, create_links=False, verbose=True):
        """
        Method to process list of DICOM fnames

        If symlinks are requested, file path follows self.PATTERN:

          (self.DIR_LINKS)/(self.PATTERN[0])/(self.PATTERN[1])/.../(self.PATTERN[-1]).dcm

        :params

          (bool) create_links : if True, create symlinks with provided self.PATTERN
          (bool) verbose : if True, print total # of DICOM objects processed

        """
        assert type(fnames) is list
        fnames = self.remove_existing_fnames(fnames)

        while len(fnames) > 0:

            docs = self.read_headers(fnames[:self.SAVE_RATE], self.HEADERS, verbose=verbose, total_count=len(self.collections['fnames']))

            self.upsert_docs_parsed(docs)
            self.upsert_docs_fnames(docs)
            self.serialize_docs()

            if create_links:
                self.create_links(docs)

            fnames = fnames[self.SAVE_RATE:]

    def read_headers_mp(self, fnames, headers, verbose=False, total_count=0):

        results = {}

        C = self.MP_CHUNKS
        P = self.MP_PROCESSES

        for i in range(round(len(fnames) / (C * P) + 0.5)):

            ss = [i * P * C + p * C for p in range(P)]
            fs = [fnames[s:s+C] for s in ss]
            rs = [self.pool.apply_async(read_headers, args=([f, headers])) for f in fs]
            rs = [r.get() for r in rs]

            for r in rs:
                results.update(r)

            if verbose:
                print('Parsed %09i DICOM files' % (len(results) + total_count), end='\r')

        return results

if __name__ == '__main__':

    pass
