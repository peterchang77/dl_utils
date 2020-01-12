import os, yaml, numpy as np, pandas as pd, tarfile
from .query import find_matching_files
from ..general import printd, printp

# ===================================================================
# OVERVIEW
# ===================================================================
# 
# DB is an object that facilitates interaction and manipulation
# of data serialized in a way that conforms to the standard `dl_*` 
# file directory hiearchy. Two primary forms of data are indexed:
#
#   (1) fnames ==> index of file locations
#   (2) header ==> index of metadata for each file 
# 
# The underlying indexed data is stored as CSV files. 
# 
# ===================================================================
# 
# [ FILE-SYS ] \
#               <----> [ CSV-FILE ] <----> [ -CLIENT- ] 
# [ MONGO_DB ] /
# 
# ===================================================================

class DB():

    def __init__(self, *args, **kwargs):
        """
        Method to initialize DB() object

        The following initialization sources can be used:

          (1) query dictionary 
          (2) CSV file (with existing data)
          (3) YML file
          (4) shell environment variables

        """
        # --- Parse args
        self.parse_args(*args, **kwargs)

        # --- Load YML and CSV files
        self.load_yml()
        self.load_csv()

        # --- Refresh
        self.refresh()

    def parse_args(self, *args, **kwargs):
        """
        Method to parse arguments into final kwargs dict

        """
        DEFAULTS = {
            'paths': {'data': '', 'code': ''},
            'files': {'csv': None, 'yml': None},
            'query': {},
            'funcs': []}

        # --- Extract values from args 
        files = {}
        paths = {}
        args_ = {}

        for arg in args:
            if type(arg) is str:
                if os.path.isdir(arg):
                    paths['data'] = arg
                    paths['code'] = arg
                else:
                    ext = arg.split('.')[-1]
                    if ext in ['yml']:
                        files['yml'] = arg
                    if ext in ['csv', 'gz']:
                        files['csv'] = arg

            elif type(arg) is dict:
                args_['query'] = arg

            elif type(arg) is list:
                args_['funcs'] = arg

        # --- Extract values from ENV 
        environ = {}
        for key in ['data', 'code']:
            ENV = 'DL_PATHS_{}'.format(key.upper())
            if ENV in os.environ:
                environ[key] = os.environ[ENV]

        # --- Initialize default values
        configs = {**DEFAULTS, **args_, **kwargs} 

        if 'root' in configs['query']:
            paths['data'] = configs['query'].pop('root')

        # --- Initialize default paths and files
        configs['paths'] = {**DEFAULTS['paths'], **paths, **kwargs.get('paths', {}), **environ}
        configs['files'] = {**DEFAULTS['files'], **files, **kwargs.get('files', {})}

        # --- Attributes to serialize in self.to_yml(...) 
        self.ATTRS = ['paths', 'files', 'query', 'funcs']

        # --- Save
        for attr in self.ATTRS:
            setattr(self, attr, configs[attr])

        # --- Set default paths and files locations
        self.set_paths(update_fnames=False)

    def set_paths(self, paths=None, update_fnames=True):
        """
        Method to set self.paths and remove prefix from fnames

        """
        paths = paths or {}
        assert type(paths) in [str, dict]

        if type(paths) is str:
            paths = {'data': paths}

        paths = {**self.paths, **paths} 
        paths = {k: os.path.abspath(p) if p != '' else '' for k, p in paths.items()}
        self.paths = paths

        if 'data' in paths and update_fnames:
            for col in self.fnames:
                self.fnames[col] = self.fnames[col].apply(lambda x : x.replace(paths['data'], ''))

        if 'code' in paths:
            self.set_files()

    def set_files(self, files=None):
        """
        Method to set self.files

        :params

          (dict) files : {'csv': ... or None, 'yml': ... or None}

        """
        files = {**self.files, **(files or {})} 

        if self.paths['code'] != '':
            self.files['csv'] = files['csv'] or '/csvs/db.csv.gz'
            self.files['yml'] = files['yml'] or '/ymls/db.yml'

    def get_files(self):
        """
        Method to get full file paths

        """
        return {k: '{}{}'.format(self.paths['code'], v) if v is not None else None 
            for k, v in self.files.items()}

    # ===================================================================
    # YML | LOAD
    # ===================================================================

    def load_yml(self, fname=None):
        """
        Method to load YML file

        """
        fname = fname or self.get_files()['yml'] or ''

        if os.path.exists(fname):
            with open(fname, 'r') as y:
                configs =  yaml.load(y, Loader=yaml.FullLoader)

            for attr in self.ATTRS:
                setattr(self, attr, configs[attr])

    # ===================================================================
    # CSV | LOAD and PREPARE
    # ===================================================================

    def load_csv(self, fname=None):
        """
        Method to load CSV file

        """
        fname = fname or self.get_files()['csv'] or ''

        if os.path.exists(fname):
            df = pd.read_csv(fname, index_col='sid')
        else: 
            df = pd.DataFrame()
            df.index.name = 'sid'

        # --- Split df into fnames and header 
        self.fnames, self.header = self.df_split(df)

    def df_split(self, df):
        """
        Method to split DataFrame into `fnames` + `header`

        """
        fnames = df[[k for k in df if k[:6] == 'fname-']]
        header = df[[k for k in df if k not in fnames]]

        # --- Rename `fnames-`
        fnames = fnames.rename(columns={k: k[6:] for k in fnames.columns})

        return fnames, header 

    def df_merge(self, rename=False):
        """
        Method to merge DataFrame

        """
        # --- Rename `fnames-`
        if rename:
            c = {k: 'fname-%s' % k for k in self.fnames.columns}
            fnames = self.fnames.rename(columns=c)
        else:
            fnames = self.fnames

        # --- Determine need for sort
        sort = ~(fnames.index == self.header.index).all()

        return pd.concat((fnames, self.header), axis=1, sort=sort)

    # ===================================================================
    # REFRESH | SYNC WITH FILE SYSTEM 
    # ===================================================================

    def refresh(self, refresh_rows=False, refresh_cols=True, **kwargs):
        """
        Method to refresh DB() object 

          (1) Refresh fnames (rows)
          (2) Refresh header (cols) 

        """
        # --- Refresh rows 
        if self.fnames.shape[0] == 0 or refresh_rows:
            self.refresh_rows()

        # --- Refresh cols
        if refresh_rows or refresh_cols:
            self.refresh_cols()

    def refresh_rows(self, matches=None):
        """
        Method to refresh rows by updating with results of query

        """
        if self.query is {} and matches is None:
            return

        # --- Query for matches
        if matches is None:
            query = self.query.copy()
            query['root'] = self.paths['data']
            matches, _ = find_matching_files(query, verbose=False)

        self.fnames = pd.DataFrame.from_dict(matches, orient='index')

        # --- Propogate indices if meta is empty 
        if self.header.shape[0] == 0:
            self.header = pd.DataFrame(index=self.fnames.index)

        self.fnames.index.name = 'sid'
        self.header.index.name = 'sid'

    def refresh_cols(self):
        """
        Method to refresh cols

        """
        # --- Find rows with a None column entry

        # --- Update rows

        pass

    # ===================================================================
    # ITERATE AND UPDATES 
    # ===================================================================

    def cursor(self, mask=None, splits_curr=None, splits_total=None, status='Iterating | {:06d}', verbose=True, flush=False):
        """
        Method to create Python generator to iterate through dataset
        
        """
        count = 0

        df = self.df_merge(rename=False) 
        fcols = self.fnames.columns
        hcols = self.header.columns
        fsize = fcols.size
        
        # --- Apply mask
        if mask is not None:
            df = df[mask]

        # --- Create splits
        if splits_total is not None:
            r, status = self.create_splits(splits_curr, splits_total, df.shape[0], status)
            df = df.iloc[r]

        for tups in df.itertuples():

            if verbose:
                count += 1
                printp(status.format(count), count / df.shape[0], flush=flush)

            fnames = {k: '{}{}'.format(self.paths['data'], t) for k, t in zip(fcols, tups[1:1+fsize])}
            header = {k: t for k, t in zip(hcols, tups[1+fsize:])}

            yield tups[0], fnames, header

    def create_splits(self, splits_curr, splits_total, rows, status):
        """
        Method to identify current split range

        """
        # --- Read from os.environ if None
        if splits_curr is None:
            splits_curr = int(os.environ.get('SPLITS_CURR', 0))

        # --- Create splits
        splits = np.linspace(0, rows, splits_total + 1)
        splits = np.round(splits).astype('int')

        # --- Update status message
        ss = status.split('|')
        ss[0] = ss[0] + '(split == {}/{}) '.format(splits_curr + 1, splits_total)
        status = '|'.join(ss)

        return range(splits[splits_curr], splits[splits_curr + 1]), status

    def apply(self, funcs, kwargs, load=None, mask=None, replace=False):
        """
        Method to apply a series of funcs to entire spreadsheet (or partial defined by mask) 

        """
        dfs = []

        for sid, fnames, header in self.cursor(mask=mask):
            dfs.append(self.apply_row(sid, funcs, kwargs, load=load, fnames=fnames, header=header, replace=replace))

        return pd.concat(dfs, axis=0)

    def apply_row(self, sid, funcs, kwargs, load=None, fnames=None, header=None, replace=False):
        """
        Method to apply a series of funcs to single row

        """
        if fnames is None:
            sid = sid if sid in self.fnames.index else int(sid)
            fnames = self.fnames.loc[sid]

        if header is None:
            sid = sid if sid in self.header.index else int(sid)
            header = self.header.loc[sid] 

        df = pd.DataFrame()
        for func, kwargs_ in zip(funcs, kwargs):

            # --- Load all fnames if load function is provided
            if load is not None:
                to_load = {v: fnames[v] for v in kwargs_.values() if v in fnames and type(fnames[v]) is str}
                for key, fname in to_load.items():
                    fnames[key] = load(fname)[0]

            # --- Ensure all kwargs values are hashable
            kwargs_ = {k: tuple(v) if type(v) is list else v for k, v in kwargs_.items()}

            fs = {k: fnames[v] for k, v in kwargs_.items() if v in fnames}
            hs = {k: header[v] for k, v in kwargs_.items() if v in header}

            ds = func(**{**kwargs_, **fs, **hs})

            # --- Make iterable
            if df.size == 0:
                ds = {k: v if hasattr(v, '__iter__') else [v] for k, v in ds.items()}

            # --- Update df
            keys = sorted(ds.keys())
            for key in keys:
                df[key] = ds[key]

        df.index = [sid] * df.shape[0]
        df.index.name = 'sid'

        # --- In-place replace if df.shape[0] == 1
        if replace and df.shape[0] == 1:
            for key, col in df.items():
                if key in self.fnames:
                    self.fnames.at[sid, key] = col.values[0]
                if key in self.header:
                    self.header.at[sid, key] = col.values[0]

        return df

    def query(self):
        """
        Method to query db for data

        """
        pass

    # ===================================================================
    # EXTRACT and SERIALIZE 
    # ===================================================================

    def to_json(self, prefix='local://', max_rows=None):
        """
        Method to serialize contents of DB to JSON

        :return 
        
          (dict) combined = {

            [sid_00]: {

                'fnames': {
                    'dat': ..., 
                    'lbl': ...},

                'header': {
                    'sid': ...,
                    'fname': '/path/to/dat', 
                    'meta_00': ...,
                    'meta_01': ...}},

            [sid_01]: {
                'fnames': {...},  ==> from self.fnames
                'header': {...}}, ==> from self.header

            [sid_02]: {
                'fnames': {...},
                'header': {...}},
            ...

        }

        """
        # --- Prepare fnames
        rename = lambda x : '{}{}{}'.format(prefix, self.paths['data'], x)
        for col in self.fnames:
            self.fnames[col] = self.fnames[col].apply(rename)

        header = self.header.to_dict(orient='index')
        fnames = self.fnames.to_dict(orient='index')

        # --- Extract sid, fname
        extract = lambda k : {'sid': k, 'fname': fnames[k].get('dat', None)}
        header = {k: {**v, **extract(k)} for k, v in header.items()} 

        header = {k: {'header': v} for k, v in header.items()}
        fnames = {k: {'fnames': v} for k, v in fnames.items()}

        return {k: {**fnames[k], **header[k]} for k in fnames}

    def to_dict(self):
        """
        Method to create dictionary of metadata

        """
        return {attr: getattr(self, attr) for attr in self.ATTRS}

    def to_yml(self, fname=None, to_csv=True):
        """
        Method to serialize metadata of DB to YML

        """
        fname = fname or self.get_files()['yml']

        if fname is not None:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'w') as y:
                yaml.dump(self.to_dict(), y)

        if to_csv:
            self.to_csv()

    def to_csv(self, fname=None):
        """
        Method to serialize contents of DB to CSV

        """
        fname = fname or self.get_files()['csv']

        if fname is not None:

            df = self.df_merge(rename=True)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            df.to_csv(fname)

    def compress(self, cols, mask=None, fname='./data.tar.gz'):
        """
        Method to create *.tar.gz archive of the specified column(s)

        """
        # --- Filter to ensure all provided columns exist as fnames
        for col in cols:
            assert col in self.fnames

        with tarfile.open(fname, 'w:gz') as t:
            for sid, fnames, header in self.cursor(mask=mask, status='Compressing | {:06d}'):
                for col in cols:
                    if os.path.exists(fnames[col]):
                        t.add(fnames[col], arcname=fnames[col].replace(self.paths['data'], ''))
    
    def decompress(self, tar, path=None):
        """
        Method to decompress *.tar.gz archive (and sort into appropriate folders)

        """
        path = path or self.paths['data'] or '.'

        with tarfile.open(tar, 'r:gz') as t:
            t.extractall(path=path)
