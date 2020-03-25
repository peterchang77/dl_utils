import os, yaml, numpy as np, pandas as pd, tarfile
from .query import find_matching_files
from . import funcs
from ..general import printd, printp
from ..general import tools as jtools
from ..display import interleave

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

        """
        # --- Initialize default values
        self.init_defaults(*args, **kwargs)

        # --- Load YML and CSV files
        self.load_yml()
        self.set_id(update_fnames=False)
        self.load_csv(**kwargs)

        # --- Refresh
        self.init_fdefs()
        self.refresh()

    def init_custom(self, *args, **kwargs): pass

    def init_defaults(self, *args, **kwargs):
        """
        Method to initialize default DB() configurations

        """
        DEFAULTS = {
            '_id': {'project': None, 'version': None},
            'paths': {'data': '', 'code': ''},
            'files': {'csv': './csvs/db-all.csv.gz', 'yml': './ymls/db-all.yml'},
            'query': {},
            'sform': {},
            'fdefs': []}

        args_, files, paths = self.parse_args(*args, **kwargs)

        # --- Initialize default values
        configs = {**DEFAULTS, **args_, **kwargs} 

        if 'root' in configs['query']:
            paths['data'] = configs['query'].pop('root')

        # --- Initialize default paths and files
        configs['paths'] = {**DEFAULTS['paths'], **paths, **kwargs.get('paths', {})}
        configs['files'] = {**DEFAULTS['files'], **files, **kwargs.get('files', {})}

        # --- Attributes to serialize in self.to_yml(...) 
        self.ATTRS = ['_id', 'files', 'query', 'sform', 'fdefs']

        # --- Set attributes 
        for attr in self.ATTRS:
            setattr(self, attr, configs[attr])

        self.paths = configs['paths']

    def parse_args(self, *args, **kwargs):
        """
        Method to parse arguments into final kwargs dict

        """
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
                args_['fdefs'] = arg

        for k in ['csv', 'yml']:
            if k in kwargs:
                files[k] = kwargs[k]

        # --- Load custom functions
        self.load_func = kwargs.pop('load', None) 

        return args_, files, paths

    def set_id(self, project_id=None, version_id=None, paths=None, update_fnames=True):
        """
        Method to set project_id of database and remove prefix from fnames

        """
        pid, vid = self.detect_id()
        project_id = project_id or pid
        version_id = version_id or vid 

        # --- Get paths
        if project_id is not None:

            # --- Replace empty self.paths
            if paths is None:
                paths = jtools.get_paths(project_id) 
                for k in self.paths:
                    if self.paths[k] == '':
                        self.paths[k] = paths.get(k, '')

            # --- Replace all self.paths
            else:
                paths = jtools.set_paths(project_id, paths)
                self.paths = {**self.paths, **paths}

        # --- Set paths attribute in current object
        if version_id is not None and self.paths['code'] != '':
            if not os.path.exists(self.paths['code']):
                if self.paths['code'][-5:] == '/data':
                    self.paths['code'] = '{}/{}/data'.format(self.paths['code'][:-5], version_id)
                else:
                    self.paths['code'] = '{}/{}'.format(self.paths['code'], version_id)

        # --- Update self.files
        self._id['project'] = project_id
        self._id['version'] = version_id 
        self.set_files()

        # --- Update fnames
        if update_fnames:
            for col in self.fnames:
                self.update_fnames(col, self.sform.get(col, '{curr}'))

    def detect_id(self):
        """
        Method to attempt auto-detection of project_id based on:
        
          (1) self._id attribute
          (2) Loaded *.yml file if any 
          (3) Current file path

        NOTE: for version detection, the following dir structure is assumed:

          * full path ==> {root}/[v][0-9]/data
          * code path ==> {root}/data

        """
        project_id = self._id['project'] 
        version_id = self._id['version'] 

        # --- Determine auto detect path to use
        auto = self.get_files()['yml']
        if not os.path.exists(auto):
            auto = os.getcwd()

        if project_id is None:

            configs = jtools.load_configs('paths')

            for pid, paths in configs.items():

                data = paths['data']
                code = paths['code']
                if code[-5:] == '/data':
                    code = code[:-5]

                if (code in auto and code != '') or (data in auto and data != ''):
                    project_id = pid

                    # --- Attempt extraction of version
                    if version_id is None and code in auto and code != auto:

                        suffix = auto.split(code)[1][1:]
                        if '/' in suffix:
                            suffix = suffix.split('/')[0]
                            if len(suffix) > 1:
                                if suffix[0] == 'v' and suffix[1].isnumeric():
                                    version_id = suffix

                    break

        return project_id, version_id 

    def set_files(self, files=None):
        """
        Method to set self.files

        :params

          (dict) files : {'csv': ..., 'yml': ...}, OR
          (str)  files : inferred string pattern e.g. 'db-all' 

        """
        if type(files) is str:
            files = {
                'csv': '/csvs/{}.csv.gz'.format(files),
                'yml': '/ymls/{}.yml'.format(files)}

        else:
            files = {**self.files, **(files or {})} 

        # --- Expand and replace paths
        files = {k: os.path.abspath(v) for k, v in files.items()}
        files = {k: v.replace(self.paths['code'], '') for k, v in files.items()}

        self.files = files

    def get_files(self):
        """
        Method to get full file paths

        """
        return {k: '{}{}'.format(self.paths['code'], v) for k, v in self.files.items()}

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

            for attr, config in configs.items():
                setattr(self, attr, config)

            # --- Infer paths['code'] and files['yml'] 
            fname = os.path.abspath(fname)
            loc = fname.find(self.files['yml']) 
            self.paths['code'] = fname[:loc]
            self.files['yml'] = fname[loc:]

    # ===================================================================
    # CSV | LOAD and PREPARE
    # ===================================================================

    def load_csv(self, fname=None, lazy=True, **kwargs):
        """
        Method to load CSV file

        """
        # --- Initialize from manually passed kwargs if possible
        if 'fnames' in kwargs:
            self.fnames = kwargs.pop('fnames', None)
            self.header = kwargs.pop('header', pd.DataFrame(index=self.fnames.index))

            return

        fname = fname or self.get_files()['csv'] or ''

        if os.path.exists(fname):
            df = pd.read_csv(fname, index_col='sid', keep_default_na=False)
        else: 
            df = pd.DataFrame()
            df.index.name = 'sid'

        # --- Split df into fnames and header 
        self.fnames, self.header = self.df_split(df)

        # --- Load full fnames 
        if not lazy:
            self.fnames = self.fnames_expand()
            self.sform = {}

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

    def refresh(self, cols=None, fdefs=None, load=None, update_query=False, **kwargs):
        """
        Method to refresh DB() object 

        """
        # --- Update query 
        if self.fnames.shape[0] == 0 or update_query:
            self.update_query()

        if fdefs is not None:
            # --- Create columns via fdefs
            self.create_column(fdefs=fdefs, load=load, **kwargs)

        else:
            # --- Create columns via defs
            cols = cols or []
            if type(cols) is str:
                cols = [cols]

            for col in cols:
                self.create_column(col=col, load=load, **kwargs)

    def update_query(self, matches=None):
        """
        Method to refresh rows by updating with results of query

        """
        if len(self.query) == 0 and matches is None:
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

    def create_column(self, col=None, fdefs=None, load=None, mask=None, indices=None, split=None, splits=None, flush=False, replace=True, skip_existing=True, **kwargs):
        """
        Method to create column

        """
        # --- Initialize fdefs
        fdefs = self.find_fdefs(col) if fdefs is None else funcs.init(fdefs)

        if len(fdefs) == 0:
            return

        for sid, fnames, header in self.cursor(mask=mask, indices=indices, flush=flush, split=split, splits=splits):

            if col is not None:
                update = not os.path.exists(fnames[col]) if col in fnames else header[col] == ''
            else:
                update = True

            if update or not skip_existing:
                self.apply_row(sid, fdefs, load=load, fnames=fnames, header=header, replace=replace)

    # ===================================================================
    # FDEFS FUNCTIONS 
    # ===================================================================

    def init_fdefs(self):
        """
        Method to initialize self.fdefs

        """
        # --- Create col_to_fdef dict
        self.col_to_fdef = {}
        for n, fdef in enumerate(self.fdefs):
            if 'return' in fdef:
                for v in fdef['return'].values():
                    self.col_to_fdef[v] = n 

    def find_fdefs(self, cols):
        """
        Method to find and initialize all fdefs corresponding to provided columns

        """
        if type(cols) is str:
            cols = [cols]

        fdefs = []
        for col in cols:
            if col in self.col_to_fdef:

                fdef = self.col_to_fdef[col]

                # --- Initialize if needed
                if type(fdef) is int:
                    rets = self.fdefs[fdef]['return'].values()
                    fdef = funcs.init([self.fdefs[fdef]])
                    self.col_to_fdef.update({k: fdef for k in rets})

                fdefs += fdef

        return fdefs

    # ===================================================================
    # FNAMES FUNCTIONS 
    # ===================================================================

    def exists(self, cols=None, verbose=True, ret=False):
        """
        Method to check if fnames exists

        """
        exists = {}
        fnames = self.fnames_expand(cols=cols)
        ljust = max([len(c) for c in fnames.columns])

        for col in fnames.columns:
            exists[col] = fnames[col].apply(lambda x : os.path.exists(x)).to_numpy()

            if verbose:
                printd('COLUMN: {} | {:06d} / {:06d} exists'.format(col.ljust(ljust), exists[col].sum(), fnames[col].shape[0]))

        if ret:
            return exists

    def fnames_like(self, suffix, like=None):
        """
        Method to create new fnames in pattern based on column defined by like

        """
        like = like or self.fnames.columns[0]

        return self.fnames[like].apply(lambda x : '{}/{}'.format(os.path.dirname(x), suffix))

    def fnames_expand(self, cols=None):
        """
        Method to expand fnames based on defined str formats (sform)

        """
        root = self.paths['data']

        if cols is None:
            cols = self.fnames.columns

        if type(cols) is str:
            cols = [cols]

        fnames = pd.DataFrame(index=self.fnames.index)
        fnames.index.name = 'sid'

        for col in cols:
            if col in self.sform:
                fnames[col] = [self.sform[col].format(root=root, curr=f, sid=s)
                    for s, f in zip(self.fnames.index, self.fnames[col])]
            else:
                fnames[col] = self.fnames[col]

        return fnames 

    def fnames_expand_single(self, sid=None, index=None, fnames=None):
        """
        Method to expand a single fnames dict based on str formats (sform)

        """
        if fnames is None:

            if index is not None:
                fnames = self.fnames.iloc[index].to_dict()
                sid = self.fnames.index[index]

            else: 
                assert sid is not None
                assert self.fnames.index.is_unique
                fnames = self.fnames.loc[sid].to_dict()

        fnames = fnames or {}

        return {k: self.sform[k].format(root=self.paths['data'], curr=v, sid=sid) 
            if k in self.sform else v for k, v in fnames.items()}

    def restack(self, columns_on, marker=None, suffix=''):
        """
        Method to stack specified columns on existing fname

        All other fnames/header data will be copied from existing row

        :params

          (str) marker : if provided, create new header indicating stack status
          (str) suffix : if provided, suffix to add to index e.g. {sid}-{suffx}{:03d}

        """
        fnames = []
        header = []

        # --- Create new header
        if marker is not None:
            self.header[marker] = False

        # --- Create baseline fnames
        cols_ = [c for cols in columns_on.values() for c in cols]
        cols_ = [c for c in self.fnames.columns if c not in cols_]
        fnames.append(self.fnames[cols_])
        header.append(self.header)

        n = len(next(iter(columns_on.values())))
        for i in range(n):

            f = fnames[0].copy()
            h = header[0].copy()

            for on, cols in columns_on.items():
                f[on] = self.fnames[cols[i]]

            index = ['{}-{}{:03d}'.format(sid, suffix, i) for sid in f.index]
            f.index = index
            h.index = index
            f.index.name = 'sid'
            h.index.name = 'sid'

            if marker is not None:
                h[marker] = True
            
            fnames.append(f)
            header.append(h)

        # --- Update index of 0th index
        fnames[0].index = ['{}-{}org'.format(sid, suffix) for sid in fnames[0].index]
        header[0].index = fnames[0].index 

        # -- Combine
        self.fnames = pd.concat(fnames, axis=0)
        self.header = pd.concat(header, axis=0)

    def init_sform(self, subdirs=None, json=[], cols=None):
        """
        Method to initialize default sform based of current fnames columns

        Default fnames column format:
        
          [base]-[dirs](-[suffix])

          base ==> used for file basename e.g. dat-xxx = dat.hdf5
          dirs ==> used for subdirectory (inferred by subdirs mapping)

        Default sform format:

          '{root}/proc/{subdir}/{sid}/{base}.{ext}'

        Default extension is *.hdf5 unless base is in JSON list

        """
        cols = cols or self.fnames.columns
        json += ['pts', 'box', 'vec']

        # --- Default subdirs mappings
        subdirs = subdirs or {}
        subdirs['dcm'] = 'dcm'
        subdirs['raw'] = 'raw'

        sform = {}

        for col in cols:
            parts = len(col.split('-'))
            if parts > 1:

                # --- Infer subdirectory
                subdir = col.split('-')[1]
                if subdir in subdirs:
                    subdir = subdirs[subdir]

                # --- Infer basename
                base = col.split('-')[0]
                if parts > 2:
                    base += '-' + '-'.join(col.split('-')[2:])

                if base == 'dcm':

                    sform[col] = '{root}/proc/dcm/{sid}/'

                else:

                    # --- Infer extension
                    ext = 'json' if col.split('-')[0] in json else 'hdf5'

                    # --- Create sform
                    sform[col] = '{{root}}/proc/{subdir}/{{sid}}/{base}.{ext}'.format(
                        root='root',
                        subdir=subdir,
                        sid='sid',
                        base=base,
                        ext=ext) 

        self.sform.update(sform)

    def update_sform(self, sform='{root}{curr}'):
        """
        Method to update sform for specified columns and corresponding fnames

        :params

          (dict) sform = {
                    [col-00]: [sform-pattern],
                    [col-01]: [sform-pattern],
                    [col-02]: [sform-pattern], ... }

        NOTE: if str is provided, sform pattern is propogated to all columns

        """
        if type(sform) is str:
            sform = {k: sform for k in self.fnames}

        if type(sform) is not dict:
            printd('ERROR, sform is not formatted correctly')
            return

        for col, s in sform.items():
            self.update_fnames(col, s)

        self.sform.update(sform)

    def update_fnames(self, col, sform):
        """
        Method to update fnames column based on provided sform:

          (1) Fully expand current fnames
          (2) Recompress fnames based on new sform

        NOTE: this method is intended as an internal function; it does not update sform
        after manipulation of fnames. The recommended approach for most sform updates
        is the self.update_sform(...) method.

        """
        assert col in self.fnames

        # --- Recompress fnames 
        root = self.paths['data']
        loc_r = sform.find('{root}')
        loc_c = sform.find('{curr}')
        loc_s = sform.find('{sid}')

        # --- CONDITION: all fnames inferred from sform alone (no '{curr}') 
        if (loc_c == -1):
            self.fnames[col] = ''
            return

        # --- CONDITION: only '{curr}' without existing self.sform entry
        if sform == '{curr}' and col not in self.sform:
            return

        # --- Expand fnames (all other conditions require at least this)
        self.fnames[col] = self.fnames_expand(cols=col)

        # --- CONDITION: only '{curr}' with existing self.sform entry
        if sform == '{curr}':
            return

        # --- CONDITION: no '{sid}' and '{curr}' at the end e.g. '{root}{curr}'
        if (loc_s == -1) and (sform[-6:] == '{curr}'):
            prefix = sform[:-6].format(root=root)
            if prefix in self.fnames[col].iloc[0]:
                n = len(prefix)
                self.fnames[col] = self.fnames[col].apply(lambda x : x[n:])

            return

        # --- ELSE: Create generic function for all alternate cases
        prefix, suffix = sform.split('{curr}')

        def recompress(f, sid):
            """
            Method to remove formatted prefix / suffix from string f

            """
            f = f.replace(prefix.format(root=root, sid=sid), '')
            f = f.replace(suffix.format(root=root, sid=sid), '')

            return f
            
        self.fnames[col] = [recompress(f, s) for f, s in zip(self.fnames[col], self.fnames.index)]

    # ===================================================================
    # DATA AUGMENTATION 
    # ===================================================================

    def create_fnames_augmented(self, column, n, basename=None):
        """
        Method to create augmented fnames in the following dir structure:

        /[root]/
          |--- dat.hdf5
          ...
          |--- augs/
               |--- aug-000/
                    |--- dat.hdf5
                    ...
               |--- aug-001/
               |--- aug-002/
               ...

        :params

          (str) column : name of column to augment
          (int) n      : number of augmentations

        """
        assert type(column) is str
        assert type(n) is int
        assert column in self.fnames

        roots = [os.path.dirname(f) for f in self.fnames[column]]
        bases = [os.path.basename(f) for f in self.fnames[column]] if basename is None else [basename] * len(roots)

        keys = ['{}-aug-{:03d}'.format(column, i) for i in range(n)]

        for i, key in enumerate(keys):

            self.fnames[key] = \
                ['{}/augs/aug-{:03d}/{}'.format(r, i, b) for r, b in zip(roots, bases)]

        return keys

    def realign(self, cols, align_with, jars, mask=None, flush=True):
        """
        Method to realign volumes in provided columns with reference volume

        """
        for fname in cols + [align_with]:
            assert fname in self.fnames

        assert hasattr(jars, 'create')

        for sid, fnames, header in self.cursor(mask=mask, flush=flush):

            # --- Load reference volume
            ref = jars.create(fnames[align_with])

            # --- Align and save
            for col in cols:

                arr = jars.create(fnames[col])
                arr = arr.align_with(ref)

                dst = '{}/{}.hdf5'.format(
                    os.path.dirname(fnames[align_with]),
                    os.path.splitext(os.path.basename(fnames[col]))[0])

                arr.to_hdf5(dst)
                self.fnames.at[sid, col] = dst

    # ===================================================================
    # ITERATE AND UPDATES 
    # ===================================================================

    def row(self, index=None, sid=None):
        """
        Method to return single row at self.fnames and self.header

        """
        fnames = self.fnames_expand_single(sid=sid, index=index)
        header = self.header.loc[sid].to_dict() if sid is not None else self.header.iloc[index].to_dict()

        return {**fnames, **header} 

    def cursor(self, mask=None, indices=None, split=None, splits=None, status='Iterating | {:06d}', verbose=True, flush=False, **kwargs):
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

        # --- Apply indices
        if indices is not None:
            df = df.iloc[indices]

        # --- Create splits
        if splits is not None:
            r, status = self.create_splits(split, splits, df.shape[0], status)
            df = df.iloc[r]

        for tups in df.itertuples():

            if verbose:
                count += 1
                printp(status.format(count), count / df.shape[0], flush=flush)

            fnames = {k: t for k, t in zip(fcols, tups[1:1+fsize])}
            header = {k: t for k, t in zip(hcols, tups[1+fsize:])}

            fnames = self.fnames_expand_single(sid=tups[0], fnames=fnames)

            yield tups[0], fnames, header

    def create_splits(self, split, splits, rows, status):
        """
        Method to identify current split range

        """
        # --- Read from os.environ if None
        if split is None:
            split = int(os.environ.get('DL_SPLIT', 0))

        # --- Create splits
        sp = np.linspace(0, rows, splits + 1)
        sp = np.round(sp).astype('int')

        # --- Update status message
        ss = status.split('|')
        ss[0] = ss[0] + '(split == {}/{}) '.format(split + 1, splits)
        status = '|'.join(ss)

        return range(sp[split], sp[split + 1]), status

    def apply(self, fdefs, load=None, mask=None, indices=None, flush=False, replace=False, **kwargs):
        """
        Method to apply a series of lambda functions
    
        :params

          (str)  fdefs = 'mr_train', 'ct_train', ... OR

          (list) fdefs = [{

            'lambda': 'coord', 'stats', ... OR lambda function,
            'python': {'file': ..., 'name': ...}
            'kwargs': {...},
            'return': {...}

            }]

        See dl_utils.db.funcs.init(...) for more information about fdefs list

        """
        fdefs = funcs.init(fdefs, **kwargs)

        dfs = []
        for sid, fnames, header in self.cursor(mask=mask, indices=indices, flush=flush):
            dfs.append(self.apply_row(sid, fdefs, load=load, fnames=fnames, header=header, replace=replace))

        return pd.concat(dfs, axis=0)

    def apply_row(self, sid, fdefs, load=None, fnames=None, header=None, replace=False, clear_arrays=True, **kwargs):
        """
        Method to apply a series of lambda functions to single row

        """
        # --- Set load func
        load = load or self.load_func

        if fnames is None:
            sid = sid if sid in self.fnames.index else int(sid)
            fnames = self.fnames_expand_single(sid=sid)

        if header is None:
            sid = sid if sid in self.header.index else int(sid)
            header = self.header.loc[sid] 

        df = pd.DataFrame()
        for fdef in fdefs:

            lambda_ = fdef['lambda']
            kwargs_ = fdef['kwargs']
            return_ = fdef['return']

            # --- Load all fnames if load function is provided
            if load is not None:
                to_load = {v: fnames[v] for v in kwargs_.values() if v in fnames and type(fnames[v]) is str}
                for key, fname in to_load.items():
                    if os.path.exists(fname):
                        fnames[key] = load(fname)
                        if type(fnames[key]) is tuple:
                            fnames[key] = fnames[key][0]

                    else:
                        # --- Recursively create new fnames 
                        fdefs = self.find_fdefs(cols=key)
                        if len(fdefs) > 0: 
                            df_ = self.apply_row(sid, fdefs, load=load, fnames=fnames, header=header, replace=replace, clear_arrays=True)
                            if df_.shape[0] > 0:
                                fnames.update({k: v for k, v in df_.iloc[0].items() if k in fnames})

            # --- Ensure all kwargs values are hashable
            kwargs_ = {k: tuple(v) if type(v) is list else v for k, v in kwargs_.items()}

            fs = {k: fnames[v] for k, v in kwargs_.items() if v in fnames}
            hs = {k: header[v] for k, v in kwargs_.items() if v in header}

            ds = lambda_(**{**kwargs_, **fs, **hs})

            # --- Make iterable
            if df.size == 0:
                ds = {k: v if hasattr(v, '__iter__') else [v] for k, v in ds.items()}

            # --- Update df
            if len(return_) == 0:
                return_ = {k: k for k in ds.keys()}

            keys = sorted(return_.keys())
            for key in keys:
                df[return_[key]] = ds.get(key, None)

        df.index = [sid] * df.shape[0]
        df.index.name = 'sid'

        # --- In-place replace if df.shape[0] == 1
        if replace and df.shape[0] == 1:
            for key, col in df.items():

                # --- Save fnames
                if key in fnames:
                    if type(fnames[key]) is str:

                        # --- Check default methods
                        for method in ['to_hdf5', 'to_json']:
                            if hasattr(col.values[0], method):
                                getattr(col.values[0], method)(fnames[key])
                                break

                        if clear_arrays:
                            df.at[sid, key] = fnames[key]

                # --- Save header
                if key in header:
                    self.header.at[sid, key] = col.values[0]

        return df

    def query(self):
        """
        Method to query db for data

        """
        pass

    def montage(self, dat, lbl=None, coord=(0.5, 0.5, 0.5), shape=(1, 0, 0), N=None, load=None, func=None, **kwargs):
        """
        Method to load montage of database

        """
        # --- Set load func
        load = load or self.load_func
        if load is None:
            return

        # --- Unravel tuples
        unravel = lambda x : x[0] if type(x) is tuple else x

        # --- Aggregate
        dats, lbls = [], []

        # --- Initialize default kwargs
        if 'mask' not in kwargs and 'indices' not in kwargs and N is not None:
            kwargs['indices'] = np.arange(N ** 2)

        for sid, fnames, header in self.cursor(**kwargs):

            # --- Load dat
            infos = {'coord': coord, 'shape': shape} if func is None else None
            d = unravel(load(fnames[dat], infos=infos, **kwargs))
            dats.append(d[:1])

            # --- Load lbl
            if lbl is not None:
                infos = {'coord': coord, 'shape': shape} if func is None else None
                l = unravel(load(fnames[lbl], infos=infos, **kwargs)) 
                lbls.append(l[:1])

                # --- Apply label conversion and slice extraction function
                if func is not None:
                    dats[-1], lbls[-1] = func(dat=d, lbl=l, **kwargs)

        # --- Interleave dats
        dats = interleave(np.stack(dats), N=N)

        # --- Interleave lbls
        if lbl is not None:
            lbls = interleave(np.stack(lbls), N=N)

        return dats, lbls

    # ===================================================================
    # CREATE SUMMARY DB 
    # ===================================================================

    def create_summary(self, fdefs, fnames=[], header=[], folds=5, yml='./ymls/db.yml', **kwargs):
        """
        Method to generate summary training stats via self.apply(...) operation

        :params

          (dict) kwargs : kwargs initialized via funcs.init(...)
          (list) fnames : list of fnames to join
          (list) header : list of header columns to join

          (int)  folds  : number of cross-validation folds
          (str)  path   : directory path to save summary (ymls/ and csvs/)

        """
        df = self.apply(fdefs, **kwargs)

        # --- Create merged
        mm = self.df_merge(rename=False)
        mm = mm[fnames + header]

        # --- Create validation folds
        self.create_valid_column(df=mm, folds=folds)

        # --- Join and split
        header = header + ['valid'] + list(df.columns)
        df = df.join(mm)

        # --- Create new DB() object
        db = DB(fnames=df[fnames], header=df[header])
        
        # --- Serialize
        db.set_paths(self.paths, update_fnames=False)
        db.set_files(yml)
        db.sform = {k: self.sform[k] for k in fnames if k in self.sform}
        db.to_yml()

        # --- Final output
        printd('Summary complete: %i patients | %i slices' % (mm.shape[0], df.shape[0]))

        return db

    def create_valid_column(self, df=None, folds=5):

        if df is None:
            df = self.header

        v = np.arange(df.shape[0]) % folds 
        v = v[np.random.permutation(v.size)]

        df['valid'] = v

    # ===================================================================
    # EXTRACT and SERIALIZE 
    # ===================================================================

    def to_json(self, dat, lbl=None, hdr=None, prefix='local://', max_rows=None, exists_only=True):
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
        # --- Prepare cols 
        cols = {dat: 'dat'} if lbl is None else {dat: 'dat', lbl: 'lbl'} 

        # --- Prepare fnames, header
        fnames = self.fnames_expand(cols=cols)
        header = self.header[hdr or self.header.columns]

        # --- Keep existing files
        if exists_only:
            mask = next(iter(self.exists(cols=[dat], verbose=False, ret=True).values()))
            fnames = fnames[mask]
            header = header[mask]

        # --- Add prefix
        for col in fnames.columns:
            fnames[col] = ['{}{}'.format(prefix, f) for f in fnames[col]]

        # --- Change names
        fnames = fnames.rename(columns=cols)

        fnames = fnames.to_dict(orient='index')
        header = header.to_dict(orient='index')

        # --- Extract sid, fname
        extract = lambda k : {'sid': k, 'fname': fnames[k]['dat']}
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
                yaml.dump(self.to_dict(), y, sort_keys=False)

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

    def archive(self, cols, fname='./data.tar', mask=None, include_db=False, compress=False):
        """
        Method to create *.tar(.gz) archive of the specified column(s)

        """
        # --- Filter to ensure all provided columns exist as fnames
        for col in cols:
            assert col in self.fnames

        # --- Determine compression
        if compress:
            mode = 'w:gz'
            if fname[-2:] != 'gz':
                fname = fname + '.gz'
        else:
            mode = 'w'

        with tarfile.open(fname, mode, dereference=True) as tf:

            if include_db:
                for fname in self.get_files().values():
                    tf.add(fname, arcname=fname.replace(self.paths['code'], ''))

            for sid, fnames, header in self.cursor(mask=mask, status='Compressing | {:06d}'):
                for col in cols:
                    if os.path.exists(fnames[col]):
                        tf.add(fnames[col], arcname=fnames[col].replace(self.paths['data'], ''))
    
    def unarchive(self, tar, path=None):
        """
        Method to unpack *.tar(.gz) archive (and sort into appropriate folders)

        """
        path = path or self.paths['data'] or '.'
        jtools.unarchive(tar, path)
