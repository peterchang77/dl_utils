import os, yaml, pandas as pd 
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
# Supported functionality includes:
# 
#   * query()
#   * iterate()
#   * to_json()
#   * to_csv()
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
        Method to initialize CoreDB

        Configuration settings can be found in:

          (1) configs dictionary
          (2) YML file
          (3) CSV file

        """
        # --- Parse args
        kwargs = self.parse_args(*args, **kwargs)

        # --- Save attributes
        self.configs = self.init_configs(**kwargs) 
        self.HEADERS = self.configs['headers'] 
        
        # --- Load CSV file
        self.load_csv(self.configs['csv_file'])

        # --- Refresh
        self.refresh()

    def parse_args(self, *args, **kwargs):
        """
        Method to parse arguments into final kwargs dict

        """
        DEFAULTS = {
            'configs': None,
            'yml_file': None,
            'csv_file': None}

        # --- Convert args to args_ dict 
        args_ = {}
        for arg in args:

            if type(arg) is dict:
                args_['config'] = arg

            elif type(arg) is str:
                ext = arg.split('.')[-1]

                if ext in ['yml']:
                    args_['yml_file'] = arg

                if ext in ['csv', 'gz']:
                    args_['csv_file'] = arg

        return {**DEFAULTS, **args_, **kwargs} 

    def init_configs(self, configs, yml_file, csv_file):
        """
        Method initialize configs

        """
        DEFAULTS = {
            'query': None,
            'headers': [],
            'csv_file': None,
            'mongo': None}

        configs = configs or {}

        yml = {} if yml_file is None else yaml.load(open(yml_file), Loader=yaml.FullLoader)
        csv = {} if csv_file is None else {'csv_file': csv_file}

        return {**DEFAULTS, **configs, **yml, **csv} 

    # ===================================================================
    # CSV | LOAD, SAVE and PREPARE
    # ===================================================================

    def load_csv(self, csv_file):
        """
        Method to load CSV file

        """
        if os.path.exists(csv_file or ''):
            df = pd.read_csv(csv_file, index_col='sid')
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
        Method to refresh CoreDB 

          (1) Refresh fnames (rows)
          (2) Refresh header (cols) 

        """
        # --- Refresh rows 
        if self.fnames.shape[0] == 0 or refresh_rows:
            self.refresh_rows()

        # --- Refresh cols
        if self.fnames.shape[0] != self.header.shape[0] or refresh_cols:
            self.refresh_cols()

    def refresh_rows(self, matches=None):
        """
        Method to refresh rows by updating with results of query

        """
        if self.configs['query'] is None and matches is None:
            return

        # --- Query for matches
        if matches is None:
            matches, _ = find_matching_files(self.configs['query'], verbose=False)

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
        # --- Update headers 
        for k in self.HEADERS:
            if k not in self.header:
                self.header = None

        # --- Find rows with a None column entry

        # --- Update rows

    # ===================================================================
    # ITERATE AND UPDATES 
    # ===================================================================

    def cursor(self, query=None, verbose=True):
        """
        Method to create Python generator to iterate through dataset
        
        """
        count = 0

        df = self.df_merge(rename=False) 
        fcols = self.fnames.columns
        hcols = self.header.columns

        fsize = fcols.size

        for tups in df.itertuples():

            if verbose:
                count += 1
                printp('Iterating | {:06d}'.format(count), count / df.shape[0])

            fnames = {k: t for k, t in zip(fcols, tups[1:1+fsize])}
            header = {k: t for k, t in zip(hcols, tups[1+fsize:])}

            yield tups[0], fnames, header

    def update_all(self):
        """
        Method to re-index all files

        """
        # --- Query for all matching data

        # --- Iterate through each match with self.update_row()

    def update_row(self, sid, arrs, overwrite=False, **kwargs):
        """
        Method to update db row with current arrays

        NOTE: add custom header data by registering methods in HEADER_FUNCTIONS dict

        """
        for h in self.HEADERS:
            if h in self.HEADER_FUNCTIONS:
                self.HEADER_FUNCTIONS[h](sid, arrs, **kwargs)

    def query(self):
        """
        Method to query db for data

        """
        pass

    def iterate(self, func):
        """
        Method to iterate through data and apply provided func

        """
        pass

    # ===================================================================
    # EXTRACT and SERIALIZE 
    # ===================================================================

    def to_json(self, max_rows=None):
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
        header = self.header.to_dict(orient='index')
        fnames = self.fnames.to_dict(orient='index')

        # --- Extract sid, fname
        extract = lambda k : {'sid': k, 'fname': fnames[k].get('dat', None)}
        header = {k: {**v, **extract(k)} for k, v in header.items()} 

        # --- Prepend local:// to fnames 
        convert = lambda d : {k: 'local://%s' % v for k, v in d.items()}
        fnames = {k: convert(v) for k, v in fnames.items()}

        header = {k: {'header': v} for k, v in header.items()}
        fnames = {k: {'fnames': v} for k, v in fnames.items()}

        return {k: {**fnames[k], **header[k]} for k in fnames}

    def to_csv(self, csv=None):
        """
        Method to serialize contents of DB to CSV

        """
        csv = csv or self.configs['csv_file']

        if csv is not None:

            df = self.df_merge(rename=True)
            df.to_csv(csv)

# ===============================================
# db = DB(yml_file='./configs.yml')
# db.to_csv()
# ===============================================
