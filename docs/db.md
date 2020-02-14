# Overview

DB() is an object that facilitates interaction and manipulation of data serialized in a way that conforms to the standard `dl_*` file directory hiearchy. 

# Initialization

The following methods can be used to initialize a DB() object:

* query (dictionary + store)
* CSV file
* YML file
* shell environment variables

## query 

The query dictionary contains a series of key-value pairs where:

* key ==> name of table column to populate
* val ==> glob.glob(...) expression that can be used to localize matching files  

All queries are performed relative to query['root']. Consider the following example query dictionary:

```python
query = {
  'root': '/data/raw/root',
  'dat': 'dat.hdf5', 
  'lbl': 'lbl.hdf5' }
```

In this example, all rows of column `dat` will be populated with the results of `glob.glob('{}/**/{}'.format(query['root'], 'dat.hdf5'))`, while rows of column `lbl` will be populated with the results of `glob.glob('{}/**/{}'.format(query['root'], 'lbl.hdf5'))`.

**Usage**

```python

db = DB(query)
print(db.fnames)

dat                             lbl
[root]/(sid_00)/dat.hdf5        [root]/(sid_00)/lbl.hdf5
[root]/(sid_01)/dat.hdf5        [root]/(sid_01)/lbl.hdf5
[root]/(sid_02)/dat.hdf5        [root]/(sid_02)/lbl.hdf5
[root]/(sid_03)/dat.hdf5        [root]/(sid_03)/lbl.hdf5
...                             ...
```

## CSV file

If a table has been previously created and serialized as a `*.csv` (or `*.csv.gz`) file, then it can be passed directly ino the constructor and used to recreate a DB() object. Note that `*.csv` files can be generated from an existing DB() object using the `.to_csv()` method.

```python
# --- Instantiate 
db = DB('./db.csv.gz')

# --- Serialize
db.to_csv()
```

## YML file

If metadata in addition to a serialized table has been previously defined in a `*.yml` file, then it can be passed directly ino the constructor and used to recreate a DB() object.

Structure of a `*.yml` file:

```
# --- Default

paths:
    data: ...
    code: ...

files:
    csv: ...
    yml: ...

query:
    dat: ...
    lbl: ...

funcs: 
    - lambda: (str) ==> name of lambda function (corresponds to dl_utils.db.funcs method)
      python:
        file: (str) ==> path to Python file wth code
        name: (str) ==> name of method in file
      kwargs:
        arg0: val0
        arg1: val1
        ...
      column: 
        key0: val0
        key1: val1
        ...

# --- Client subclass

current: ...
specs: ...
infos: ...
tiles: ...

```


