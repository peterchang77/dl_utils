import unittest
import os, shutil
from dl_utils.db import DB

class TestDB(unittest.TestCase):
    """
    Class to test DB() object functionality

    """
    @classmethod
    def setUpClass(self):
        """
        Set up test DB() object

        """
        # --- Create from query dictionary
        query = {
            'root': '..', 
            'dat': 'data/**/dat.hdf5', 
            'bet': 'data/**/bet.hdf5'}

        self.db = DB(query)
        self.db.set_paths({'code': '.'})

    def test_csv(self):

        # --- Create CSV file
        self.db.to_csv()

        # --- Reload from CSV
        fname = self.db.get_files()['csv']
        db = DB(fname)

        self.assertTrue((db.fnames == self.db.fnames).all().all())
        self.assertTrue((db.header == self.db.header).all().all())

        # --- Remove
        for key, fname in self.db.get_files().items():
            if os.path.exists(fname):
                os.remove(fname)
                shutil.rmtree(os.path.dirname(fname))

    def test_yml(self):

        # --- Create YML file
        self.db.to_yml()

        # --- Reload from YML 
        fname = self.db.get_files()['yml']
        db = DB(fname)

        self.assertTrue((db.fnames == self.db.fnames).all().all())
        self.assertTrue((db.header == self.db.header).all().all())

        for attr in db.ATTRS:
            self.assertTrue(getattr(db, attr) == getattr(self.db, attr))

        # --- Remove
        for key, fname in self.db.get_files().items():
            if os.path.exists(fname):
                os.remove(fname)
                shutil.rmtree(os.path.dirname(fname))

    def test_compress(self):

        # --- Compress to current directory
        fname = './data.tar.gz'
        self.db.compress(cols=['bet'], fname=fname)
        self.assertTrue(os.path.exists(fname))

        # --- Decompress
        self.db.decompress(fname, path='.')
        self.assertTrue(os.path.exists('./data/hdfs'))

        # --- Remove
        os.remove(fname)
        shutil.rmtree('./data')

    @classmethod
    def tearDownClass(self):
        
        pass

class TestFuncs(unittest.TestCase):
    """
    Class to test anonymous functions module

    """
    pass

if __name__ == '__main__':

    unittest.main()
