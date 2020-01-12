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
        query = {'dat': 'dat.hdf5', 'bet': 'bet.hdf5'}
        store = '../data/hdfs'
        self.db = DB(store, query)

    def test_csv(self):
        # --- Create from CSV file
        pass

    def test_yml(self):
        # --- Create from YML file
        pass

    def test_compress(self):

        # --- Set store so that relative paths are archived
        self.db.set_store('..')

        # --- Compress to current directory
        fname = './data.tar.gz'
        self.db.compress(cols=['bet'], fname=fname)
        self.assertTrue(os.path.exists(fname))

        # --- Decompress
        self.db.decompress(fname, store='.')
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
