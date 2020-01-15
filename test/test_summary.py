import unittest
import os, shutil
from dl_utils.db import DB, funcs
from dl_utils import io

class TestSummary(unittest.TestCase):
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

    def test_create_summary(self):

        funcs_def = funcs.get_default_funcs_def('ct_train', 
            mapping={'dat': 'dat', 'lbl': 'bet'}, classes=2)

        db = self.db.create_summary(
            kwargs=funcs.init(funcs_def, load=io.load),
            fnames=['dat', 'bet'],
            header=[],
            yml='../data/ymls/db.yml')

        # --- Create cohorts
        db.header['fg'] = db.header['lbl-02']
        db.header['bg'] = ~db.header['fg']
        db.to_yml()

    @classmethod
    def tearDownClass(self):
        
        pass

if __name__ == '__main__':

    unittest.main()
