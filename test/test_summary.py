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
            'root': '../data/bet', 
            'dat': 'dat.hdf5', 
            'bet': 'bet.hdf5',
            'msk': 'msk.hdf5'}

        self.db = DB(query)
        self.db.set_paths({'code': '../data/bet'})

    def test_create_summary(self):

        funcs_def = funcs.get_default_funcs_def('ct_train', 
            dats=['dat'], lbls=['bet'], classes=2)

        db = self.db.create_summary(
            kwargs=funcs.init(funcs_def, load=io.load),
            fnames=['dat', 'bet', 'msk'],
            header=[],
            yml='../data/bet/ymls/db.yml')

        # --- Create cohorts
        db.header['fg'] = db.header['bet-01']
        db.header['bg'] = ~db.header['fg']
        db.to_yml()

    @classmethod
    def tearDownClass(self):
        
        pass

if __name__ == '__main__':

    unittest.main()
