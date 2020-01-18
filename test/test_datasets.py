import unittest
import os, shutil
from dl_utils import datasets

class TestDatasets(unittest.TestCase):
    """
    Class to test dl_utils.datasets object functionality

    """
    def test_download(self):

        datasets.download(name='bet', path='.')

        self.assertTrue(os.path.exists('./bet/proc'))
        self.assertTrue(os.path.exists('./bet/csvs'))
        self.assertTrue(os.path.exists('./bet/ymls'))

        # --- Remove
        shutil.rmtree('./bet')

    def test_prepare(self):

        pass

if __name__ == '__main__':

    unittest.main()

