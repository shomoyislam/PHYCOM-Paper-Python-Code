import tempfile
import unittest
import zipfile
from pathlib import Path
from phycom_v4.storage import safe_extract
from phycom_v4.reporting import holm
import numpy as np


class IntegrityChecks(unittest.TestCase):
    def test_archive_traversal_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z: z.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError): safe_extract(p,Path(td)/'out')
            self.assertFalse((Path(td)/'escape.txt').exists())

    def test_holm_matches_hand_computation(self):
        np.testing.assert_allclose(holm([.04,.001,.02,.3]),[.08,.004,.06,.3])


if __name__=='__main__': unittest.main(verbosity=2)
