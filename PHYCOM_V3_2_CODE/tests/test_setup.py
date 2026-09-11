"""Regression checks for the actual Kaggle installation and result-display failures."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import bootstrap as b
from diagnostics import create_diagnostics


TARGET = dict(torch='2.10.0+cu128',cuda='12.8',python_tag='cp312',
              machine='x86_64',system='Linux',cxx11abi=True)


class SetupTests(unittest.TestCase):
    def test_exact_official_wheels_for_reported_runtime(self):
        assets = json.loads((b.ROOT/'wheel_manifest.json').read_text())['assets']
        for package,pin in [('causal_conv1d',b.CONV_PIN),('mamba_ssm',b.PIN)]:
            asset = b.choose_wheel(TARGET,package,pin,assets)
            self.assertIn('cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64',asset['name'])
            self.assertEqual(len(asset['sha256']),64)
            self.assertGreater(asset['bytes'],100_000_000)

    def test_wrong_abi_and_unsupported_torch_are_rejected(self):
        assets = json.loads((b.ROOT/'wheel_manifest.json').read_text())['assets']
        for change in [dict(cxx11abi=False),dict(torch='2.99.0+cu128'),dict(cuda=None)]:
            with self.assertRaises(RuntimeError):
                b.choose_wheel(dict(TARGET,**change),'mamba_ssm',b.PIN,assets)

    def test_local_wheel_tag_preserves_post_release(self):
        self.assertEqual(b.public_version('2.3.2.post1+cu12torch2.10cxx11abitrue'),b.PIN)
        self.assertEqual(b.public_version('1.6.2.post1+cu12torch2.10cxx11abitrue'),b.CONV_PIN)
        self.assertNotEqual(b.public_version('2.3.2.post2+cu12torch2.10'),b.PIN)

    def test_corrupt_cached_download_is_replaced_then_reused(self):
        blob=b'verified fixture bytes'
        asset=dict(name='fixture.whl',url='https://github.com/state-spaces/mamba/releases/download/test/fixture.whl',
                   sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp);(cache/asset['name']).write_bytes(b'corrupt')
            with patch.object(b.urllib.request,'urlopen',return_value=io.BytesIO(blob)) as remote:
                path=b.download_wheel(asset,cache,attempts=1)
                self.assertEqual(path.read_bytes(),blob)
                self.assertEqual(b.download_wheel(asset,cache),path)
                self.assertEqual(remote.call_count,1)

    def test_bad_download_hash_never_installs(self):
        asset=dict(name='fixture.whl',url='https://github.com/state-spaces/mamba/releases/download/test/fixture.whl',
                   sha256='0'*64,bytes=3)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(b.urllib.request,'urlopen',return_value=io.BytesIO(b'bad')):
                with self.assertRaises(RuntimeError): b.download_wheel(asset,Path(tmp),attempts=1)
            self.assertFalse((Path(tmp)/asset['name']).exists())

    def test_logs_survive_failure_before_results_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);project=root/'project';logs=project/'setup_logs'/'attempt'
            logs.mkdir(parents=True);(logs/'error.log').write_text('original root cause')
            with patch('diagnostics.subprocess.run',side_effect=FileNotFoundError('fixture')):
                archive=create_diagnostics(project,root,root/'no_results_yet')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('setup_logs/attempt/error.log'),b'original root cause')
                self.assertIn('environment.json',z.namelist())
                self.assertNotIn('FINAL_AUDIT.json',z.namelist())

    def test_pip_failure_reports_actual_error_and_saves_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'regression failed'):
                b.pip_run(['--intentionally-invalid-phycom-option'],Path(tmp),'regression',timeout=20)
            self.assertIn('no such option',(Path(tmp)/'regression.log').read_text().lower())


if __name__ == '__main__': unittest.main()
