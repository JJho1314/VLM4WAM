import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).with_name('download_task.py')
spec=importlib.util.spec_from_file_location('downloader',path)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class DownloadTests(unittest.TestCase):
    def test_timeout_resumes_partial_file_and_publishes_verified_result(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'scene/task/data/episode0.hdf5'
            row=dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(b'valid').hexdigest(),url='https://us.aws.cdn.hf.co/test')
            def interrupted_curl(args,**kwargs):
                part=Path(args[args.index('-o')+1])
                if not part.exists():
                    part.write_bytes(b'va')
                    return subprocess.CompletedProcess(args,28)
                part.write_bytes(part.read_bytes()+b'lid')
                return subprocess.CompletedProcess(args,0)
            with patch.object(d.subprocess,'run',side_effect=interrupted_curl):
                self.assertEqual(d.download(row,Path(root)),5)
            self.assertEqual(p.read_bytes(),b'valid')
            self.assertFalse(p.with_name(p.name+'.direct.part').exists())
    def test_verified_file_never_downloaded(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'scene/task/data/episode0.hdf5';p.parent.mkdir(parents=True);p.write_bytes(b'valid')
            row=dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(b'valid').hexdigest(),url='https://us.aws.cdn.hf.co/test')
            with patch.object(d.subprocess,'run',side_effect=AssertionError('must reuse valid file')):
                self.assertEqual(d.download(row,Path(root)),0)
    def test_corruption_repaired_only_after_hash_verification(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'scene/task/data/episode0.hdf5';p.parent.mkdir(parents=True);p.write_bytes(b'original')
            row=dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(b'valid').hexdigest(),url='https://us.aws.cdn.hf.co/test')
            def curl(args,**kwargs):
                Path(args[args.index('-o')+1]).write_bytes(b'wrong')
                return subprocess.CompletedProcess(args,0)
            with patch.object(d.subprocess,'run',side_effect=curl):
                with self.assertRaises(ValueError):d.download(row,Path(root))
            self.assertEqual(p.read_bytes(),b'original')
            def good_curl(args,**kwargs):
                Path(args[args.index('-o')+1]).write_bytes(b'valid')
                return subprocess.CompletedProcess(args,0)
            with patch.object(d.subprocess,'run',side_effect=good_curl):
                self.assertEqual(d.download(row,Path(root)),5)
            self.assertEqual(p.read_bytes(),b'valid')
    def test_path_and_url_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            for path,url in [('../outside','https://us.aws.cdn.hf.co/test'),('/tmp/outside','https://us.aws.cdn.hf.co/test'),('scene/a','http://example.com/a'),('scene/a','https://example.com/a')]:
                with self.assertRaises(ValueError):d.download(dict(path=path,url=url,sha256='0'*64),Path(root))

if __name__=='__main__':unittest.main()
