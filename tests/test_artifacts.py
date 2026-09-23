import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.artifacts import build_signature, cache_matches, file_sha256, write_manifest


class FileHashTests(unittest.TestCase):
    def test_shared_checkpoint_is_read_once_for_multiple_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"shared weights")
            images = [root / "one.png", root / "two.png"]
            for image in images:
                image.write_bytes(image.name.encode())

            original_open = Path.open
            checkpoint_reads = []

            def counting_open(path, *args, **kwargs):
                if path == checkpoint:
                    checkpoint_reads.append(path)
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", counting_open):
                signatures = [
                    build_signature(
                        inputs={"checkpoint": checkpoint, "image": image},
                        parameters={"step": "test"},
                    )
                    for image in images
                ]
            self.assertEqual(len(checkpoint_reads), 1)
            self.assertEqual(
                signatures[0]["inputs"]["checkpoint"]["sha256"],
                hashlib.sha256(b"shared weights").hexdigest(),
            )
            self.assertNotEqual(
                signatures[0]["inputs"]["image"], signatures[1]["inputs"]["image"]
            )

    def test_same_size_rewrite_with_restored_mtime_invalidates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            artifact = root / "cached"
            source.write_bytes(b"old")
            # Give the old file a stable mtime that the rewrite will preserve.
            os.utime(source, ns=(1_000_000_000, 1_000_000_000))
            old_stat = source.stat()
            artifact.write_bytes(b"result")
            before = build_signature(inputs={"source": source}, parameters={})
            write_manifest(artifact, before)
            self.assertTrue(cache_matches(artifact, before))

            source.write_bytes(b"new")
            os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            after = build_signature(inputs={"source": source}, parameters={})
            self.assertEqual(source.stat().st_mtime_ns, old_stat.st_mtime_ns)
            self.assertEqual(
                after["inputs"]["source"]["sha256"],
                hashlib.sha256(b"new").hexdigest(),
            )
            self.assertFalse(cache_matches(artifact, after))

    def test_atomic_replacement_with_same_size_and_mtime_is_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input"
            source.write_bytes(b"old")
            old_stat = source.stat()
            before = file_sha256(source)
            replacement = Path(tmp) / "replacement"
            replacement.write_bytes(b"new")
            os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            replacement.replace(source)
            self.assertNotEqual(file_sha256(source), before)

    def test_deleted_file_does_not_return_cached_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input"
            source.write_bytes(b"old")
            file_sha256(source)
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                file_sha256(source)

    def test_file_changed_during_hash_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input"
            source.write_bytes(b"old")
            original_open = Path.open

            def changing_open(path, *args, **kwargs):
                with original_open(path, "wb") as stream:
                    stream.write(b"replacement")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", changing_open):
                with self.assertRaisesRegex(OSError, "File changed while hashing"):
                    file_sha256(source)
            self.assertEqual(
                file_sha256(source), hashlib.sha256(b"replacement").hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
