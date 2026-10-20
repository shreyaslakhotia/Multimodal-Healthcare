import hashlib
import io
import zipfile
import pytest
from icu_mm import download


def test_copy_is_size_verified_before_finalization(tmp_path):
    destination = tmp_path / 'test.csv'
    with pytest.raises(RuntimeError, match='Source size differs'):
        download.copy_checked(io.BytesIO(b'a,b\n1,2\n'), destination, expected_bytes=99)
    assert not destination.exists()
    record = download.copy_checked(io.BytesIO(b'a,b\n1,2\n'), destination, expected_bytes=8)
    assert record == {'bytes': 8, 'sha256': hashlib.sha256(b'a,b\n1,2\n').hexdigest()}


def test_extracts_only_explicit_source_member(tmp_path, monkeypatch):
    monkeypatch.setattr(download, 'P', tmp_path / 'project')
    source = tmp_path / 'source.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('mimic-iv-2.1/hosp/patients.csv', b'a,b\n1,2\n')
        archive.writestr('../../unexpected', b'not extracted')
    download.extract_iv(source, 'hosp/patients.csv', 8)
    assert (tmp_path / 'project/raw/mimiciv/hosp/patients.csv').read_bytes() == b'a,b\n1,2\n'
    assert not (tmp_path / 'unexpected').exists()
