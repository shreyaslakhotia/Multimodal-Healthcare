"""Selective, restartable source acquisition. Run only in a CPU Slurm job.

Never print response bodies, patient rows, image paths, or authentication data.
Public Kaggle endpoints are used; an HTTP authorization failure stops this job.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
import zipfile

P = Path('/projects/icumultimodal')
A = Path('/projects/icumultimodalarchive')
IV = 'mangeshwagle/mimic-iv-2-1'
CXR = 'cvlearning0616/the-mimic-cxr-database'
NEEDED = ('hosp/admissions.csv', 'hosp/patients.csv', 'hosp/labevents.csv',
          'hosp/d_labitems.csv', 'hosp/diagnoses_icd.csv', 'hosp/d_icd_diagnoses.csv',
          'icu/icustays.csv', 'icu/chartevents.csv', 'icu/d_items.csv')
CHUNK = 8 * 1024 * 1024


def public_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def usage_guard(temporary_extra=0):
    # Never recursively scan dataset/image directories. Start with CPU-measured
    # environment/cache sizes and stat only this job's exact known outputs.
    sizes = P / 'results' / 'setup' / 'storagebytes.txt'
    if not sizes.exists():
        raise RuntimeError('Environment storage accounting must finish first')
    total = sum(int(line.split()[0]) for line in sizes.read_text().splitlines())
    total += 1_000_000_000  # Conservative code, logs, manifests and cache growth.
    targets = [P / 'raw' / 'mimiciv' / relative for relative in NEEDED]
    targets += [P / 'raw' / 'mimiccxr' / f'metadata{i}.csv' for i in range(100)]
    for target in targets:
        for path in (target, target.with_suffix(target.suffix + '.part')):
            if path.exists():
                total += path.stat().st_size
    if total + temporary_extra >= 90_000_000_000:
        raise RuntimeError('SSD 90-percent usage guard reached or planned allocation exceeds it')
    return total


def fetch(url, destination):
    if destination.exists():
        return
    part = destination.with_suffix(destination.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        resume = offset > 0 and response.status == 206
        if resume:
            content_range = response.headers.get('Content-Range', '')
            if not content_range.startswith(f'bytes {offset}-'):
                raise RuntimeError('Unexpected download resume offset')
        with part.open('ab' if resume else 'wb') as target:
            shutil.copyfileobj(response, target, CHUNK)
    part.replace(destination)


def copy_checked(source, destination, expected_bytes=None):
    part = destination.with_suffix(destination.suffix + '.part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with part.open('wb') as target:
        while block := source.read(CHUNK):
            target.write(block)
            digest.update(block)
            size += len(block)
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError('Source size differs from published uncompressed file size')
    part.replace(destination)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def extract_iv(source, relative, expected_bytes):
    target = P / 'raw' / 'mimiciv' / relative
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            candidates = [name for name in archive.namelist() if name == 'mimic-iv-2.1/' + relative or name == relative or name == Path(relative).name]
            if len(candidates) != 1:
                raise RuntimeError('Cannot uniquely identify required IV table inside single-file archive')
            with archive.open(candidates[0]) as stream:
                record = copy_checked(stream, target, expected_bytes)
        return record
    with source.open('rb') as stream:
        magic = stream.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(source, 'rb') as stream:
        return copy_checked(stream, target, expected_bytes)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Download must run in a Slurm CPU allocation')
    archive_dir = A / 'zips'
    archive_dir.mkdir(exist_ok=True)
    report_dir = P / 'results' / 'acquisition'
    report_dir.mkdir(parents=True, exist_ok=True)
    listing = public_json(f'https://www.kaggle.com/api/v1/datasets/list/{IV}?pageSize=100')
    indexed = {file['name']: file for file in listing['datasetFiles']}
    required = {name: indexed['mimic-iv-2.1/' + name] for name in NEEDED}
    cxr_metadata = public_json(f'https://www.kaggle.com/api/v1/datasets/view/{CXR}')
    iv_bytes = sum(int(file['totalBytes']) for file in required.values())
    cxr_bytes = int(cxr_metadata['totalBytes'])
    # Conservative bound reserves ALL CXR uncompressed bytes, although only CSV
    # metadata will be extracted initially, plus 8 GB arrays/results/scratch.
    occupied = usage_guard()
    if occupied + iv_bytes + cxr_bytes + 8_000_000_000 >= 90_000_000_000:
        raise RuntimeError('Conservative acquisition plan exceeds 90-percent SSD guard')
    # Originals, their temporary ZIP conversion, and CXR archive fit under 90%
    # of the confirmed 200 GB HDD assignment using uncompressed upper bounds.
    if 2 * iv_bytes + 2 * cxr_bytes >= 180_000_000_000:
        raise RuntimeError('Conservative archive plan exceeds HDD guard')
    write_json(report_dir / 'plan.json', {'iv_selected_files': len(required),
               'iv_uncompressed_bytes': iv_bytes, 'cxr_uncompressed_bytes': cxr_bytes,
               'ssd_occupied_bytes_before': occupied, 'ssd_assigned_bytes': 100_000_000_000,
               'hdd_assigned_bytes': 200_000_000_000})
    print('ACQUISITION_PLAN_PASSED', flush=True)
    for relative, metadata in required.items():
        stem = relative.replace('/', '__')
        marker = report_dir / (stem + '.json')
        raw = P / 'raw' / 'mimiciv' / relative
        if marker.exists() and raw.exists() and raw.stat().st_size == int(metadata['totalBytes']):
            continue
        usage_guard(int(metadata['totalBytes']))
        original = archive_dir / (stem + '.source')
        backup = archive_dir / (stem + '.zip')
        if not backup.exists():
            url = f'https://www.kaggle.com/api/v1/datasets/download/{IV}?fileName=' + urllib.parse.quote('mimic-iv-2.1/' + relative, safe='')
            fetch(url, original)
        record = extract_iv(backup if backup.exists() else original, relative, int(metadata['totalBytes']))
        if not backup.exists():
            temporary = backup.with_suffix('.zip.part')
            if zipfile.is_zipfile(original):
                original.replace(backup)
            else:
                with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                    # Store the verified extracted original CSV under its exact
                    # source-relative name, independent of transport compression.
                    archive.write(raw, arcname='mimic-iv-2.1/' + relative)
                temporary.replace(backup)
        write_json(marker, record)
        if original.exists():
            original.unlink()  # Fixed path under own HDD zips; verified ZIP exists.
        print('IV_TABLE_VERIFIED', relative, record['bytes'], flush=True)
    cxr_archive = archive_dir / 'mimiccxr.zip'
    fetch(f'https://www.kaggle.com/api/v1/datasets/download/{CXR}', cxr_archive)
    if not zipfile.is_zipfile(cxr_archive):
        raise RuntimeError('CXR download is not a ZIP archive')
    metadata_records = []
    with zipfile.ZipFile(cxr_archive) as archive:
        csvs = [item for item in archive.infolist() if item.filename.lower().endswith('.csv')]
        if not csvs or len(csvs) > 100:
            raise RuntimeError('Unexpected number of CXR metadata CSVs')
        for index, item in enumerate(csvs):
            # Locally assigned names prevent archive-path traversal and avoid
            # patient identifiers leaking into exported file names.
            target = P / 'raw' / 'mimiccxr' / f'metadata{index}.csv'
            usage_guard(item.file_size)
            with archive.open(item) as stream:
                record = copy_checked(stream, target, item.file_size)
            record['archive_member'] = item.filename
            metadata_records.append(record)
    write_json(report_dir / 'cxrmetadata.json', metadata_records)
    print('CXR_METADATA_VERIFIED', len(metadata_records), flush=True)
    usage_guard()
    print('ACQUISITION_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not print exception messages that may contain signed URLs or rows.
        reason = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        (P / 'reports' / 'BLOCKED.md').write_text(
            f'Acquisition stopped: {reason}. No automatic retries.\n')
        print('ACQUISITION_STOPPED', reason, flush=True)
        raise SystemExit(1)
