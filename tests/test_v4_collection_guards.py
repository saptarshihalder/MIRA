import io
import json
import zipfile
import pytest
from infra.collect_lifted_v4_results import validate_members

def archive(extra=None, failed=None, confirmation=True):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('manifest.json', json.dumps({'failed': failed or []}))
        z.writestr('run_identity.json', json.dumps({'metadata': {
            'commit': 'e1b1d9d09e98667dc4590e6f54a6c6641c3e24a9'}}))
        if confirmation:
            z.writestr('runs/confirm_v4.json', '{}')
        if extra:
            z.writestr(extra, '{}')
    buffer.seek(0)
    return zipfile.ZipFile(buffer)

def test_accepts_minimal_complete_transport():
    with archive('runs/model/train.json') as z:
        assert len(validate_members(z)) == 4

@pytest.mark.parametrize('name', ['../outside.json', '/outside.json',
    'C:/outside.json', 'runs\\..\\outside.json', 'runs/model.pt'])
def test_rejects_unsafe_or_unexpected_members(name):
    with archive(name) as z, pytest.raises(ValueError):
        validate_members(z)

def test_rejects_recorded_failures():
    with archive(failed=['fine_tune']) as z, pytest.raises(ValueError):
        validate_members(z)

def test_rejects_incomplete_confirmation():
    with archive(confirmation=False) as z, pytest.raises(ValueError):
        validate_members(z)
