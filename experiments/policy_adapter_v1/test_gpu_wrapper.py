"""Local mocked worker checks: never import Modal, call a remote or allocate GPU."""
import ast
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile


def worker():
    source = Path(__file__).resolve().parents[2]/'infra/modal_policy_gpu.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'worker_factory')
    isolated = ast.Module(body=[factory], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(isolated), str(source), 'exec'), namespace)
    return namespace['worker_factory']()


def check_case(exit_code=0, timeout=False, partial=False):
    with tempfile.TemporaryDirectory(prefix='mira_gpu_wrapper_check_') as directory:
        workspace = Path(directory).resolve()
        assert Path(tempfile.gettempdir()).resolve() in workspace.parents

        def fake_run(command, **kwargs):
            output = Path(command[command.index('--out')+1])
            assert output == workspace/'results'
            assert not output.exists()
            assert (workspace/'runner.log').exists()
            assert kwargs['timeout'] == 780
            assert kwargs['timeout'] < 900
            assert command[1].replace('\\', '/').endswith('/experiments/policy_adapter_v1/gpu_native.py')
            kwargs['stdout'].write('Mocked subprocess output; no remote execution.\n')
            if partial or exit_code == 0:
                output.mkdir()
                (output/'partial.json').write_text(json.dumps({'mocked':True}))
            if timeout:
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            return SimpleNamespace(returncode=exit_code)

        with patch('tempfile.mkdtemp', return_value=str(workspace)), patch('subprocess.run', side_effect=fake_run) as runner:
            result = worker()()
        assert runner.call_count == 1
        expected_code = 124 if timeout else exit_code
        assert result['exit_code'] == expected_code
        assert result['seconds'] >= 0
        with zipfile.ZipFile(io.BytesIO(result['archive'])) as archive:
            names = set(archive.namelist())
            assert {'runner.log', 'wrapper_runtime.json'} <= names
            assert all(not name.startswith(('/', '../')) for name in names)
            runtime = json.loads(archive.read('wrapper_runtime.json'))
            assert runtime['exit_code'] == expected_code
            assert runtime['gpu'] == 'T4'
            # Returned duration includes ZIP creation after the runtime file is saved.
            assert 0 <= runtime['seconds'] <= result['seconds']
            assert ('results/partial.json' in names) == (partial or exit_code == 0)
            assert b'Mocked subprocess output' in archive.read('runner.log')


def test_success_preserves_child_artifact_and_parent_log():
    check_case(exit_code=0)


def test_startup_failure_archives_log_without_precreated_results():
    check_case(exit_code=3)


def test_partial_failure_preserves_partial_artifacts():
    check_case(exit_code=4, partial=True)


def test_timeout_is_bounded_counted_once_and_archived():
    check_case(timeout=True, partial=True)


if __name__ == '__main__':
    tests = sorted((name, value) for name, value in list(globals().items()) if name.startswith('test_'))
    for name, test in tests:
        test()
        print(name+' passed')
    print(str(len(tests))+' mocked GPU worker checks passed; no Modal import or remote call')
