#!/usr/bin/env python3
"""Build/export a local FC image from one public commit; never publish or authorize it."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile

COMMIT = 'afc93c6fb94af01239f51b06d5c29a40c6fe7d84'
REMOTE = 'https://github.com/mottopanikeiku/EzPC.git'
BASE = 'sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade'
BINARY = '/opt/ringlpn/source/GPU-MPC/ringlpn/bin/test_two_party_fc_preprocess'
RECIPES = ('Dockerfile.runtime', 'build_source_runtime_candidate.py', 'capture_runtime_build.py')
# Only CUTLASS is a gitlink needed by the canonical FC compilation. SCI/src,
# Sytorch, cryptoTools, LLAMA and bitpack are tracked in the pinned parent tree.
# In particular, NEVER initialize the weights or dataset submodules.
SUBMODULE = 'GPU-MPC/ext/cutlass'
SUBMODULE_URL = 'https://github.com/NVIDIA/cutlass.git'
SOURCE_ROOTS = (
    'SCI/src/', 'GPU-MPC/backend/', 'GPU-MPC/fss/', 'GPU-MPC/utils/',
    'GPU-MPC/ext/sytorch/', 'GPU-MPC/ringlpn/src/',
    'GPU-MPC/ext/cutlass/include/', 'GPU-MPC/ext/cutlass/tools/util/include/',
)
BUILDERS = tuple('GPU-MPC/ringlpn/scripts/' + name for name in (
    'build_component.sh', 'build_common.sh', 'build_two_party_fc_preprocess.sh'))
SOURCE_SUFFIXES = {'.h', '.hpp', '.hh', '.hxx', '.c', '.cc', '.cpp', '.cxx',
                   '.cu', '.cuh', '.inc', '.inl', '.ipp', '.tpp', '.s', '.S'}
EXTENSIONLESS_HEADER_ROOT = 'GPU-MPC/ext/sytorch/ext/cryptoTools/cryptoTools/gsl/'
FORBIDDEN_PARTS = {'results', 'measurements', 'manuscript', 'datasets', 'weights',
                   'evidence', '.git', 'build', 'bin', 'host_bin', '__pycache__'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def fresh_path(value):
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)):
        raise ValueError('roots must be normalized absolute paths')
    if path.exists() or path.is_symlink():
        raise ValueError(f'stale/reused root is forbidden: {path}')
    if not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise ValueError(f'root needs an existing non-symlink parent: {path}')
    return path


def admitted(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or any(p in FORBIDDEN_PARTS for p in path.parts):
        return False
    license_file = re.match(r'^(LICENSE|LICENCE|COPYING|NOTICE|COPYRIGHT)([.-].*)?$', path.name, re.I)
    if license_file:
        # Preserve upstream license/notice files, not arbitrary Markdown/docs.
        return True
    return (name in BUILDERS or
            (name.startswith(SOURCE_ROOTS) and path.suffix in SOURCE_SUFFIXES) or
            (name.startswith(EXTENSIONLESS_HEADER_ROOT) and path.suffix == ''))


class Runner:
    def __init__(self, output, env):
        self.output = output
        self.env = env
        self.commands = []

    def run(self, command, *, cwd=None):
        index = len(self.commands)
        record = {'argv': list(map(str, command)), 'cwd': str(cwd) if cwd else None,
                  'log': f'command-{index:03d}.log'}
        self.commands.append(record)
        with (self.output / record['log']).open('wb') as log:
            process = subprocess.Popen(record['argv'], cwd=cwd, env=self.env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record['returncode'] = process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                json_write(self.output / 'commands.json', self.commands)
        data = (self.output / record['log']).read_bytes()
        if record['returncode']:
            raise RuntimeError(f"command failed ({record['returncode']}), see {record['log']}")
        return data


def export_tree(run, repository, prefix, context, archive, files):
    run.run(['git', '-C', repository, 'archive', '--format=tar', '--output', archive, 'HEAD'])
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            name = prefix + member.name
            if not admitted(name):
                continue
            if not member.isfile():
                if member.isdir():
                    continue
                raise RuntimeError(f'non-regular source admission rejected: {name}')
            destination = context / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target)
            mode = 0o755 if member.mode & 0o111 else 0o644
            destination.chmod(mode)
            os.utime(destination, (1786320000, 1786320000))
            files.append({'path': name, 'mode': oct(mode), 'sha256': digest(destination)})
    archive.unlink()


def interrupted(signum, _frame):
    raise InterruptedError(f'interrupted by signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', required=True, help='fresh external scratch root (retains a tombstone)')
    parser.add_argument('--output-root', required=True, help='fresh external artifact/log root')
    parser.add_argument('--docker-host', default='unix:///var/run/docker.sock',
                        help='local Docker Unix socket only; no remote daemon or Podman socket')
    args = parser.parse_args()
    os.umask(0o077)
    work, output = fresh_path(args.work_root), fresh_path(args.output_root)
    recipe_root = Path(__file__).resolve().parent
    repo_root = recipe_root.parents[2]
    if (work == output or work in output.parents or output in work.parents or
            repo_root == work or repo_root in work.parents or
            repo_root == output or repo_root in output.parents):
        raise ValueError('work/output roots must be disjoint and outside the source repository')
    if not args.docker_host.startswith('unix:///') or 'podman' in args.docker_host.lower():
        raise ValueError('only a local Docker Unix socket is admitted')
    socket = Path(args.docker_host.removeprefix('unix://'))
    if not stat.S_ISSOCK(socket.stat().st_mode):
        raise ValueError('Docker endpoint is not a Unix socket')
    for name in RECIPES:
        path = recipe_root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'packaging recipe must be a regular file: {path}')
    work.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    home = work / 'home'
    home.mkdir()
    docker_config = work / 'docker-config'
    docker_config.mkdir()
    # Do not inherit Git config, SSH agent, askpass, proxies, Docker credentials,
    # compiler flags, registry auth, GPU selectors or client-side plugins.
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(home), 'LANG': 'C.UTF-8',
           'LC_ALL': 'C.UTF-8', 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
           'GIT_TERMINAL_PROMPT': '0', 'GIT_ALLOW_PROTOCOL': 'https',
           'DOCKER_CONFIG': str(docker_config), 'DOCKER_BUILDKIT': '1'}
    run = Runner(output, env)
    docker = ['docker', '--host', args.docker_host]
    container = None
    completed = False
    try:
        docker_version = json.loads(run.run(docker + ['version', '--format', '{{json .}}']))
        if 'podman' in json.dumps(docker_version).lower():
            raise RuntimeError('Podman is not admitted by this Docker candidate recipe')
        git_version = run.run(['git', '--version']).decode().strip()
        checkout = work / 'checkout'
        run.run(['git', 'init', '--template=', checkout])
        run.run(['git', '-C', checkout, '-c', 'credential.helper=', 'fetch', '--depth=1',
                 '--no-tags', REMOTE, COMMIT])
        run.run(['git', '-C', checkout, 'checkout', '--detach', 'FETCH_HEAD'])
        head = run.run(['git', '-C', checkout, 'rev-parse', 'HEAD']).decode().strip()
        if head != COMMIT:
            raise RuntimeError('public source commit mismatch')
        tree = run.run(['git', '-C', checkout, 'rev-parse', 'HEAD^{tree}']).decode().strip()
        gitlink = run.run(['git', '-C', checkout, 'ls-tree', 'HEAD', SUBMODULE]).decode().split()
        if len(gitlink) != 4 or gitlink[:2] != ['160000', 'commit'] or gitlink[3] != SUBMODULE:
            raise RuntimeError('required CUTLASS gitlink is absent')
        pinned_submodule = gitlink[2]
        configured_url = run.run(['git', '-C', checkout, 'config', '-f', '.gitmodules',
                                  '--get', f'submodule.{SUBMODULE}.url']).decode().strip()
        if configured_url != SUBMODULE_URL:
            raise RuntimeError('CUTLASS public source URL mismatch')
        run.run(['git', '-C', checkout, '-c', 'credential.helper=',
                 'submodule', 'update', '--init', '--depth=1', '--', SUBMODULE])
        subroot = checkout / SUBMODULE
        if run.run(['git', '-C', subroot, 'rev-parse', 'HEAD']).decode().strip() != pinned_submodule:
            raise RuntimeError('CUTLASS commit differs from public parent gitlink')
        # A new checkout has no permitted ignored artifacts either.
        for root in (checkout, subroot):
            if run.run(['git', '-C', root, 'status', '--porcelain=v1', '--ignored',
                        '--untracked-files=all']):
                raise RuntimeError('source checkout is not clean')
            visible = run.run(['git', '-C', root, 'ls-files', '-v', '-z'])
            if any(item[:1] != b'H' for item in visible.split(b'\0') if item):
                raise RuntimeError('source checkout has hidden or non-normal index entries')
        context = work / 'context'
        context.mkdir()
        files = []
        export_tree(run, checkout, '', context, work / 'parent.tar', files)
        export_tree(run, subroot, SUBMODULE + '/', context, work / 'submodule.tar', files)
        if len({item['path'] for item in files}) != len(files):
            raise RuntimeError('duplicate source archive path')
        for path in BUILDERS:
            if not (context / 'source' / path).is_file():
                raise RuntimeError(f'missing canonical builder: {path}')
        if not any(PurePosixPath(item['path']).parent == PurePosixPath('.') and
                   re.match(r'LICENSE|LICENCE|COPYING', PurePosixPath(item['path']).name, re.I)
                   for item in files):
            raise RuntimeError('parent source license is missing')
        recipes = []
        for name in RECIPES:
            shutil.copyfile(recipe_root / name, context / name)
            recipes.append({'path': name, 'sha256': digest(context / name)})
        admission = {
            'schema': 'ringlpn-source-runtime-admission-v1', 'remote': REMOTE,
            'commit': COMMIT, 'tree': tree,
            'submodules': [{'path': SUBMODULE, 'remote': SUBMODULE_URL, 'commit': pinned_submodule}],
            'submodule_scope': 'exact required FC gitlink; data and unused source gitlinks not initialized',
            'files': sorted(files, key=lambda item: item['path']), 'recipes': recipes,
            'source_policy': 'tracked regular source plus upstream licenses; no Git metadata, data, evidence or approvals',
            'base_image': 'nvidia/cuda@' + BASE, 'ubuntu_snapshot': '20260810T000000Z',
        }
        json_write(context / 'source-admission.json', admission)
        shutil.copyfile(context / 'source-admission.json', output / 'source-admission.json')
        # The context is constructed exclusively from Git objects and the three
        # explicitly identified packaging files. No caller-provided context/source.
        run.run(docker + ['build', '--pull', '--no-cache', '--platform', 'linux/amd64',
                          '--iidfile', str(work / 'image.id'), '-f', str(context / 'Dockerfile.runtime'),
                          str(context)])
        image_id = (work / 'image.id').read_text().strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
            raise RuntimeError('Docker did not return an immutable local image ID')
        inspect = json.loads(run.run(docker + ['image', 'inspect', image_id]))
        if len(inspect) != 1 or inspect[0]['Id'] != image_id:
            raise RuntimeError('local image identity mismatch')
        # Container creation/copy does not execute the runtime or grant GPU access.
        container = run.run(docker + ['create', '--network', 'none', image_id]).decode().strip()
        if not re.fullmatch(r'[0-9a-f]{64}', container):
            raise RuntimeError('unexpected container identity')
        run.run(docker + ['cp', container + ':/opt/ringlpn/provenance', str(output / 'provenance')])
        run.run(docker + ['cp', container + ':' + BINARY, str(output / 'test_two_party_fc_preprocess')])
        build = json.loads((output / 'provenance/build.json').read_text())
        binary_sha = digest(output / 'test_two_party_fc_preprocess')
        if (build['binary'] != {'path': BINARY, 'sha256': binary_sha} or
                build['source_admission_sha256'] != digest(output / 'source-admission.json')):
            raise RuntimeError('in-image build/source/binary identity mismatch')
        archive = output / 'runtime-candidate.docker.tar'
        run.run(docker + ['image', 'save', '--output', str(archive), image_id])
        with tarfile.open(archive, 'r:') as saved:
            manifest = json.load(saved.extractfile('manifest.json'))
            if len(manifest) != 1:
                raise RuntimeError('portable artifact must contain exactly one image')
            config = saved.extractfile(manifest[0]['Config']).read()
            if 'sha256:' + hashlib.sha256(config).hexdigest() != image_id:
                raise RuntimeError('exported config digest does not equal local image ID')
            (output / 'image-config.json').write_bytes(config)
        json_write(output / 'candidate.json', {
            'schema': 'ringlpn-source-runtime-candidate-v1',
            'classification': 'local-image-candidate-not-authorized-registry-runtime',
            'source_commit': COMMIT, 'source_admission_sha256': digest(output / 'source-admission.json'),
            'build_provenance_sha256': digest(output / 'provenance/build.json'),
            'base_image': 'nvidia/cuda@' + BASE, 'ubuntu_snapshot': '20260810T000000Z',
            'binary': {'path': BINARY, 'sha256': binary_sha},
            'image': {'local_id': image_id, 'config_sha256': digest(output / 'image-config.json'),
                      'platform': 'linux/amd64', 'rootfs': inspect[0]['RootFS'],
                      'registry_manifest_digest': None, 'authorized_reference': None},
            'export': {'path': archive.name, 'format': 'docker-save',
                       'sha256': digest(archive), 'bytes': archive.stat().st_size},
            'host_tools': {'git': git_version, 'docker': docker_version},
            'gpu_execution_performed': False, 'runtime_planner_executed': False,
            'publication_authorized': False, 'security_claim': 'none',
        })
        completed = True
        print(output / 'candidate.json')
    finally:
        # Preserve status/logs and the non-reusable roots, never caller data.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        cleanup_error = None
        if container and re.fullmatch(r'[0-9a-f]{64}', container):
            try:
                run.run(docker + ['rm', '-f', container])
            except Exception as error:
                cleanup_error = str(error)
        for path in work.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        if not completed or cleanup_error:
            (output / 'candidate.json').unlink(missing_ok=True)
            (output / 'runtime-candidate.docker.tar').unlink(missing_ok=True)
        json_write(work / 'consumed.json', {'completed': completed and cleanup_error is None,
                                           'cleanup_error': cleanup_error,
                                           'output_root': str(output)})
        if cleanup_error:
            raise RuntimeError('temporary container cleanup failed: ' + cleanup_error)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'source runtime candidate failed: {error}', file=sys.stderr)
        sys.exit(1)
