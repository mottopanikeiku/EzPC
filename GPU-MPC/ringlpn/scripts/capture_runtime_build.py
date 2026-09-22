#!/usr/bin/env python3
"""Build only through the public canonical FC wrapper and record its identities."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/opt/ringlpn')
SOURCE = ROOT / 'source'
PROVENANCE = ROOT / 'provenance'
BINARY = SOURCE / 'GPU-MPC/ringlpn/bin/test_two_party_fc_preprocess'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(*command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def main():
    admission = json.loads((PROVENANCE / 'source-admission.json').read_text())
    for item in admission['files']:
        path = SOURCE / item['path']
        if not path.is_file() or path.is_symlink() or digest(path) != item['sha256']:
            raise RuntimeError(f"source admission changed: {item['path']}")
    for item in admission['recipes']:
        if digest(ROOT / 'recipe' / item['path']) != item['sha256']:
            raise RuntimeError(f"recipe admission changed: {item['path']}")
    env = dict(os.environ)
    for name, filename in (
        ('RINGLPN_LINEAR_DEPFILE', 'fc.deps'),
        ('RINGLPN_LINEAR_LINK_MAP', 'fc.link.map'),
        ('RINGLPN_LINEAR_COMMAND_FILE', 'fc.command.nul'),
        ('RINGLPN_LINEAR_ENVIRONMENT_FILE', 'fc.environment.nul'),
    ):
        env[name] = str(PROVENANCE / filename)
    builder = SOURCE / 'GPU-MPC/ringlpn/scripts/build_two_party_fc_preprocess.sh'
    subprocess.run([str(builder)], env=env, check=True)
    libraries = []
    ldd = capture('ldd', str(BINARY))
    if 'not found' in ldd:
        raise RuntimeError(f'unresolved runtime library: {ldd}')
    for line in ldd.splitlines():
        match = re.search(r'(?:=>\s+)?(/\S+)\s+\(', line)
        if match:
            path = Path(match.group(1))
            libraries.append({'path': str(path), 'resolved': str(path.resolve()),
                              'sha256': digest(path)})
    if not libraries:
        raise RuntimeError('ldd returned no dynamic runtime dependencies')
    tools = {}
    for name in ('nvcc', 'g++', 'gcc', 'objcopy', 'ld', 'as', 'python3', 'git', 'cmake'):
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(f'missing build tool: {name}')
        tools[name] = {'path': path, 'resolved': str(Path(path).resolve()),
                       'sha256': digest(path), 'version': capture(path, '--version')}
    (PROVENANCE / 'packages.tsv').write_text(capture(
        'dpkg-query', '-W', '-f=${Package}\t${Version}\t${Architecture}\n'))
    (PROVENANCE / 'fc.ldd.txt').write_text(ldd)
    (PROVENANCE / 'fc.dynamic.txt').write_text(capture('readelf', '-d', str(BINARY)))
    # Verify the compiler did not mutate the admitted source or recipes.
    for item in admission['files']:
        if digest(SOURCE / item['path']) != item['sha256']:
            raise RuntimeError(f"build mutated source: {item['path']}")
    metadata = {
        'schema': 'ringlpn-source-runtime-build-v1',
        'classification': 'local-image-candidate-not-publication-authorization',
        'source_commit': admission['commit'],
        'source_admission_sha256': digest(PROVENANCE / 'source-admission.json'),
        'binary': {'path': str(BINARY), 'sha256': digest(BINARY)},
        'canonical_builder': str(builder), 'tools': tools, 'shared_libraries': libraries,
        'cuda_arch': os.environ['CUDA_ARCH'],
        'ubuntu_snapshot': '20260810T000000Z',
        'package_inventory_sha256': digest(PROVENANCE / 'packages.tsv'),
        'artifacts': {path.name: digest(path) for path in sorted(PROVENANCE.iterdir())
                      if path.is_file()},
        'gpu_execution_performed': False,
        'security_claim': 'none',
    }
    (PROVENANCE / 'build.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
