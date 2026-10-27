import concurrent.futures, hashlib, pathlib, subprocess, tomllib
root = pathlib.Path(__file__).resolve().parents[2]
lock = tomllib.loads((root / 'uv.lock').read_text())
wheelhouse = pathlib.Path(__file__).parent / 'wheels'
wheelhouse.mkdir(exist_ok=True)
selected = []
for package in lock['package']:
    candidates = [w for w in package.get('wheels', []) if w['url'].endswith(('cp312-cp312-win_amd64.whl', 'cp312-abi3-win_amd64.whl', 'py3-none-any.whl', 'py2.py3-none-any.whl', 'py3-none-win_amd64.whl'))]
    if candidates:
        selected.append(candidates[0])
def fetch(wheel):
    target = wheelhouse / wheel['url'].rsplit('/', 1)[-1]
    expected = wheel['hash'].split(':')[1]
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        subprocess.run(['curl.exe', '-4', '-fsSL', '--connect-timeout', '15', '--continue-at', '-', '--max-time', '1800', '--retry', '2', '-o', str(target), wheel['url'].replace('https://files.pythonhosted.org/', 'https://mirrors.aliyun.com/pypi/')], check=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, target
    print('Verified ' + target.name, flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, selected))
print(f'Verified {len(selected)} locked wheels', flush=True)
