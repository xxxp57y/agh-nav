"""Compile Python sources and reject private deployment artifacts without executing them."""
from pathlib import Path
import ast
import re
import sys

root = Path(__file__).resolve().parents[1]
paths = sys.argv[1:] or [p.relative_to(root).as_posix() for p in root.rglob('*')
    if p.is_file() and not any(x in {'.git', '.venv', 'node_modules', '__pycache__'} for x in p.relative_to(root).parts)]
errors = []
private_markers = ['omni' + '-yezh', '121.196.' + '148.13', 'F:' + '/AGH', 'C:/Users/' + 'admin']
credential = re.compile(r'(?:sk|ghp|github_pat)[-_][A-Za-z0-9_]{20,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----')
checked = 0
for name in paths:
    path = root / name
    if not path.is_file():
        continue
    if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
        errors.append(f'{name}: binary assets require a separate publication review')
        continue
    text = path.read_text(encoding='utf-8-sig')
    if any(s in text for s in private_markers) or credential.search(text):
        errors.append(f'{name}: private deployment or credential marker')
    if path.name in {'.llm-key', '.env'} or 'workspaces' in path.parts:
        errors.append(f'{name}: local runtime artifact')
    if path.suffix == '.py':
        try:
            ast.parse(text, filename=name)
        except SyntaxError as exc:
            errors.append(f'{name}: {exc}')
    checked += 1
if errors:
    print('\n'.join(errors))
    raise SystemExit(1)
print(f'Source and publication checks passed: {checked} files')
