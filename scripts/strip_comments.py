                      
"""Strip comments from Python and GitHub Actions YAML workflow files.

This script removes single-line and inline comments from Python (.py) files
while preserving docstrings (module, class, function). It also removes
standalone triple-quoted string expression statements (often used as
multiline comments) except for docstrings.

It also strips comments from YAML files under .github/workflows by removing
text after a '#' that is not inside quotes.

Backups are created with a .bak extension next to each modified file.

Usage: python scripts/strip_comments.py

Be cautious: this edits files in-place. Review backups if necessary.
"""
import ast
import io
import os
import sys
import shutil
import tokenize
from pathlib import Path


def find_python_files(root: Path):
    for p in root.rglob('*.py'):
                                                      
        if any(part in ('__pycache__', '.git', 'modeling/models', 'node_modules') for part in p.parts):
            continue
        yield p


def find_workflow_files(root: Path):
    wf_dir = root / '.github' / 'workflows'
    if not wf_dir.exists():
        return []
    return list(wf_dir.glob('*.yml')) + list(wf_dir.glob('*.yaml'))


def get_docstring_spans(source: str):
    """Return set of (start_line, end_line) ranges that correspond to docstrings."""
    spans = []
    try:
        tree = ast.parse(source)
    except Exception:
        return spans

                      
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], 'value', None), ast.Constant) and isinstance(tree.body[0].value.value, str):
        node = tree.body[0]
        spans.append((node.lineno, getattr(node, 'end_lineno', node.lineno)))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
                ds = node.body[0]
                spans.append((ds.lineno, getattr(ds, 'end_lineno', ds.lineno)))

    return spans


def get_standalone_string_expr_spans(source: str):
    """Return ranges (lineno,end_lineno) for Expr nodes that are standalone strings (not docstrings).

    These are likely multiline comment blocks.
    """
    spans = []
    try:
        tree = ast.parse(source)
    except Exception:
        return spans

    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(getattr(node, 'value', None), ast.Constant) and isinstance(node.value.value, str):
            spans.append((node.lineno, getattr(node, 'end_lineno', node.lineno)))

    return spans


def remove_comments_from_python(path: Path):
    text = path.read_text(encoding='utf8')
    doc_spans = set(get_docstring_spans(text))
    expr_spans = set(get_standalone_string_expr_spans(text))

                                                                    
    removable_spans = [span for span in expr_spans if span not in doc_spans]

                                                                
    lines_to_remove = set()
    for s, e in removable_spans:
        for l in range(s, e + 1):
            lines_to_remove.add(l)

    backup = path.with_suffix(path.suffix + '.bak')
    if not backup.exists():
        shutil.copy2(path, backup)

    out_tokens = []
    try:
        with path.open('rb') as f:
            tokgen = tokenize.tokenize(f.readline)
            for tok in tokgen:
                ttype = tok.type
                tstring = tok.string
                srow, scol = tok.start
                erow, ecol = tok.end

                               
                if ttype == tokenize.COMMENT:
                    continue

                                                                                        
                if ttype == tokenize.STRING:
                    if all(row in lines_to_remove for row in range(srow, erow + 1)):
                        continue

                out_tokens.append(tok)

    except Exception as e:
        print(f"Failed to tokenize {path}: {e}")
        return False

                               
    new_source = tokenize.untokenize(out_tokens).decode('utf8')
    path.write_text(new_source, encoding='utf8')
    return True


def strip_yaml_comments(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        s = line
                                                   
        new_line = ''
        i = 0
        in_sq = False
        in_dq = False
        while i < len(s):
            ch = s[i]
            if ch == "'" and not in_dq:
                in_sq = not in_sq
                new_line += ch
                i += 1
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
                new_line += ch
                i += 1
                continue
            if ch == '#' and not in_sq and not in_dq:
                                       
                break
            new_line += ch
            i += 1
        out_lines.append(new_line.rstrip())
    return '\n'.join(out_lines) + ('\n' if text.endswith('\n') else '')


def remove_comments_from_yaml(path: Path):
    text = path.read_text(encoding='utf8')
    backup = path.with_suffix(path.suffix + '.bak')
    if not backup.exists():
        shutil.copy2(path, backup)
    new_text = strip_yaml_comments(text)
    path.write_text(new_text, encoding='utf8')
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Apply changes; default is dry-run')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py_files = list(find_python_files(root))
    wf_files = find_workflow_files(root)

    print(f"Found {len(py_files)} Python files, {len(wf_files)} workflow files")

    if not args.apply:
        print('\nDry-run mode (no files will be changed). Use --apply to perform edits.')
        for p in py_files:
            print(f"Would process Python: {p}")
        for p in wf_files:
            print(f"Would process workflow: {p}")
        return

    failed = []
    for p in py_files:
        print(f"Processing Python: {p}")
        ok = remove_comments_from_python(p)
        if not ok:
            failed.append(str(p))

    for p in wf_files:
        print(f"Processing workflow: {p}")
        ok = remove_comments_from_yaml(p)
        if not ok:
            failed.append(str(p))

    if failed:
        print("Some files failed:")
        for f in failed:
            print(" -", f)
        sys.exit(2)
    print("Done. Backups saved with .bak extensions.")


if __name__ == '__main__':
    main()
