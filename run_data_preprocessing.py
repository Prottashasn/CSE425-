#!/usr/bin/env python
"""Script to execute Jupyter notebook code cells (without requiring Jupyter).

Notes:
- Filters out IPython magics like `%matplotlib inline` and shell escapes like `!pip ...`
- Forces UTF-8 stdout/stderr so Unicode (e.g. ✅) doesn't crash on Windows codepages
- Forces a non-interactive matplotlib backend so `plt.show()` doesn't block
"""

import json
import sys
import subprocess

def execute_notebook(notebook_path):
    """Execute all code cells from a Jupyter notebook"""
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract and execute code cells
    code_cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            lines = cell.get('source', [])
            if isinstance(lines, str):
                lines = [lines]

            filtered_lines = []
            for line in lines:
                stripped = line.lstrip()
                # Skip IPython magics and shell escapes that aren't valid in plain Python
                if stripped.startswith('!') or stripped.startswith('%'):
                    continue
                filtered_lines.append(line)

            source = ''.join(filtered_lines)
            if source.strip():
                code_cells.append(source)
    
    # Combine all code cells into a single script
    prelude = (
        "import os\n"
        "import sys\n"
        "# Force UTF-8 output on Windows consoles that default to legacy codepages\n"
        "try:\n"
        "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
        "    sys.stderr.reconfigure(encoding='utf-8', errors='replace')\n"
        "except Exception:\n"
        "    pass\n"
        "# Ensure matplotlib is non-interactive (prevents GUI popups / blocking)\n"
        "os.environ.setdefault('MPLBACKEND', 'Agg')\n"
    )
    full_script = prelude + '\n\n' + '\n\n'.join(code_cells)
    
    # Write to temporary Python file
    temp_script = notebook_path.replace('.ipynb', '_temp.py')
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(full_script)
    
    print(f"Executing notebook: {notebook_path}")
    print("=" * 60)
    
    # Execute the script
    try:
        import os
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("MPLBACKEND", "Agg")

        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=False,
            text=True,
            cwd='E:\\425 Project',
            env=env,
        )
        return result.returncode
    finally:
        # Clean up temporary file
        import os
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'data_preprocessing.ipynb'
    exit_code = execute_notebook(notebook_path)
    sys.exit(exit_code)

