import json
import re

# Read the notebook
notebook_path = r"Chapter 18 – Reinforcement Learning.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Track changes
changes = 0

# Fix all cells
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source_str = ''.join(source)
        else:
            source_str = source
        
        original = source_str
        
        # Fix env.seed() patterns
        # Pattern 1: env.seed(X)\nobs = env.reset()
        source_str = re.sub(
            r'env\.seed\((\d+)\)\s*\n\s*obs\s*=\s*env\.reset\(\)',
            r'obs, info = env.reset(seed=\1)',
            source_str
        )
        
        # Pattern 2: env.seed(X)\nstate = env.reset()
        source_str = re.sub(
            r'env\.seed\((\d+)\)\s*\n\s*state\s*=\s*env\.reset\(\)',
            r'state, info = env.reset(seed=\1)',
            source_str
        )
        
        # Pattern 3: standalone env.seed(X)
        source_str = re.sub(
            r'env\.seed\((\w+)\)\s*\n',
            r'# env.seed(\1) - removed (deprecated)\n',
            source_str
        )
        
        # Pattern 4: env.seed(X);
        source_str = re.sub(
            r'env\.seed\((\w+)\);',
            r'# env.seed(\1); - removed (deprecated)',
            source_str
        )
        
        # Fix render(mode="rgb_array") to render()
        source_str = re.sub(
            r'\.render\(mode="rgb_array"\)',
            r'.render()',
            source_str
        )
        
        # Fix env.make without render_mode for CartPole
        source_str = re.sub(
            r'gym\.make\("CartPole-v1"\)(?!\s*,)',
            r'gym.make("CartPole-v1", render_mode="rgb_array")',
            source_str
        )
        
        # Update source if changed
        if source_str != original:
            changes += 1
            if isinstance(source, list):
                cell['source'] = [source_str]
            else:
                cell['source'] = source_str

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Fixed {changes} cells")
