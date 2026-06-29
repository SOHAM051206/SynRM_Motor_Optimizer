"""
utils/data_parser.py
Parses HyperStudy .data files using explicit mapping headers.
Retains actual physical variable names (T1A, VA, etc.) for AI models and export.
"""

import re
import pandas as pd
from pathlib import Path

def _parse_meta_labels(lines: list[str]) -> dict[str, str]:
    varnames, labels = [], []
    for line in lines:
        if 'hstVarnames' in line:
            varnames = re.findall(r'"([^"]*)"', line.split('=', 1)[1])
        elif 'hstLabels' in line:
            labels = re.findall(r'"([^"]*)"', line.split('=', 1)[1])
    length = min(len(varnames), len(labels))
    return dict(zip(varnames[:length], labels[:length]))

def _parse_custom_mappings(lines: list[str]) -> list[dict]:
    """Reads lines like: # MAP | C1_B01 | var_1, var_2 | r_1, r_2"""
    mappings = []
    for line in lines:
        if line.startswith("# MAP |"):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                mappings.append({
                    "geom_id": parts[1],
                    "inputs": [v.strip() for v in parts[2].split(',')],
                    "outputs": [r.strip() for r in parts[3].split(',')]
                })
    return mappings

def parse_batch_file(filepath: str | Path) -> list[dict]:
    filepath = Path(filepath)
    lines = filepath.read_text(encoding='utf-8', errors='replace').splitlines()

    comment_lines = [l for l in lines if l.startswith('#')]
    label_map = _parse_meta_labels(comment_lines)
    custom_mappings = _parse_custom_mappings(comment_lines)

    if not custom_mappings:
        raise ValueError(f"No '# MAP |' lines found at the top of {filepath.name}. Please add them!")

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('v202'):
            header_idx = i + 1 
            break

    if header_idx is None:
        raise ValueError(f"Could not find column header line in {filepath.name}")

    # Read data
    df = pd.read_csv(filepath, sep='\t', skiprows=header_idx, index_col=False)
    df.columns = [str(c).strip() for c in df.columns]

    results = []
    
    # Process exactly according to your manual map
    for geom_map in custom_mappings:
        geom_id = geom_map["geom_id"]
        inp_cols = geom_map["inputs"]
        out_cols = geom_map["outputs"]

        # Validate columns exist
        missing = [c for c in inp_cols + out_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns {missing} not found in {filepath.name} for {geom_id}")

        # Extract data
        inputs_df = df[inp_cols].copy().apply(pd.to_numeric, errors='coerce').dropna()
        outputs_df = df[out_cols].copy().apply(pd.to_numeric, errors='coerce').dropna()
        
        common_idx = inputs_df.index.intersection(outputs_df.index)
        inputs_df = inputs_df.loc[common_idx].reset_index(drop=True)
        outputs_df = outputs_df.loc[common_idx].reset_index(drop=True)

        in_labels = {c: label_map.get(c, c) for c in inp_cols}
        out_labels = {c: label_map.get(c, c) for c in out_cols}

        # ---> FIX: Assign the actual physical names to the dataframes <---
        inputs_df.columns = [in_labels[c] for c in inp_cols]
        outputs_df.columns = [out_labels[c] for c in out_cols]

        results.append({
            'geometry_id': geom_id,
            'batch_file': filepath.name,
            'inputs': inputs_df,
            'outputs': outputs_df,
            'input_labels': in_labels,
            'output_labels': out_labels,
            'n_runs': len(inputs_df),
            'raw_input_cols': inp_cols,
            'raw_out_cols': out_cols,
        })

    return results

def load_all_data_files(data_dir: str | Path, pattern: str = "*.data") -> list[dict]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .data files found in {data_dir} matching '{pattern}'")

    all_geoms = []
    for f in files:
        print(f"  Parsing: {f.name}")
        geoms = parse_batch_file(f)
        all_geoms.extend(geoms)
        print(f"    → {len(geoms)} geometries mapped perfectly!")

    return all_geoms