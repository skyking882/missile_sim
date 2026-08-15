"""Build the reader-facing notebook for the H4.5 provisional fit."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "h4_5_provisional_fit.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# H4.5 provisional transonic fit

## tl;dr

The strict Plan 4.5 coverage gate remains failed. An explicitly provisional whole-trajectory replay fit was nevertheless run on the saved labels. With `alpha <= 3 deg`, the best two-free-knot model uses `CdA(0.90)=0.010604 m²`, `CdA(1.05)=0.022313 m²`, and the frozen H4 join `CdA(1.20)=0.023566 m²`. Its independent T4/G3 mean relative speed RMSE is about `0.24%` after excluding the imposed zero-residual initial state.

The stable result is mainly the `M≈1.0–1.2` rise. The `M0.90` value is not stable to the alpha filter and must not be frozen as an identified node."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

This is a model-exploration notebook, not a replacement for the Phase-3 gate report. It compares:

- RM-10 normalized shape with `beta=1`;
- transferred supersonic `beta=1.669`;
- a fitted single `beta`;
- fitted `beta + slope`;
- a no-RM-10 three-knot log-linear control model (`K2`).

T1/T2 are fitted. T3 is a deterministic repeat, while T4 and historical G3 are independent diagnostics. Speed is replayed along observed altitude and flight-path angle. H4 is frozen at and above `M1.2`.

### Key Assumptions

- Visible labels at 2–4 s intervals represent the saved StatShark output correctly.
- Allowing alpha up to 3 deg yields an effective small-alpha drag measure; it is not strict zero-lift drag.
- Observed altitude and flight-path angle may be used as exogenous inputs for an axial-only replay.
- The first replay state is an imposed initial condition and is excluded from RMSE."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import runpy
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = Path.cwd()
SCRIPT_PATH = PROJECT_ROOT / 'scripts' / 'fit_h4_5_provisional.py'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'h4_5_transonic_drag' / 'provisional_fit'
assert SCRIPT_PATH.exists()
assert (PROJECT_ROOT / 'outputs' / 'h4_glide_drag' / 'cda_knots_fit.json').exists()
PROJECT_ROOT"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\nRerun the deterministic fit from the saved T1–T4 and G3 artifacts."),
        nbf.v4.new_code_cell(
            """namespace = runpy.run_path(str(SCRIPT_PATH), run_name='h45_provisional_notebook')
assert namespace['main']() == 0
result = json.loads((OUTPUT_DIR / 'provisional_fit.json').read_text(encoding='utf-8'))

segment_rows = []
for sensitivity_name, sensitivity in result['sensitivity_fits'].items():
    for case_id, segment in sensitivity['segments'].items():
        segment_rows.append({'sensitivity': sensitivity_name, 'case_id': case_id, **segment})
pd.DataFrame(segment_rows)"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### Model comparison at alpha <= 3 deg"),
        nbf.v4.new_code_cell(
            """primary = result['sensitivity_fits']['alpha_3p0']
comparison_rows = []
for model_name in primary['ranking_by_independent_validation']:
    model = primary['models'][model_name]
    comparison_rows.append({
        'model': model_name,
        'parameters': json.dumps(model['parameters'], sort_keys=True),
        'train relative RMSE (%)': 100 * model['training_mean_relative_rmse'],
        'T4/G3 relative RMSE (%)': 100 * model['independent_validation_mean_relative_rmse'],
        'T3 repeat RMSE (%)': 100 * model['repeat_T3_relative_rmse'],
    })
pd.DataFrame(comparison_rows)"""
        ),
        nbf.v4.new_markdown_cell("### Exact curve lookup and alpha-filter sensitivity"),
        nbf.v4.new_code_cell(
            """lookup_mach = {0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20}
curve_table = pd.DataFrame([row for row in primary['curve_rows'] if round(row['mach'], 2) in lookup_mach])
display(curve_table)

alpha_rows = []
for sensitivity_name in ('alpha_2p5', 'alpha_3p0'):
    sensitivity = result['sensitivity_fits'][sensitivity_name]
    k2 = sensitivity['models']['K2_log_knots']['parameters']
    p1 = sensitivity['models']['P1_beta_free']['parameters']
    alpha_rows.append({
        'sensitivity': sensitivity_name,
        'supported Mach min (T1/T2)': min(sensitivity['segments']['T1']['mach_min'], sensitivity['segments']['T2']['mach_min']),
        'K2 CdA M0.90': k2['cda_m09_m2'],
        'K2 CdA M1.05': k2['cda_m105_m2'],
        'P1 beta': p1['beta'],
    })
pd.DataFrame(alpha_rows)"""
        ),
        nbf.v4.new_markdown_cell("### Curve and whole-trajectory residuals"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(OUTPUT_DIR / 'h4_5_provisional_fit.png')))"
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. The data support a sharp reduction in effective CdA below roughly `M1.0–1.05`; the fitted center node near `M1.05` is comparatively stable across the 2.5 deg and 3 deg filters.
2. The `M0.90` endpoint is not stable. The stricter dataset only reaches about `M0.99`, so its M0.90 knot is an extrapolated fit parameter.
3. The RM-10-derived single-beta model is directionally useful, but beta changes materially with the alpha filter. It is not ready to freeze.
4. K2 improves independent replay error, but sparse labels, short segments, and alpha coupling keep the result provisional.
5. No H4 node at or above `M1.2` was modified."""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
