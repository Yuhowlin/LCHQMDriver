"""Readout-frequency estimate adapter: adapts the raw dataset and delegates to scqat.

Fitting and figure generation are split (`fit` with skip_figures, then
`figures`) so the shell can keep qualibrate's analyse-now/plot-later flow and
the fit runs only once.
"""

from typing import Dict, Tuple

import xarray as xr


def fit(ds_raw: xr.Dataset) -> Tuple[Dict, Dict, object]:
    """Scan each qubit's swept-frequency single-shot data with scqat's
    ReadoutFreqFidelityEstimator (figures skipped here).

    `ds_raw` already carries the I/Q vars and shot_idx/frequency/prepared_state
    coords that scqat expects (see `scqo_qm/probes/readout_frequency`), so no
    renaming is needed.

    Returns (fit_results, sep_results, estimator):
      * fit_results[qubit] = {"best_detuning", "best_fidelity", "success"} -
        small and JSON-friendly, what the shell persists and `update` consumes;
      * sep_results[qubit] = (per-qubit dataset, full estimator results) - kept
        for deferred figure drawing;
      * estimator - the instance to pass back to `figures`.
    """
    from scqat.parsers import repetition_data
    from scqat.estimators.readout_fidelity import ReadoutFreqFidelityEstimator

    estimator = ReadoutFreqFidelityEstimator()
    fit_results: Dict = {}
    sep_results: Dict = {}

    for sq in repetition_data(ds_raw, repetition_dim="qubit"):
        qubit_name = sq["qubit"].values.item()
        results = estimator.analyze(sq, output_dir=None, skip_figures=True)[0]
        best = results["best_sweep_value"]
        fit_results[qubit_name] = {
            "best_detuning": float(best) if best is not None else float("nan"),
            "best_fidelity": float(results["best_fidelity"]) if results["best_fidelity"] is not None else float("nan"),
            "success": bool(results["success"]),
        }
        sep_results[qubit_name] = (sq, results)

    return fit_results, sep_results, estimator


def figures(estimator, sep_results: Dict) -> Dict:
    """Draw the scqat fidelity figures from the stored (dataset, results) pairs."""
    return {
        qubit_name: estimator.generate_figures(sq, results)
        for qubit_name, (sq, results) in sep_results.items()
    }
