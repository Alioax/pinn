# -*- coding: utf-8 -*-
"""
E0: evaluate the saved Report 4 models (April 2026) on the held-out test of homog_common.py,
so they can be compared with the H01-H12 grid runs. Inference only, no training.

Writes grid_results/E0_pinn_report4_saved/ and grid_results/E0_pino_report4_saved/
(run_meta.json + test_errors.csv).
"""
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import homog_common as H  # noqa: E402

torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAINED = {"n_cfl_samples": 50, "optimizer": "lbfgs", "lbfgs_lr": 1.0, "lbfgs_max_iter": 1,
           "lbfgs_history_size": 50, "steps_requested": 1000, "early_stop_patience": 0,
           "note": "Report 4 model as saved on 14 Apr 2026; training time was not logged"}

MODELS = [
    ("E0_pinn_report4_saved", "PARA-PINN", os.path.join("parametric_cfl_pinn", "results", "parametric_cfl_pinn_model.pt"),
     lambda: H.ParametricPINN([3, 16, 16, 16, 16, 1], nn.Tanh), [3, 16, 16, 16, 16, 1]),
    ("E0_pino_report4_saved", "PINO", os.path.join("pino_cfl", "results", "pino_cfl_model.pt"),
     lambda: H.DeepONetParametric([1, 16, 16, 16], [2, 16, 16, 16], nn.Tanh),
     {"branch": [1, 16, 16, 16], "trunk": [2, 16, 16, 16], "latent_q": 16}),
]

for run_name, kind, rel_path, build, arch in MODELS:
    t0 = time.perf_counter()
    model = build().to(device)
    state = torch.load(os.path.join(H.HOMOG_DIR, rel_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    if kind == "PINO":
        predict = lambda x, t, cfl: model(x, t, torch.full((x.shape[0], 1), cfl, dtype=x.dtype, device=x.device))  # noqa: E731
    else:
        predict = lambda x, t, cfl: model(x, t, torch.full_like(x, cfl))  # noqa: E731
    rows, summary = H.evaluate(predict, device, DTYPE)
    out_dir = os.path.join(H.GRID_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"run_name": run_name, "model": kind, "source_model": rel_path.replace(os.sep, "/"),
            "finished": H.timestamp(), "architecture": arch, "n_params": H.n_params(model),
            "dtype": str(DTYPE), **TRAINED, "train_wall_clock_s": None, "steps_actual": None,
            **summary, "script_wall_clock_s": round(time.perf_counter() - t0, 3),
            "environment": H.environment(device)}
    H.write_outputs(out_dir, meta, rows)
    H.print_summary(run_name, summary)
