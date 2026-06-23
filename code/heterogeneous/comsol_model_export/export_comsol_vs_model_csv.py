# -*- coding: utf-8 -*-
"""
Export pointwise COMSOL vs model C* predictions to CSV for web plotting.

Writes under ``comsol_model_export/results/<export_name>/`` by default:
  - comsol_vs_model_pointwise.csv  (long: one row per case × time × x*)
  - comsol_vs_model_summary.csv      (one row per case: mean rel-L2)

Usage (from repo root or this directory):
  python code/heterogeneous/comsol_model_export/export_comsol_vs_model_csv.py \\
    --model-family pino \\
    --experiment-dir code/heterogeneous/pino_heterogeneous/results/exp_J_maximin_N500_float64_lr01_maxiter20

  python code/heterogeneous/comsol_model_export/export_comsol_vs_model_csv.py \\
    --model-family parametric_pinn \\
    --checkpoint code/heterogeneous/parametric_heterogeneous_pinn/results/.../parametric_heterogeneous_pinn_model.pt \\
    --run-meta code/heterogeneous/parametric_heterogeneous_pinn/results/.../run_meta.json

  # Export every trained checkpoint (all 81 COMSOL media cases each):
  python code/heterogeneous/comsol_model_export/export_comsol_vs_model_csv.py --export-all
  python code/heterogeneous/comsol_model_export/export_comsol_vs_model_csv.py --export-all --dry-run
  python code/heterogeneous/comsol_model_export/export_comsol_vs_model_csv.py --export-all --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm

_HETERO_ROOT = Path(__file__).resolve().parent.parent
_EXPORT_ROOT = Path(__file__).resolve().parent
_DEFAULT_COMSOL = _HETERO_ROOT / "data" / "comsol_4zones.txt"
_DEFAULT_RESULTS = _EXPORT_ROOT / "results"
_PINO_RESULTS = _HETERO_ROOT / "pino_heterogeneous" / "results"
_PINN_RESULTS = _HETERO_ROOT / "parametric_heterogeneous_pinn" / "results"

ModelFamily = Literal["pino", "parametric_pinn"]

POINTWISE_FIELDS = [
    "case_id",
    "u1",
    "u2",
    "u3",
    "u4",
    "t_days",
    "t_star",
    "x_star",
    "c_comsol_star",
    "c_model_star",
    "abs_err",
    "rel_err",
]
SUMMARY_FIELDS = [
    "case_id",
    "u1",
    "u2",
    "u3",
    "u4",
    "mean_rel_l2",
    "n_points",
]


def _ensure_paths(*paths: Path) -> None:
    for p in paths:
        sys_path = str(p)
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)


def _import_module(script_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe_load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _l2_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref)
    return float(num / den) if den > 0 else float(num)


def _match_comsol_times(
    series: dict[float, np.ndarray], requested: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    out: list[tuple[float, np.ndarray]] = []
    for target in requested:
        hit = None
        for t_key, values in series.items():
            if np.isclose(float(t_key), float(target), rtol=0.0, atol=1e-10):
                hit = (float(t_key), values)
                break
        if hit is None:
            raise KeyError(f"No COMSOL slice at t={target} d.")
        out.append(hit)
    return out


def _is_smoke_dir(path: Path) -> bool:
    return path.name.startswith("_smoke")


def discover_all_experiments() -> list[tuple[ModelFamily, Path]]:
    """Return (model_family, experiment_dir) for every trained checkpoint."""
    found: list[tuple[ModelFamily, Path]] = []

    if _PINO_RESULTS.is_dir():
        for exp_dir in sorted(_PINO_RESULTS.glob("exp_*")):
            if not exp_dir.is_dir() or _is_smoke_dir(exp_dir):
                continue
            ckpt = exp_dir / "pino_heterogeneous_model.pt"
            meta = exp_dir / "run_meta.json"
            if ckpt.is_file() and meta.is_file():
                found.append(("pino", exp_dir))

    if _PINN_RESULTS.is_dir():
        for exp_dir in sorted(_PINN_RESULTS.glob("exp_*")):
            if not exp_dir.is_dir() or _is_smoke_dir(exp_dir):
                continue
            meta = exp_dir / "run_meta.json"
            if not meta.is_file():
                continue
            pinn_ckpt = exp_dir / "parametric_heterogeneous_pinn_model.pt"
            pino_ckpt = exp_dir / "pino_heterogeneous_model.pt"
            if pinn_ckpt.is_file():
                found.append(("parametric_pinn", exp_dir))
            elif pino_ckpt.is_file():
                found.append(("pino", exp_dir))

    return found


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    if args.experiment_dir is not None:
        exp = Path(args.experiment_dir).resolve()
        if args.model_family == "pino":
            ckpt = exp / "pino_heterogeneous_model.pt"
        else:
            ckpt = exp / "parametric_heterogeneous_pinn_model.pt"
        meta = exp / "run_meta.json"
        if not ckpt.is_file():
            pts = sorted(exp.glob("*.pt"))
            if not pts:
                pts = sorted(exp.rglob("*.pt"))

            if not pts:
                raise FileNotFoundError(
                    f"Missing checkpoint .pt in experiment-dir={exp}. "
                    f"Looked for {ckpt} and for any '*.pt' files."
                )

            if args.model_family == "parametric_pinn":
                # Avoid silently loading a PINO checkpoint into the parametric PINN
                # architecture (state_dict keys won’t match).
                preferred = [
                    p
                    for p in pts
                    if "parametric_heterogeneous_pinn_model" in p.name
                    or "parametric_heterogeneous_pinn" in p.name
                ]
                if not preferred:
                    found_names = ", ".join(p.name for p in pts)
                    raise FileNotFoundError(
                        f"Expected a parametric_pinn checkpoint like '{ckpt.name}', but none was found. "
                        f"Found .pt files in {exp}: {found_names}. "
                        f"Either pass --checkpoint pointing to the correct file, "
                        f"or use --model-family pino if that is the checkpoint you have."
                    )
                ckpt = preferred[0]
            else:
                # PINO runs sometimes saved under the expected name already, but
                # keep a fallback for older/variant checkpoints.
                ckpt = pts[0]
        if args.export_name is None:
            export_name = exp.name
        else:
            export_name = args.export_name
    else:
        if args.checkpoint is None or args.run_meta is None:
            raise ValueError(
                "Provide --experiment-dir or both --checkpoint and --run-meta"
            )
        ckpt = Path(args.checkpoint).resolve()
        meta = Path(args.run_meta).resolve()
        export_name = args.export_name or ckpt.parent.name

    comsol = Path(args.comsol_data).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _DEFAULT_RESULTS / export_name
    return ckpt, meta, comsol, out_dir


def _configure_pino_from_meta(mod: Any, meta: dict[str, object]) -> None:
    mod.media_mode = str(meta.get("media_mode", "zone"))
    if mod.media_mode == "grf" and "sensor_xstar" in meta:
        mod.sensor_xstar = np.asarray(meta["sensor_xstar"], dtype=np.float64)
    else:
        mod.sensor_xstar = None


def _pino_architecture_from_meta(
    mod: Any, meta: dict[str, object]
) -> tuple[list[int], list[int]]:
    branch = meta.get("branch_architecture")
    trunk = meta.get("trunk_architecture")
    if branch and trunk:
        return (
            list(branch),  # type: ignore[arg-type]
            list(trunk),  # type: ignore[arg-type]
        )

    media_mode = str(meta.get("media_mode", "zone"))
    arch_preset = str(meta.get("arch_preset", "default"))
    if arch_preset not in mod.ARCH_PRESETS:
        arch_preset = "default"
    preset_branch, preset_trunk = mod.ARCH_PRESETS[arch_preset]
    if media_mode == "grf":
        if "sensor_xstar" in meta:
            sensor_count = len(meta["sensor_xstar"])
        else:
            sensor_count = int(meta.get("n_sensors", preset_branch[0]))
        branch_arch = [sensor_count, *preset_branch[1:]]
    else:
        branch_arch = list(preset_branch)
    return branch_arch, list(preset_trunk)


def _pinn_architecture_from_meta(meta: dict[str, object]) -> list[int]:
    if "pinn_architecture" in meta:
        return list(meta["pinn_architecture"])  # type: ignore[arg-type]
    # Legacy / mis-copied meta from PINO runs: default 4-zone CFL input (6 = x*, t*, 4 CFL)
    return [6, 36, 36, 36, 36, 1]


def _build_model(
    family: ModelFamily,
    meta: dict[str, object],
    *,
    device: torch.device,
) -> torch.nn.Module:
    _ensure_paths(_HETERO_ROOT)
    if family == "pino":
        pino_dir = _HETERO_ROOT / "pino_heterogeneous"
        _ensure_paths(pino_dir)
        mod = _import_module(
            pino_dir / "pinn1d_heterogeneous_parametric_neural_operator.py",
            "pino_train_mod",
        )
        _configure_pino_from_meta(mod, meta)
        branch_arch, trunk_arch = _pino_architecture_from_meta(mod, meta)
        model = mod.build_deeponet(
            branch_arch,
            trunk_arch,
            mod.activation_cls,
        )
        return model.to(device), mod

    pinn_dir = _HETERO_ROOT / "parametric_heterogeneous_pinn"
    _ensure_paths(pinn_dir)
    mod = _import_module(
        pinn_dir / "pinn1d_heterogeneous_parametric_pinn.py",
        "pinn_train_mod",
    )
    arch = _pinn_architecture_from_meta(meta)
    model = mod.build_parametric_pinn(arch, mod.activation_cls)
    return model.to(device), mod


def _select_u_cases(
    all_cases: list[tuple[float, float, float, float]],
    *,
    max_cases: int | None,
    case_ids: list[int] | None,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    if case_ids is not None:
        out: list[tuple[int, tuple[float, float, float, float]]] = []
        for cid in case_ids:
            if cid < 0 or cid >= len(all_cases):
                raise ValueError(f"case_id {cid} out of range [0, {len(all_cases) - 1}]")
            out.append((cid, all_cases[cid]))
        return out
    n = len(all_cases) if max_cases is None else min(max_cases, len(all_cases))
    return [(i, all_cases[i]) for i in range(n)]


def export_csv(
    *,
    family: ModelFamily,
    checkpoint: Path,
    run_meta: Path,
    comsol_data: Path,
    out_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    max_cases: int | None,
    case_ids: list[int] | None,
) -> tuple[Path, Path]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not run_meta.is_file():
        raise FileNotFoundError(f"Missing run_meta: {run_meta}")
    if not comsol_data.is_file():
        raise FileNotFoundError(f"Missing COMSOL data: {comsol_data}")

    meta = json.loads(run_meta.read_text(encoding="utf-8"))
    model, mod = _build_model(family, meta, device=device)
    model.load_state_dict(_safe_load_state(checkpoint, device))
    # Ensure model parameters match the requested forward dtype.
    model = model.to(dtype=dtype)
    model.eval()

    from utils.comsol_4zones import list_parameter_combos, load_case  # noqa: E402

    all_u = list_parameter_combos(comsol_data)
    selected = _select_u_cases(all_u, max_cases=max_cases, case_ids=case_ids)

    x_star = np.asarray(mod.pde_collocation_x_star_1d(), dtype=np.float64)
    times_days = np.asarray(mod.COMSOL_TIMES_DAYS, dtype=np.float64)
    t_max = float(mod.T_MAX)
    n_x = x_star.size

    x_t = torch.tensor(x_star.reshape(-1, 1), dtype=dtype, device=device)
    comsol_on_x = mod.comsol_c_star_on_x_star
    predict_at = mod.predict_at

    out_dir.mkdir(parents=True, exist_ok=True)
    pointwise_path = out_dir / "comsol_vs_model_pointwise.csv"
    summary_path = out_dir / "comsol_vs_model_summary.csv"

    summary_rows: list[dict[str, object]] = []

    with open(pointwise_path, "w", newline="", encoding="utf-8") as f_pw:
        pw_writer = csv.DictWriter(f_pw, fieldnames=POINTWISE_FIELDS)
        pw_writer.writeheader()

        for case_id, u_case in tqdm(selected, desc="Export COMSOL vs model"):
            x_m, comsol_by_time = load_case(comsol_data, u_case)
            u1, u2, u3, u4 = u_case

            l2_sum = 0.0
            n_time = 0
            matched = _match_comsol_times(comsol_by_time, times_days)

            for t_days, c_comsol_raw in matched:
                t_star = float(t_days / t_max)
                t_t = torch.full((n_x, 1), t_star, dtype=dtype, device=device)
                with torch.no_grad():
                    c_model = predict_at(model, x_t, t_t, u_case).detach().cpu().numpy().flatten()
                c_ref = comsol_on_x(x_star, x_m, c_comsol_raw)
                l2_sum += _l2_rel(c_model, c_ref)
                n_time += 1

                abs_err = np.abs(c_model - c_ref)
                rel_err = abs_err / (np.abs(c_ref) + 1e-12)

                for xi, cc, cm, ae, re in zip(
                    x_star, c_ref, c_model, abs_err, rel_err, strict=True
                ):
                    pw_writer.writerow(
                        {
                            "case_id": case_id,
                            "u1": u1,
                            "u2": u2,
                            "u3": u3,
                            "u4": u4,
                            "t_days": t_days,
                            "t_star": t_star,
                            "x_star": float(xi),
                            "c_comsol_star": float(cc),
                            "c_model_star": float(cm),
                            "abs_err": float(ae),
                            "rel_err": float(re),
                        }
                    )

            mean_l2 = l2_sum / max(n_time, 1)
            summary_rows.append(
                {
                    "case_id": case_id,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                    "u4": u4,
                    "mean_rel_l2": mean_l2,
                    "n_points": n_x * n_time,
                }
            )

    with open(summary_path, "w", newline="", encoding="utf-8") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    export_meta = {
        "model_family": family,
        "checkpoint": str(checkpoint),
        "run_meta": str(run_meta),
        "comsol_data": str(comsol_data),
        "n_cases_exported": len(selected),
        "n_cases_total": len(all_u),
        "n_x_star": int(n_x),
        "times_days": times_days.tolist(),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(export_meta, indent=2) + "\n", encoding="utf-8"
    )

    return pointwise_path, summary_path


def export_all_models(
    *,
    comsol_data: Path,
    out_root: Path,
    dtype_override: torch.dtype | None,
    device: torch.device,
    skip_existing: bool,
    dry_run: bool,
) -> Path:
    jobs = discover_all_experiments()
    if not jobs:
        raise FileNotFoundError(
            f"No exportable experiments under {_PINO_RESULTS} or {_PINN_RESULTS}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []

    print(f"Discovered {len(jobs)} experiment(s); exporting all 81 COMSOL media each.")
    failures: list[str] = []

    for family, exp_dir in jobs:
        export_name = exp_dir.name
        out_dir = out_root / export_name
        ckpt = (
            exp_dir / "pino_heterogeneous_model.pt"
            if family == "pino"
            else exp_dir / "parametric_heterogeneous_pinn_model.pt"
        )
        meta_path = exp_dir / "run_meta.json"

        pw_path = out_dir / "comsol_vs_model_pointwise.csv"
        sum_path = out_dir / "comsol_vs_model_summary.csv"
        if skip_existing and pw_path.is_file() and sum_path.is_file():
            print(f"Skip (exists): {export_name}")
            index_rows.append(
                {
                    "export_name": export_name,
                    "model_family": family,
                    "experiment_dir": str(exp_dir),
                    "out_dir": str(out_dir),
                    "skipped": True,
                }
            )
            continue

        print(f"Export {family}: {export_name}")
        if dry_run:
            index_rows.append(
                {
                    "export_name": export_name,
                    "model_family": family,
                    "experiment_dir": str(exp_dir),
                    "dry_run": True,
                }
            )
            continue

        if dtype_override is not None:
            dtype = dtype_override
        else:
            meta_preview = json.loads(meta_path.read_text(encoding="utf-8"))
            dtype_name = str(meta_preview.get("dtype", "float64"))
            dtype = torch.float64 if dtype_name == "float64" else torch.float32

        try:
            pw_path, sum_path = export_csv(
                family=family,
                checkpoint=ckpt,
                run_meta=meta_path,
                comsol_data=comsol_data,
                out_dir=out_dir,
                dtype=dtype,
                device=device,
                max_cases=None,
                case_ids=None,
            )
            print(f"  -> {pw_path.name}, {sum_path.name}")
            index_rows.append(
                {
                    "export_name": export_name,
                    "model_family": family,
                    "experiment_dir": str(exp_dir),
                    "pointwise_csv": str(pw_path),
                    "summary_csv": str(sum_path),
                    "skipped": False,
                }
            )
        except Exception as exc:
            failures.append(f"{export_name}: {exc}")
            print(f"  FAILED: {exc}")

    index_path = out_root / "export_index.json"
    index_path.write_text(
        json.dumps(
            {
                "n_exports": len(index_rows),
                "comsol_data": str(comsol_data),
                "exports": index_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote index {index_path} ({len(index_rows)} exports)")
    if failures:
        print("\nFailures:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export COMSOL vs model predictions to CSV for web plots."
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all trained PINO + parametric PINN checkpoints (81 COMSOL cases each).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --export-all, list experiments only (no CSV writes).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --export-all, skip exports whose CSV pair already exists.",
    )
    parser.add_argument(
        "--model-family",
        choices=["pino", "parametric_pinn"],
        default=None,
        help="pino = pino_heterogeneous DeepONet; parametric_pinn = concatenated MLP PINN.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Folder with checkpoint + run_meta.json (e.g. results/exp_J_...).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model .pt path (if not using --experiment-dir).",
    )
    parser.add_argument(
        "--run-meta",
        type=Path,
        default=None,
        help="run_meta.json path (if not using --experiment-dir).",
    )
    parser.add_argument(
        "--comsol-data",
        type=Path,
        default=_DEFAULT_COMSOL,
        help="COMSOL reference file (default data/comsol_4zones.txt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default {_DEFAULT_RESULTS}/<export_name>).",
    )
    parser.add_argument(
        "--export-name",
        type=str,
        default=None,
        help="Subfolder name under comsol_model_export/results/.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Export only first N of 81 cases (for quick tests).",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        metavar="IDS",
        help="Comma-separated case indices to export, e.g. 0,5,12.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Forward-pass dtype (default: from run_meta, else float64).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device (default cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    comsol = Path(args.comsol_data).resolve()
    dtype_override = (
        torch.float64 if args.dtype == "float64" else torch.float32
        if args.dtype is not None
        else None
    )

    if args.export_all:
        out_root = (
            Path(args.out_dir).resolve()
            if args.out_dir is not None
            else _DEFAULT_RESULTS
        )
        export_all_models(
            comsol_data=comsol,
            out_root=out_root,
            dtype_override=dtype_override,
            device=device,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        return

    if args.model_family is None:
        raise SystemExit("Provide --model-family or use --export-all")

    ckpt, meta_path, comsol, out_dir = _resolve_paths(args)

    if dtype_override is not None:
        dtype = dtype_override
    elif meta_path.is_file():
        meta_preview = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype_name = str(meta_preview.get("dtype", "float64"))
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
    else:
        dtype = torch.float64

    case_ids: list[int] | None = None
    if args.case_ids is not None:
        case_ids = [int(s.strip()) for s in args.case_ids.split(",") if s.strip()]

    pw_path, sum_path = export_csv(
        family=args.model_family,  # type: ignore[arg-type]
        checkpoint=ckpt,
        run_meta=meta_path,
        comsol_data=comsol,
        out_dir=out_dir,
        dtype=dtype,
        device=device,
        max_cases=args.max_cases,
        case_ids=case_ids,
    )
    print(f"Wrote {pw_path}")
    print(f"Wrote {sum_path}")
    print(f"Meta  {out_dir / 'export_meta.json'}")


if __name__ == "__main__":
    main()
