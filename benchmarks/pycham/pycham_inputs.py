"""Write PyCHAM inputs for a benchmark case and run PyCHAM headless.

* ``write_eqn``: copy of the tutorial .eqn with selected reactions' rate expressions
  replaced (PyCHAM reads the file; TOMAS-JAX applies its edits in code).
* ``write_xml``: PyCHAM's MCM v3.3.1 species xml + the species the limonene file adds.
* ``scenario_model_var`` / ``write_model_var``: model-variables file from a Scenario.
* ``ensure_pycham_root``: per-case copy of the PyCHAM checkout (PyCHAM rewrites
  rate_coeffs.py, ode_solv.py, pickle.pkl in place, so one checkout = one run at a time);
  ``kelvin_off=True`` makes a copy with PyCHAM's hard-coded surface tension set to 0
  (Kelvin factor 1 for every component, water included; attribution runs only).
* ``run_pycham``: headless driver in the PyCHAM venv with a timeout; raises on failure.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from tomas_jax.box.scenario import Scenario
from tomas_jax.gaschem.config import LIMONENE_EQN

from .config import (COPY_TIMEOUT_S, EXTRA_XML_SPECIES, PYCHAM_INPUTS, PYCHAM_INT_TOL,
                     PYCHAM_MCM_CONSTANTS, PYCHAM_MCM_XML, PYCHAM_OUTPUTS, PYCHAM_PYTHON,
                     PYCHAM_RECORDING_STEP_S, PYCHAM_TEMPLATE, PYCHAM_TIMEOUT_S,
                     PYCHAM_UPDATE_STEP_S, RUN_PYCHAM)

_REACTION_RE = re.compile(r"^(\s*<\s*(?P<label>[^>]+?)\s*>[^:]*:)(?P<rate>.*);\s*$")


def case_input_dir(case: str) -> Path:
    d = PYCHAM_INPUTS / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def case_output_dir(case: str, variant: str = "") -> Path:
    return PYCHAM_OUTPUTS / (case + (f"_{variant}" if variant else ""))


def write_eqn(dst: Path, rate_overrides: dict[str, str], src: Path = LIMONENE_EQN) -> Path:
    """Copy ``src`` replacing the rate expression of each labelled reaction.

    Each label must occur exactly once. Returns ``dst``.
    """
    lines = Path(src).read_text().splitlines(keepends=True)
    found = {label: 0 for label in rate_overrides}
    out = []
    for line in lines:
        m = _REACTION_RE.match(line.rstrip("\n"))
        if m and m.group("label") in rate_overrides:
            label = m.group("label")
            found[label] += 1
            # no inline comment: PyCHAM's parser expects nothing after the ';'
            line = f"{m.group(1)} {rate_overrides[label]} ;\n"
        out.append(line)
    bad = {k: v for k, v in found.items() if v != 1}
    if bad:
        raise ValueError(f"reaction labels not found exactly once: {bad}")
    Path(dst).write_text("".join(out))
    return Path(dst)


_SPECIES_DECL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*[^;]*;\s*$")


def write_merged_eqn(dst: Path, rate_overrides: dict[str, str]) -> Path:
    """The limonene + toluene mechanism as one KPP file for PyCHAM (case T1).

    Written from the two files exactly as :func:`tomas_jax.gaschem.merge.merge_mechanisms`
    merges them in code: the limonene file's species declarations, the toluene
    export's declarations of its 181 new species, one ``RO2`` sum (the union), the
    limonene reactions (with ``rate_overrides`` as in :func:`write_eqn`), then the 505
    toluene reactions the limonene file lacks, verbatim from the export with their labels
    prefixed ``T``. The result is parsed back and checked to equal the in-code merge
    (species order, reaction labels, reactants, products and rate expressions).
    """
    from tomas_jax.box.coupling import toluene_merge
    from tomas_jax.gaschem.config import TOLUENE_EQN
    from tomas_jax.gaschem.kpp_parser import parse_eqn

    merged, report = toluene_merge()
    lim_lines = Path(LIMONENE_EQN).read_text().splitlines()
    tol_lines = Path(TOLUENE_EQN).read_text().splitlines()
    i_inline = next(i for i, l in enumerate(lim_lines) if l.strip().startswith("#INLINE"))
    i_end = next(i for i, l in enumerate(lim_lines) if l.strip().startswith("#ENDINLINE"))
    new_species = set(report.new_species)
    decls = [l for l in tol_lines if (m := _SPECIES_DECL_RE.match(l)) and m.group(1) in new_species]
    if len(decls) != len(new_species):
        raise ValueError(f"found {len(decls)} declarations for {len(new_species)} new species")
    ro2 = ["  RO2 = " + " + ".join(f"C(ind_{s})" for s in merged.ro2[:4]) + " + &"]
    rest = merged.ro2[4:]
    for k in range(0, len(rest), 5):
        chunk = " + ".join(f"C(ind_{s})" for s in rest[k:k + 5])
        ro2.append("      " + chunk + (" + &" if k + 5 < len(rest) else ""))
    new_labels = {lab[1:] for lab in report.new_reactions}          # strip the T prefix
    added = []
    for l in tol_lines:
        m = _REACTION_RE.match(l)
        if m and m.group("label") in new_labels:
            added.append(l.replace(f"<{m.group('label')}>", f"<T{m.group('label')}>", 1))
    if len(added) != len(new_labels):
        raise ValueError(f"found {len(added)} of {len(new_labels)} added reactions in the export")
    text = "\n".join([*lim_lines[:i_inline],
                      "// --- species of the MCM v3.3.1 toluene subset (toluene_mcm.eqn) ---",
                      *decls, "", "#INLINE F90_RCONST", "  USE constants_mcm", *ro2,
                      lim_lines[i_end], *lim_lines[i_end + 1:],
                      "// --- reactions of the MCM v3.3.1 toluene subset not in the limonene file ---",
                      *added]) + "\n"
    tmp = Path(dst).with_suffix(".unedited.eqn")
    tmp.write_text(text)
    check = parse_eqn(tmp)
    same = (check.species == merged.species and check.ro2 == merged.ro2
            and [(r.label, r.reactants, r.products, r.rate_expr) for r in check.reactions]
            == [(r.label, r.reactants, r.products, r.rate_expr) for r in merged.reactions])
    if not same:
        raise RuntimeError(f"{tmp} does not parse back to the in-code merge")
    out = write_eqn(dst, rate_overrides, src=tmp)
    tmp.unlink()
    return out


def write_xml(dst: Path, extra: dict[str, str] = EXTRA_XML_SPECIES) -> Path:
    """PyCHAM's MCM species xml with ``extra`` {name: SMILES} appended."""
    text = PYCHAM_MCM_XML.read_text()
    marker = "</species_defs>"
    if text.count(marker) != 1:
        raise ValueError(f"unexpected xml layout in {PYCHAM_MCM_XML}")
    add = "".join(f'<species species_number="s{name}" species_name="{name}">\n'
                  f"<smiles>{smiles}</smiles>\n</species>\n" for name, smiles in extra.items())
    Path(dst).write_text(text.replace(marker, add + marker))
    return Path(dst)


def write_constants(dst: Path) -> Path:
    """KPP constants_mcm.f90 shipped with PyCHAM (generic MCM rate coefficients)."""
    shutil.copyfile(PYCHAM_MCM_CONSTANTS, dst)
    return Path(dst)


def scenario_model_var(scenario: Scenario, *, res_file: Path, eqn: Path, xml: Path,
                       constants: Path, j_xlsx: Path | None, particles: dict | None = None,
                       extra: dict | None = None) -> dict[str, str]:
    """PyCHAM model variables (ordered) for a TOMAS-JAX Scenario.

    Gas-phase settings come from the Scenario; ``particles`` adds PyCHAM particle keys
    (None = no particles: number_size_bins = 0). Walls, dilution and coagulation are
    off. ``light_status`` points at the J.xlsx (no ``light_time`` key: it would
    override the times read from the file); ``light_adapt = 0`` because J is given.
    ``scenario.hold_constant`` becomes PyCHAM ``const_comp`` (d[X]/dt = 0 inside the ODE;
    TOMAS-JAX instead resets the species after every coupling interval - a recorded
    difference, see case G3).
    """
    env = scenario.environment
    names = list(scenario.gases_ppb)
    mv: dict[str, str] = {
        "res_file_name": str(Path(res_file).resolve()),
        "chem_sch_name": str(Path(eqn).resolve()),
        "xml_name": str(Path(xml).resolve()),
        "rate_constant_file": str(Path(constants).resolve()),
        "total_model_time": f"{scenario.duration_h * 3600.0:.1f}",
        "update_step": f"{PYCHAM_UPDATE_STEP_S:.1f}",
        "recording_time_step": f"{PYCHAM_RECORDING_STEP_S:.1f}",
        "int_tol": f"{PYCHAM_INT_TOL[0]:g}, {PYCHAM_INT_TOL[1]:g}",
        "temperature": f"{env.temp_k}",
        "tempt": "0.",
        "p_init": f"{env.pres_pa}",
        "rh": f"{env.rh}",
        "rht": "0.",
        "Comp0": ", ".join(names),
        "C0": ", ".join(repr(float(scenario.gases_ppb[n])) for n in names),
        "wall_on": "0",
        "coag_on": "0",
        "McMurry_flag": "-1",
        "dil_fac": "0.",
        "light_adapt": "0",
        "ac_by_cs": "0",
        "umansysprop_update": "0",
    }
    if j_xlsx is not None:
        mv["light_status"] = str(Path(j_xlsx).resolve())
    else:
        mv["light_status"] = "0"
        mv["light_time"] = "0."
    if scenario.hold_constant:
        missing = [h for h in scenario.hold_constant if h not in scenario.gases_ppb]
        if missing:
            raise ValueError(f"held species without an initial value: {missing}")
        mv["const_comp"] = ", ".join(scenario.hold_constant)
    if particles is None:
        mv["number_size_bins"] = "0"
    else:
        mv.update(particles)
    if extra:
        mv.update(extra)
    return mv


def write_model_var(path: Path, mv: dict[str, str]) -> Path:
    for key in ("res_file_name", "chem_sch_name", "xml_name", "rate_constant_file", "p_init"):
        if key not in mv:
            raise ValueError(f"model variables need {key}")
    if not Path(mv["res_file_name"]).is_absolute():
        raise ValueError("res_file_name must be an absolute path")
    if "light_time" in mv and mv.get("light_status", "").endswith(".xlsx"):
        raise ValueError("light_time would override the times of the J.xlsx file")
    Path(path).write_text("".join(f"{k} = {v}\n" for k, v in mv.items()))
    return Path(path)


# PyCHAM v5.8.0 hard-codes the surface tension of every particle (partit_var_prep.py)
SURFT_LINE = "surfT = 72."
SURFT_PATCHED = "surfT = 0. # benchmark attribution copy: Kelvin factor 1"


def ensure_pycham_root(case: str, kelvin_off: bool = False) -> Path:
    """Per-case copy of the pinned PyCHAM checkout (created once with cp -R).

    ``kelvin_off`` uses the copy ``PyCHAM_<case>_nokelvin`` with ``surfT = 0`` in
    partit_var_prep.py. surfT also sets the seed water before t = 0 and the water
    partitioning, so the Kelvin term is removed for water too.
    """
    name = f"PyCHAM_{case}_nokelvin" if kelvin_off else f"PyCHAM_{case}"
    dst = PYCHAM_TEMPLATE.parent / name
    if not (dst / "PyCHAM" / "middle.py").is_file():
        if dst.exists():
            raise RuntimeError(f"{dst} exists but is not a PyCHAM checkout; remove it")
        subprocess.run(["cp", "-R", str(PYCHAM_TEMPLATE), str(dst)], check=True,
                       timeout=COPY_TIMEOUT_S)
    if kelvin_off:
        f = dst / "PyCHAM" / "partit_var_prep.py"
        text = f.read_text()
        if "surfT = 0." not in text:          # (also accepts copies patched earlier)
            if text.count(SURFT_LINE) != 1:
                raise RuntimeError(f"expected exactly one '{SURFT_LINE}' in {f}")
            f.write_text(text.replace(SURFT_LINE, SURFT_PATCHED))
    return dst


def run_pycham(model_var: Path, pycham_root: Path, log_path: Path,
               timeout_s: float = PYCHAM_TIMEOUT_S) -> float:
    """Run one PyCHAM simulation; returns wall time [s]. Raises on failure/timeout.

    ``-I`` keeps this directory (with its config.py etc.) off PyCHAM's sys.path.
    """
    env = dict(os.environ, PYCHAM_ROOT=str(pycham_root), QT_QPA_PLATFORM="offscreen")
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.run([str(PYCHAM_PYTHON), "-I", str(RUN_PYCHAM), str(model_var)],
                              stdout=log, stderr=subprocess.STDOUT, env=env,
                              timeout=timeout_s)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = Path(log_path).read_text()[-3000:]
        raise RuntimeError(f"PyCHAM failed (exit {proc.returncode}); log {log_path}:\n{tail}")
    return wall
