"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

from dmflow.data.disorder_utils import (
    adjust_to_sum_one,
    build_crystal_from_disorder,
    recorders_to_vecs,
)
from dmflow.data.eval_utils import (
    get_disorder_crystals_list,
    load_data,
    smact_validity_disorder,
    structure_validity,
)
from dmflow.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES
from dmflow.utils.joblib_ import joblib_map
from dmflow.rfm.manifold_getter import ManifoldGetter
from dmflow.utils import make_default_disorder_selector

CrysArrayListType = list[dict[str, np.ndarray]]
DisorderMaskSource = Literal["raw_positive", "selector"]
DISORDER_EVAL_KEYS = ("sd_weights", "sd_onehot", "pd_onehot")

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie", impute_nan=True)

from dmflow.utils.matminer_ import get_fingerprints_by_ensemble_averaging


def _require_array(
    crys_array_dict: dict[str, np.ndarray], key: str, context: str
) -> np.ndarray:
    value = crys_array_dict.get(key)
    if value is None:
        raise ValueError(f"{context} is missing required disorder field {key!r}.")
    return value


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _is_multi_eval_array_list(crys_array_list) -> bool:
    return bool(crys_array_list) and isinstance(crys_array_list[0], list)


def _validate_disorder_fields(crys_array_list, context: str) -> None:
    if _is_multi_eval_array_list(crys_array_list):
        for eval_idx, eval_arrays in enumerate(crys_array_list):
            _validate_disorder_fields(eval_arrays, f"{context}[eval={eval_idx}]")
        return

    for crystal_idx, crys_array_dict in enumerate(crys_array_list):
        for key in DISORDER_EVAL_KEYS:
            _require_array(crys_array_dict, key, f"{context}[{crystal_idx}]")


def _atom_numbers_to_onehot(atom_types: np.ndarray) -> np.ndarray:
    atom_types = np.asarray(atom_types)
    indices = atom_types.astype(int) - 1
    if not np.all((0 <= indices) & (indices < NUM_ATOMIC_TYPES)):
        raise ValueError(f"Cannot convert {atom_types=} to one-hot atom weights.")

    onehot = np.zeros((len(atom_types), NUM_ATOMIC_TYPES), dtype=float)
    onehot[np.arange(len(atom_types)), indices] = 1.0
    return onehot


def _get_raw_sd_weights(crys_array_dict: dict[str, np.ndarray]) -> np.ndarray:
    sd_weights = crys_array_dict.get("sd_weights")
    if sd_weights is not None:
        return _to_numpy(sd_weights)

    atom_types = _to_numpy(_require_array(crys_array_dict, "atom_types", "prediction"))
    if atom_types.ndim == 1:
        return _atom_numbers_to_onehot(atom_types)
    return atom_types


def _positive_mask(weights: np.ndarray) -> np.ndarray:
    return (_to_numpy(weights) > 0).astype(int)


def _select_disorder_weights(
    weights: np.ndarray,
    *,
    selector,
    set_norm: bool,
) -> tuple[np.ndarray, np.ndarray]:
    onehot, selected_weights = selector.select(
        _to_numpy(weights),
        method="combined",
        set_norm=set_norm,
    )
    return _to_numpy(onehot).astype(int), _to_numpy(selected_weights)


def _prepare_disorder_fields_for_crystal(
    crys_array_dict: dict[str, np.ndarray],
    disorder_mask_source: DisorderMaskSource,
    *,
    selector,
    set_norm: bool,
) -> dict[str, np.ndarray]:
    out = dict(crys_array_dict)
    raw_sd_weights = _get_raw_sd_weights(out)
    raw_pd_weights = _to_numpy(_require_array(out, "pd_weights", "prediction"))

    if disorder_mask_source == "raw_positive":
        out["sd_weights"] = raw_sd_weights
        out["pd_weights"] = raw_pd_weights
        out["sd_onehot"] = _positive_mask(raw_sd_weights)
        out["pd_onehot"] = _positive_mask(raw_pd_weights)
    elif disorder_mask_source == "selector":
        sd_onehot, sd_weights = _select_disorder_weights(
            raw_sd_weights,
            selector=selector,
            set_norm=set_norm,
        )
        pd_onehot, pd_weights = _select_disorder_weights(
            raw_pd_weights,
            selector=selector,
            set_norm=set_norm,
        )
        out["atom_types"] = sd_weights
        out["sd_weights"] = sd_weights
        out["pd_weights"] = pd_weights
        out["sd_onehot"] = sd_onehot
        out["pd_onehot"] = pd_onehot
    else:
        raise ValueError(f"Unknown disorder mask source: {disorder_mask_source!r}.")

    return out


def _prepare_disorder_fields(
    crys_array_list,
    disorder_mask_source: DisorderMaskSource,
    *,
    selector=None,
    set_norm: bool,
):
    if disorder_mask_source == "selector" and selector is None:
        selector = make_default_disorder_selector()

    if _is_multi_eval_array_list(crys_array_list):
        return [
            _prepare_disorder_fields(
                eval_arrays,
                disorder_mask_source,
                selector=selector,
                set_norm=set_norm,
            )
            for eval_arrays in crys_array_list
        ]

    return [
        _prepare_disorder_fields_for_crystal(
            crys_array_dict,
            disorder_mask_source,
            selector=selector,
            set_norm=set_norm,
        )
        for crys_array_dict in crys_array_list
    ]


class DisorderCrystal:
    def __init__(
        self,
        crys_array_dict,
        make_atom_types_discrete: bool = True
    ):
        super().__init__()
        self.frac_coords = crys_array_dict["frac_coords"]

        self.lengths = crys_array_dict["lengths"].squeeze()
        assert self.lengths.ndim == 1
        self.angles = crys_array_dict["angles"].squeeze()
        assert self.lengths.ndim == 1
        self.dict = crys_array_dict

        self.atom_types = crys_array_dict["atom_types"]
        self.pd_weights = crys_array_dict["pd_weights"]
        self.pd_frac_coords = crys_array_dict["pd_frac_coords"]
        self.sd_weights = _require_array(crys_array_dict, "sd_weights", "crystal array")
        self.sd_onehot = _require_array(crys_array_dict, "sd_onehot", "crystal array")
        self.pd_onehot = _require_array(crys_array_dict, "pd_onehot", "crystal array")

        assert isinstance(self.sd_weights, np.ndarray), f"{type(self.sd_weights)=}"
        assert isinstance(self.pd_weights, np.ndarray), f"{type(self.pd_weights)=}"
        assert isinstance(self.sd_onehot, np.ndarray), f"{type(self.sd_onehot)=}"
        assert isinstance(self.pd_onehot, np.ndarray), f"{type(self.pd_onehot)=}"

        if make_atom_types_discrete and (self.frac_coords.ndim == self.atom_types.ndim):
            if self.atom_types.shape[-1] == NUM_ATOMIC_TYPES:
                self.atom_types = ManifoldGetter._inverse_atomic_one_hot(
                    self.atom_types
                )
            elif self.atom_types.shape[-1] == NUM_ATOMIC_BITS:
                self.atom_types = ManifoldGetter._inverse_atomic_bits(self.atom_types)
            else:
                raise ValueError()

        self.parse_disorder()
        self.get_disorder_structure()
        if self.constructed:
            self.get_validity()
            self.get_fingerprints()
        else:
            self.valid = False

    def parse_disorder(self):
        site_comps, site_weights = [], []
        site_frac_coords = []
        for i in range(len(self.atom_types)):
            atom_sd_weight = self.sd_weights[i]
            atom_pd_weight = self.pd_weights[i]
            if sum(self.pd_onehot[i]) == 2:
                site_comps.extend([[self.atom_types[i]]] * 2)
                site_weights.extend([[atom_pd_weight[0]], [atom_pd_weight[1]]])
                site_frac_coords.extend([self.frac_coords[i], self.pd_frac_coords[i]])
            else:
                site_atoms = self.sd_onehot[i].nonzero()[0]
                site_occs = adjust_to_sum_one(atom_sd_weight[site_atoms])
                site_comps.append((site_atoms + 1).tolist())
                site_weights.append(site_occs)
                site_frac_coords.append(self.frac_coords[i])

        self.site_comps = site_comps
        self.site_weights = site_weights
        self.site_frac_coords = site_frac_coords

    def get_disorder_structure(self):
        if min(self.lengths) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        # this catches validity failures down the line
        elif (1 > self.atom_types).any() or (self.atom_types > 104).any():
            self.constructed = False
            self.invalid_reason = f"{self.atom_types=} are not with range"
        else:
            try:
                species = [
                    dict(zip(comp, weight))
                    for comp, weight in zip(self.site_comps, self.site_weights)
                ]

                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=species,
                    coords=self.site_frac_coords,
                    coords_are_cartesian=False,
                )

                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = "unrealistically_small_lattice"
            except TypeError:
                self.constructed = False
                self.invalid_reason = f"{self.atom_types=} are not possible"
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"

    def get_validity(self):
        self.comp_valid = smact_validity_disorder(self.site_comps, self.site_weights)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False

        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        # elem_counter = Counter(self.atom_types)
        # comp = Composition(elem_counter)
        comp = self.structure.composition
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = get_fingerprints_by_ensemble_averaging(
                self.structure, CrystalNNFP
            )
        except Exception as e:
            warnings.warn(
                f"Exception while constructing fingerprint: {e}",
                RuntimeWarning,
            )
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def get_file_paths(root_path, task, label="", suffix="pt"):

    if label == "":
        out_name = f"eval_{task}.{suffix}"
    else:
        out_name = f"eval_{task}_{label}.{suffix}"
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(
    file_path: Path,
    batch_idx: int = 0,  # 0 for single eval, -1 for multi-eval
) -> CrysArrayListType:
    """batch_idx == -1, diffcsp format
    batch_idx == -2, cdvae format
    """

    data = load_data(str(file_path.resolve()))

    if batch_idx == -1:
        batch_size = len(data["frac_coords"])
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_disorder_crystals_list(
                data["frac_coords"][i],
                data["atom_types"][i],
                data["lengths"][i],
                data["angles"][i],
                data["num_atoms"][i],
                data["pd_weights"][i],
                data["pd_frac_coords"][i],
                data["sd_weights"][i] if "sd_weights" in data else None,
                data["sd_onehot"][i] if "sd_onehot" in data else None,
                data["pd_onehot"][i] if "pd_onehot" in data else None,
            )
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_disorder_crystals_list(
            data["frac_coords"],
            data["atom_types"],
            data["lengths"],
            data["angles"],
            data["num_atoms"],
            data["pd_weights"],
            data["pd_frac_coords"],
            data["sd_weights"] if "sd_weights" in data else None,
            data["sd_onehot"] if "sd_onehot" in data else None,
            data["pd_onehot"] if "pd_onehot" in data else None,
        )
    else:
        crys_array_list = get_disorder_crystals_list(
            data["frac_coords"][batch_idx],
            data["atom_types"][batch_idx],
            data["lengths"][batch_idx],
            data["angles"][batch_idx],
            data["num_atoms"][batch_idx],
            data["pd_weights"][batch_idx],
            data["pd_frac_coords"][batch_idx],
            data["sd_weights"][batch_idx] if "sd_weights" in data else None,
            data["sd_onehot"][batch_idx] if "sd_onehot" in data else None,
            data["pd_onehot"][batch_idx] if "pd_onehot" in data else None,
        )

    if "input_data_batch" in data:
        batch_by_eval = data["input_data_batch"]

        if isinstance(batch_by_eval, dict):
            true_crystal_array_list = get_disorder_crystals_list(
                batch_by_eval["frac_coords"],
                batch_by_eval["atom_types"],
                batch_by_eval["lengths"],
                batch_by_eval["angles"],
                batch_by_eval["num_atoms"],
                batch_by_eval["pd_weights"],
                batch_by_eval["pd_frac_coords"],
                sd_weights=batch_by_eval.get("sd_weights", None),
                sd_onehot=batch_by_eval.get("sd_onehot", None),
                pd_onehot=batch_by_eval.get("pd_onehot", None),
            )
        elif hasattr(batch_by_eval, "frac_coords"):
            true_crystal_array_list = get_disorder_crystals_list(
                batch_by_eval.frac_coords,
                batch_by_eval.atom_types,
                batch_by_eval.lengths,
                batch_by_eval.angles,
                batch_by_eval.num_atoms,
                batch_by_eval.pd_weights,
                batch_by_eval.pd_frac_coords,
                sd_weights=getattr(batch_by_eval, "sd_weights", None),
                sd_onehot=getattr(batch_by_eval, "sd_onehot", None),
                pd_onehot=getattr(batch_by_eval, "pd_onehot", None),
            )
        elif isinstance(batch_by_eval, list) and not isinstance(batch_by_eval[0], Data):
            if isinstance(batch_by_eval[0][0], Data) and hasattr(
                batch_by_eval[0][0], "frac_coords"
            ):
                true_crystal_array_list = []
                features = [
                    "frac_coords",
                    "atom_types",
                    "lengths",
                    "angles",
                    "pd_weights",
                    "pd_frac_coords",
                    "sd_weights",
                    "sd_onehot",
                    "pd_onehot",
                ]

                # we collect crystals from the first eval
                crystals_in_first_eval = batch_by_eval[0]
                for crystal in crystals_in_first_eval:
                    true_crystal_array_list.append(
                        {k: crystal[k].detach().cpu().numpy() for k in features}
                    )

                # ... and make sure that every eval is in the same order
                for crystals_in_batch in batch_by_eval[1:]:
                    for i, crystal in enumerate(crystals_in_batch):
                        for k in features:
                            t = true_crystal_array_list[i][k]
                            other = crystal[k].detach().cpu().numpy()
                            assert np.allclose(
                                t, other
                            ), f"{t=} != {other=}, that means the first eval is not in the same order as the subsequent"
            else:
                true_crystal_array_list = None
        else:
            true_crystal_array_list = None
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def cif_to_crys_array_dict(cif: str) -> dict[str, np.ndarray]:
    with HiddenPrints():
        structure = Structure.from_str(cif, fmt="cif", primitive=True)
    lattice = structure.lattice
    return {
        "frac_coords": structure.frac_coords,
        "atom_types": np.array([_.Z for _ in structure.species]),
        "lengths": np.array(lattice.abc),
        "angles": np.array(lattice.angles),
    }


def cif_to_crystal(cif: str) -> DisorderCrystal:
    structure, disorder_info = build_crystal_from_disorder(cif)
    sd_onehot, sd_weights, pd_weights, pd_frac_coords, pd_onehot = recorders_to_vecs(
        disorder_info
    )
    graph_arrays = {
        "frac_coords": structure.frac_coords,
        "atom_types": np.array([_.Z for _ in structure.species]),
        "lengths": np.array(structure.lattice.abc),
        "angles": np.array(structure.lattice.angles),
        "sd_weights": sd_weights.numpy(),
        "pd_weights": pd_weights.numpy(),
        "pd_frac_coords": pd_frac_coords.numpy(),
        "sd_onehot": sd_onehot.numpy(),
        "pd_onehot": pd_onehot.numpy(),
    }
    return DisorderCrystal(graph_arrays)


def cache_to_crystal(cache: dict[str, np.ndarray]) -> DisorderCrystal:
    sd_onehot, sd_weights, pd_weights, pd_frac_coords, pd_onehot = recorders_to_vecs(
        cache["disorder_info"]
    )

    graph_arrays = {
        "frac_coords": cache["graph_arrays"][0],
        "atom_types": cache["graph_arrays"][1],
        "lengths": cache["graph_arrays"][2],
        "angles": cache["graph_arrays"][3],
        "sd_weights": sd_weights.numpy(),
        "pd_weights": pd_weights.numpy(),
        "pd_frac_coords": pd_frac_coords.numpy(),
        "sd_onehot": sd_onehot.numpy(),
        "pd_onehot": pd_onehot.numpy(),
    }

    return DisorderCrystal(graph_arrays)


def load_gt_crystal_array_list(ground_truth_path: Path) -> CrysArrayListType:
    ground_truth_path = Path(ground_truth_path)
    if ground_truth_path.suffix == ".csv":
        csv = pd.read_csv(ground_truth_path)
        gt_crys = joblib_map(
            cif_to_crys_array_dict,
            csv["cif"],
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(csv["cif"]),
        )
    elif ground_truth_path.suffix == ".pt":
        pkl = torch.load(ground_truth_path)
        gt_crys = joblib_map(
            cif_to_crys_array_dict,
            [p["cif"] for p in pkl],
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(pkl),
        )
    else:
        raise ValueError(f"{ground_truth_path=} had an unrecognized suffix.")
    return gt_crys


def load_gt_crystals(ground_truth_path: Path) -> CrysArrayListType:
    ground_truth_path = Path(ground_truth_path)
    if ground_truth_path.suffix == ".csv":
        csv = pd.read_csv(ground_truth_path)
        gt_crys = joblib_map(
            cif_to_crystal,
            csv["cif"],
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(csv["cif"]),
        )
    elif ground_truth_path.suffix == ".pt":
        pkl = torch.load(ground_truth_path)
        # test_crys = cache_to_crystal(pkl[10])
        gt_crys = joblib_map(
            cache_to_crystal,
            pkl,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(pkl),
        )

    else:
        raise ValueError(f"{ground_truth_path=} had an unrecognized suffix.")
    return gt_crys


def safe_crystal(x: dict[str, np.ndarray]) -> DisorderCrystal:
    angle_valid = False
    if np.all(50 < x["angles"]) and np.all(x["angles"] < 130):
        angle_valid = True
        try:
            return DisorderCrystal(x)
        except Exception as e:
            warnings.warn(
                f"Exception while creating crystal; using fallback crystal: {e}",
                RuntimeWarning,
            )
            pass

    if not angle_valid:
        warnings.warn(
            "Crystal angles are outside the range (50, 130); using fallback crystal.",
            RuntimeWarning,
        )

    atom_types = np.zeros((1, NUM_ATOMIC_TYPES))
    atom_types[0, -1] = 1
    return DisorderCrystal(
        {
            "frac_coords": np.zeros((1, 3)),
            "atom_types": atom_types,
            "lengths": 100 * np.ones((3,)),
            "angles": np.ones((3,)) * 90,
            "sd_weights": atom_types.copy(),
            "sd_onehot": atom_types.astype(int),
            "pd_weights": np.array([[1.0, 0.0]]),
            "pd_frac_coords": np.zeros((1, 3)),
            "pd_onehot": np.array([[1, 0]]),
        }
    )


def get_Crystal_obj_lists(
    path: Path,
    multi_eval: bool,
    ground_truth_path: Path | None = None,
    disorder_mask_source: DisorderMaskSource = "selector",
    set_norm: bool = True,
) -> tuple[list[DisorderCrystal], list[DisorderCrystal]]:
    batch_idx = -1 if multi_eval else 0
    crys_array_list, true_crystal_array_list = get_crystal_array_list(
        path,
        batch_idx=batch_idx,
    )

    if ground_truth_path is not None:
        gt_crys = load_gt_crystals(ground_truth_path)
    else:
        if true_crystal_array_list is None:
            raise ValueError(
                "Ground-truth crystal arrays were not found in the prediction file."
            )

        gt_crys = joblib_map(
            lambda x: DisorderCrystal(x),
            true_crystal_array_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="gt_cryst",
            total=len(true_crystal_array_list),
        )

    crys_array_list = _prepare_disorder_fields(
        crys_array_list,
        disorder_mask_source,
        set_norm=set_norm,
    )
    _validate_disorder_fields(crys_array_list, "prediction")

    # TIMEOUT: this MAYBE could be speed up (with timeout) using https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers
    # I could not find a way to have timeouts on single processes using joblib without crashing the whole array.
    # the issue is that if there's a crystal that does not process, the whole thing doesn't return...
    #
    # I tried this but it doesn't work
    # https://github.com/joblib/joblib/pull/366#issuecomment-267603530
    #
    # instead of using timeout, I just replace all crystals with unrealistic angles with a null crystal
    if multi_eval:
        pred_crys = []
        num_evals = len(crys_array_list)
        for eval_crystal_arrays in crys_array_list:
            pred_crys.append(
                joblib_map(
                    safe_crystal,  # instead I remove unrealistic crystals
                    eval_crystal_arrays,
                    n_jobs=-4,
                    inner_max_num_threads=1,
                    desc="pred_cryst",
                    total=len(eval_crystal_arrays),
                )
            )
    else:
        num_evals = 1

        pred_crys = joblib_map(
            safe_crystal,  # instead I remove unrealistic crystals
            crys_array_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="pred_cryst",
            total=len(crys_array_list),
        )

    return pred_crys, gt_crys, num_evals


def save_metrics_only_overwrite_newly_computed(
    path: Path, metrics: dict[str, float]
) -> None:
    # only overwrite metrics computed in the new run.
    if Path(path).exists():
        with open(path, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(metrics)
            else:
                with open(path, "w") as f:
                    json.dump(metrics, f)
        if isinstance(written_metrics, dict):
            with open(path, "w") as f:
                json.dump(written_metrics, f)
    else:
        with open(path, "w") as f:
            json.dump(metrics, f)
