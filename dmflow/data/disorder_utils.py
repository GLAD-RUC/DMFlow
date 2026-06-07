import copy
import re
import warnings
from decimal import Decimal, getcontext
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from diffcsp.common.data_utils import build_crystal_graph, get_symmetry_info
from p_tqdm import p_umap
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from tqdm import tqdm


from dmflow.data import NUM_ATOMIC_TYPES

getcontext().prec = 10

DisorderCodebook = {0: "order", 1: "sd", 2: "pd", 3: "spd"}


class DisorderRecorder:
    """Record the disorder information of a site."""

    def __init__(self, site_number, sd_info=None, pd_info=None):
        self.site_number = site_number
        self.sd_info = {} if sd_info is None else sd_info
        self.pd_info = {} if pd_info is None else pd_info

    def get_disorder_type(self):
        """Encode order, SD, PD, and SPD sites as 1, 2, 3, and 4."""
        disorder_type = 1
        if len(self.sd_info) > 1:
            disorder_type += 1
        if self.pd_info:
            disorder_type += 2

        return disorder_type

    def get_sd_vectors(self):
        one_hot = [0] * NUM_ATOMIC_TYPES
        occ_vector = [0.0] * NUM_ATOMIC_TYPES
        for element, occ in self.sd_info.items():
            atomic_number = Element(element).Z
            one_hot[atomic_number - 1] = 1
            occ_vector[atomic_number - 1] = occ
        return torch.LongTensor(one_hot), torch.FloatTensor(occ_vector)

    def get_pd_vectors(self):
        if self.pd_info:
            occ_vector = [1.0 - self.pd_info[1], self.pd_info[1]]
            pd_frac = self.pd_info[0]
            pd_onehot = [1, 1]
        else:
            occ_vector = [1.0, 0.0]
            pd_frac = [0.0] * 3
            pd_onehot = [1, 0]

        return (
            torch.FloatTensor(pd_frac),
            torch.FloatTensor(occ_vector),
            torch.LongTensor(pd_onehot),
        )

    def __repr__(self):
        return f"DisorderRecorder(site_number={self.site_number}, sd_info={self.sd_info}, pd_info={self.pd_info})"


def recorders_to_vecs(
    disorder_recs: List[DisorderRecorder],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a list of DisorderRecorder to tensors.
    Returns:
        - sd_vecs: Tensor of shape (N, NUM_ATOMIC_TYPES) for substitutional disorder.
        - pd_vecs: Tensor of shape (N, 3) for positional disorder.
        - sd_occupancies: Tensor of shape (N, NUM_ATOMIC_TYPES) for occupancies.
    """
    if isinstance(disorder_recs[0], tuple):
        disorder_recs = [DisorderRecorder(*rec) for rec in disorder_recs]

    sd_onehot, sd_weights = zip(*(rec.get_sd_vectors() for rec in disorder_recs))
    sd_onehot = torch.stack(sd_onehot, dim=0)
    sd_weights = torch.stack(sd_weights, dim=0)

    pd_fracs, pd_weights, pd_onehot = zip(
        *(rec.get_pd_vectors() for rec in disorder_recs)
    )

    pd_weights = torch.stack(pd_weights, dim=0)
    pd_fracs = torch.stack(pd_fracs, dim=0)
    pd_onehot = torch.stack(pd_onehot, dim=0)

    return sd_onehot, sd_weights, pd_weights, pd_fracs, pd_onehot


def vecs_to_recorders(
    sd_onehot: torch.Tensor,
    sd_weights: torch.Tensor,
    pd_fracs: torch.Tensor,
    pd_weights: torch.Tensor,
) -> List[DisorderRecorder]:
    """
    Convert tensors back to a list of DisorderRecorder.
    Args:
        - sd_onehot: Tensor of shape (N, NUM_ATOMIC_TYPES) for substitutional disorder.
        - sd_weights: Tensor of shape (N, NUM_ATOMIC_TYPES) for occupancies.
        - pd_fracs: Tensor of shape (N, 3) for positional disorder.
        - pd_weights: Tensor of shape (N, 2) for positional disorder occupancies.
    Returns:
        - List of DisorderRecorder objects.
    """
    raise NotImplementedError("This function is not implemented yet.")


def extract_element_symbol(s):
    match = re.match(r"^([A-Za-z]+)", s)
    if match is None:
        raise ValueError(f"Could not extract an element symbol from {s!r}.")
    res = match.group(1)
    Element(res).Z
    return str(res)


def adjust_to_sum_one(values):
    values = [Decimal(str(v)) for v in values]

    total = sum(values)

    if total == Decimal("1.0"):
        return list(map(float, values))

    values[-1] = values[-1] + (Decimal("1.0") - total)

    if values[-1] < 0:
        raise ValueError("Could not adjust the last occupancy to make the sum 1.0.")

    return list(map(float, values))


def preprocess_disorder(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    def process_one(
        row,
        niggli,
        primitive,
        graph_method,
        prop_list,
        use_space_group=False,
        tol=0.01,
    ):
        crystal_str = row["cif"]
        try:
            crystal, disorder_info = build_crystal_from_disorder(
                crystal_str, niggli=niggli, primitive=primitive
            )
        except Exception as e:
            warnings.warn(
                f"Error processing crystal {crystal_str!r}: {e}",
                RuntimeWarning,
            )
            return None

        result_dict = {}
        if use_space_group:
            crystal, sym_info = get_symmetry_info(crystal, tol=tol)
            result_dict.update(sym_info)
        else:
            result_dict["spacegroup"] = 1

        graph_arrays = build_crystal_graph(crystal, graph_method)
        if graph_arrays is None:
            return None
        properties = {k: row[k] for k in prop_list if k in row.keys()}
        result_dict.update(
            {
                "mp_id": row["material_id"],
                "cif": crystal_str,
                "graph_arrays": graph_arrays,
                "disorder_info": disorder_info,
            }
        )
        result_dict.update(properties)
        return result_dict

    df = pd.read_csv(input_file)

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers,
    )

    mpid_to_results = {
        result["mp_id"]: result for result in unordered_results if result
    }

    ordered_results = []
    for idx in range(len(df)):
        mp_id = df.iloc[idx]["material_id"]
        if mp_id in mpid_to_results:
            ordered_results.append(mpid_to_results[mp_id])

    return ordered_results


def update_batch(batch, gen_data):
    from diffcsp.common.data_utils import lattices_to_params_shape

    new_batch = copy.deepcopy(batch)

    gen_keys = ["num_atoms", "atom_types", "frac_coords", "lattices"]
    unknown_keys = set(gen_data.keys()) - set(gen_keys)
    if unknown_keys:
        raise ValueError(f"Unknown keys in gen_data: {unknown_keys}")

    for key, value in gen_data.items():
        setattr(new_batch, key, value)

    lengths, angles = lattices_to_params_shape(new_batch.lattices)
    new_batch.lengths = lengths
    new_batch.angles = angles
    new_batch.atom_types = (
        new_batch.atom_types.argmax(dim=-1) + 1
    )  # Convert one-hot to indices

    return new_batch


def detect_pd(structure: Structure) -> Tuple[Dict, bool]:
    """Return positional disorder records and whether they satisfy the pairing rule."""

    disorder_groups = {}
    pd_records = {}

    for i, site in enumerate(structure.sites):
        if len(site.species) != 1:
            continue
        elem = list(site.species.keys())[0]
        occ = list(site.species.values())[0]
        if occ < 1.0:
            disorder_groups.setdefault(elem, []).append(
                {
                    "index": i,
                    "site": site,
                    "coords": site.frac_coords,
                    "occ": occ,
                }
            )

    for elem, sites in disorder_groups.items():
        n = len(sites)

        if n % 2 != 0:
            warnings.warn(
                f"Element {elem} has an odd number of sites with occupancy < 1.0: {n}.",
                RuntimeWarning,
            )
            return {}, False

        used = set()

        for i in range(n):
            if i in used:
                continue
            best_j = None
            best_dist = float("inf")

            for j in range(i + 1, n):
                if j in used:
                    continue
                # Pair sites only when their occupancies sum to one.
                occ_sum = sites[i]["occ"] + sites[j]["occ"]
                if abs(occ_sum - 1.0) > 0.01:
                    continue
                dist = structure.get_distance(sites[i]["index"], sites[j]["index"])
                if dist < best_dist:
                    best_j = j
                    best_dist = dist

            if best_j is not None:
                used.add(i)
                used.add(best_j)
                if sites[i]["occ"] < sites[best_j]["occ"]:
                    sites[i], sites[best_j] = sites[best_j], sites[i]
                pd_records.update(
                    {
                        sites[i]["index"]: {
                            "index": sites[best_j]["index"],
                            "coords": sites[best_j]["coords"].tolist(),
                            "occ": sites[best_j]["occ"],
                        },
                    }
                )

        if len(used) != n:
            warnings.warn(
                f"Element {elem} has unmatched sites: {len(used)} used out of {n}.",
                RuntimeWarning,
            )
            return {}, False

    return pd_records, True


def detect_sd(structure: Structure) -> Tuple[List, List, List]:
    """Return SD records, dominant atoms, and dominant coordinates."""
    sd_records = []
    dominant_eles = []
    dominant_coords = []
    for i, site in enumerate(structure.sites):
        species_dict = site.species.as_dict()
        species_dict = {
            extract_element_symbol(str(e)): occu for e, occu in species_dict.items()
        }

        if len(species_dict) > 1:
            sorted_species = sorted(
                species_dict.items(), key=lambda x: x[1], reverse=True
            )
            dominant_element, _ = sorted_species[0]

            elems, occs = zip(*sorted_species)
            occs = adjust_to_sum_one(occs)
            sd_records.append(dict(zip(elems, occs)))
        else:
            dominant_element = next(iter(species_dict.keys()))
            sd_records.append({str(dominant_element): 1.0})

        dominant_eles.append(dominant_element)
        dominant_coords.append(site.frac_coords.tolist())

    return sd_records, dominant_eles, dominant_coords


def build_crystal_from_disorder(
    crystal_str,
    niggli=True,
    primitive=True,
    min_sites=2,
    max_sites=50,
):
    """Build crystal from cif string."""
    structure = Structure.from_str(crystal_str, fmt="cif")

    if primitive:
        structure = structure.get_primitive_structure()

    if niggli:
        structure = structure.get_reduced_structure()

    pd_pairs, pd_valid = detect_pd(structure)
    if not pd_valid:
        raise ValueError(
            "Crystal does not meet the criteria for positional disorder (PD)."
        )

    disorder_recs = []
    dominant_eles = []
    dominant_coords = []

    for i, site in enumerate(structure.sites):
        species_dict = site.species.as_dict()
        species_dict = {
            extract_element_symbol(str(e)): occu for e, occu in species_dict.items()
        }

        sd_info, pd_info = {}, ()

        if len(species_dict) > 1:
            sorted_species = sorted(
                species_dict.items(), key=lambda x: x[1], reverse=True
            )
            dominant_element, _ = sorted_species[0]
            dominant_coord = site.frac_coords.tolist()

            elems, occs = zip(*sorted_species)
            occs = adjust_to_sum_one(occs)
            sd_info = dict(zip(elems, occs))
        else:
            ele, occ = next(iter(species_dict.items()))
            if occ < 1.0:
                if i in pd_pairs:
                    dominant_element = ele
                    dominant_coord = site.frac_coords.tolist()
                    sd_info = {dominant_element: 1.0}
                    pd_info = (pd_pairs[i]["coords"], pd_pairs[i]["occ"])
                else:
                    dominant_element, dominant_coord = None, None
            else:
                dominant_element = ele
                dominant_coord = site.frac_coords.tolist()
                sd_info = {dominant_element: 1.0}

        if sd_info:
            dominant_eles.append(dominant_element)
            dominant_coords.append(dominant_coord)
            disorder_recs.append((i, sd_info, pd_info))

    if len(dominant_eles) < min_sites or len(dominant_eles) > max_sites:
        raise ValueError(
            f"Crystal has {len(dominant_eles)} sites, outside the allowed range "
            f"[{min_sites}, {max_sites}]."
        )

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*structure.lattice.parameters),
        species=dominant_eles,
        coords=dominant_coords,
        coords_are_cartesian=False,
    )

    return canonical_crystal, disorder_recs


def mask_and_renormalize(data, mask):
    assert data.shape == mask.shape
    masked = data * mask
    return masked / masked.sum(axis=-1, keepdims=True)
