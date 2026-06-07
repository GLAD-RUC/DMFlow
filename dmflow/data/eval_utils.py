import itertools
import smact
from smact.screening import pauling_test
import torch
from dmflow.data import NUM_ATOMIC_TYPES
import numpy as np
from pymatgen.core.periodic_table import Element
from tqdm import tqdm
from pymatgen.core.structure import Structure


def load_data(file_path):
    if file_path[-3:] == "npy" or file_path[-2:] == "np":
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == "input_data_batch":
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path, map_location="cpu")
    return data


def get_disorder_crystals_list(
    frac_coords,
    atom_types,
    lengths,
    angles,
    num_atoms,
    pd_weights,
    pd_frac_coords,
    sd_weights=None,
    sd_onehot=None,
    pd_onehot=None,
):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
        pd_weights: (num_atoms, 2)
        pd_frac_coords: (num_atoms, 3)

        sd_weights: (num_atoms, ATOMIC_TYPES), optional.
        "for reconstruction task: gt_cryst has this field, pred_cryst does not."
        "for generation task: both gt_cryst and pred_cryst have this field."

        sd_onehot: (num_atoms, ATOMIC_TYPES), optional.
        pd_onehot: (num_atoms, ATOMIC_TYPES), optional.
        "for generation task: both gt_cryst and pred_cryst have this field."
        "for reconstruction task: gt_cryst has this field, pred_cryst does not."
    """
    assert (
        frac_coords.size(0)
        == atom_types.size(0)
        == num_atoms.sum()
        == pd_weights.size(0)
        == pd_frac_coords.size(0)
    )
    assert frac_coords.size(1) == pd_frac_coords.size(1)
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_pd_weights = pd_weights.narrow(0, start_idx, num_atom)
        cur_pd_frac_coords = pd_frac_coords.narrow(0, start_idx, num_atom)

        if sd_weights is not None:
            cur_sd_weights = sd_weights.narrow(0, start_idx, num_atom)
        else:
            cur_sd_weights = None

        if sd_onehot is not None:
            cur_sd_onehot = sd_onehot.narrow(0, start_idx, num_atom)
        else:
            cur_sd_onehot = None

        if pd_onehot is not None:
            cur_pd_onehot = pd_onehot.narrow(0, start_idx, num_atom)
        else:
            cur_pd_onehot = None

        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
                "pd_weights": cur_pd_weights.detach().cpu().numpy(),
                "pd_frac_coords": cur_pd_frac_coords.detach().cpu().numpy(),
                "sd_weights": (
                    cur_sd_weights.detach().cpu().numpy()
                    if cur_sd_weights is not None
                    else None
                ),
                "sd_onehot": (
                    cur_sd_onehot.detach().cpu().numpy()
                    if cur_sd_onehot is not None
                    else None
                ),
                "pd_onehot": (
                    cur_pd_onehot.detach().cpu().numpy()
                    if cur_pd_onehot is not None
                    else None
                ),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


# The values for threshold and top_p can be tuned from training-set statistics.
def probs_to_binary(probs, method="threshold", threshold=0.3, top_p=0.9, temper=0.1):
    """
    Convert a probability matrix to a binary 0-1 matrix using specified method.

    Parameters:
        probs (np.ndarray): Shape (n, d), each row is a normalized probability distribution
        method (str): 'threshold' or 'top_p'
        threshold (float): Threshold for the threshold-based method. Probabilities > threshold become 1.
        top_p (float): Cumulative probability threshold for top-p (nucleus) method (0 < top_p <= 1)

    Returns:
        binary_preds (np.ndarray): Shape (n, d), binary matrix with 0s and 1s
    """

    def top_p_selection(probs, top_p):
        """
        Selects the top-p classes based on cumulative probability.
        """
        n, d = probs.shape
        binary_preds = np.zeros_like(probs, dtype=int)

        for i in range(n):
            # Sort class indices in descending order of probability
            sorted_indices = np.argsort(probs[i])[::-1]
            sorted_probs = probs[i][sorted_indices]

            # Compute cumulative probability
            cumsum_probs = np.cumsum(sorted_probs)

            # Find the smallest index where cumulative probability >= top_p
            cutoff_idx = np.searchsorted(cumsum_probs, top_p, side="right")
            cutoff_idx = max(1, cutoff_idx)

            # Mark the top-p classes as 1
            selected_indices = sorted_indices[:cutoff_idx]
            binary_preds[i, selected_indices] = 1
        return binary_preds

    probs = np.array(probs)

    if method == "threshold":
        # Set to 1 where probability is greater than or equal to threshold
        binary_preds = (probs > threshold).astype(int)

    elif method == "top_p":
        binary_preds = top_p_selection(probs, top_p)

    elif method == "softmax_top_p":
        probs = probs / temper
        exp_probs = np.exp(probs - np.max(probs, axis=-1, keepdims=True))
        softmax_probs = exp_probs / np.sum(exp_probs, axis=-1, keepdims=True)
        binary_preds = top_p_selection(softmax_probs, top_p)

    elif method == "zscore":
        binary_preds = np.zeros_like(probs)
        for i in range(probs.shape[0]):
            threshold = min(np.mean(probs[i]) + 2.5 * np.std(probs[i]), max(probs[i]))
            binary_preds[i] = (probs[i] >= threshold).astype(int)
        return binary_preds

    else:
        raise ValueError(
            "method must be 'threshold' or 'top_p' or 'softmax_top_p' or 'zscore'"
        )

    return binary_preds


def smact_validity_disorder(
    disorder_comp,
    disorder_weights,
    use_pauling_test=True,
    include_alloys=True,
    tolerance=1e-2,
):
    """
    Check SMACT validity for disordered crystals with substitutional disorder.

    Args:
        disorder_comp: List of lists, where each inner list contains atomic numbers
                      for elements that can occupy each site
        disorder_weights: List of lists, where each inner list contains occupancy
                         weights for corresponding elements at each site
        use_pauling_test: Whether to use Pauling electronegativity test
        include_alloys: Whether to include metallic alloys as valid
        tolerance: Numerical tolerance for charge balance

    Returns:
        bool: Whether the disordered composition is chemically valid
    """

    def atomic_number_to_symbol(atomic_number):
        return str(Element.from_Z(atomic_number))

    # Convert atomic numbers to symbols for all possible elements
    all_elements = set()
    for site_comp in disorder_comp:
        for atom_number in site_comp:
            all_elements.add(atomic_number_to_symbol(atom_number))

    # Get SMACT element information
    space = smact.element_dictionary(tuple(all_elements))
    elem_to_smact = {elem: space[elem] for elem in all_elements}

    # Handle single element case
    if len(all_elements) == 1:
        return True

    # Handle all-metal alloy case
    if include_alloys:
        if all(elem in smact.metals for elem in all_elements):
            return True

    # Calculate effective composition by weighted average
    effective_comp = {}
    for site_idx, (site_comp, site_weights) in enumerate(
        zip(disorder_comp, disorder_weights)
    ):
        # Normalize weights for this site
        # total_weight = sum(site_weights)
        # if total_weight == 0:
        #     continue

        # normalized_weights = [w / total_weight for w in site_weights]
        normalized_weights = site_weights

        for atom_number, weight in zip(site_comp, normalized_weights):
            elem = atomic_number_to_symbol(atom_number)
            if elem not in effective_comp:
                effective_comp[elem] = 0
            effective_comp[elem] += weight

    # Get oxidation states and electronegativity for effective elements
    effective_elements = list(effective_comp.keys())
    ox_combos = []
    electronegs = []

    for elem_symbol in effective_elements:
        smact_elem = elem_to_smact[elem_symbol]
        ox_combos.append(smact_elem.oxidation_states)
        electronegs.append(smact_elem.pauling_eneg)

    # Check if oxidation state combinations are manageable
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False

    # Test all oxidation state combinations
    for ox_states in itertools.product(*ox_combos):
        # Calculate charge balance with fractional occupancies
        total_charge = 0
        for elem_symbol, ox_state in zip(effective_elements, ox_states):
            total_charge += effective_comp[elem_symbol] * ox_state

        # Check charge neutrality within tolerance
        if abs(total_charge) <= tolerance:
            # Electronegativity test
            electroneg_OK = True
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # If no electronegativity data, assume it's okay
                    electroneg_OK = True

            if electroneg_OK:
                return True

    return False


def structure_validity(crystal, cutoff=0.5, volume_cutoff=0.1):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))

    if dist_mat.min() < cutoff or crystal.volume < volume_cutoff:
        return False
    else:
        return True


def get_element_set(structure: Structure):
    """
    Extract the set of all chemical elements present in a (possibly disordered) pymatgen Structure.

    In disordered structures, a single site may be occupied by multiple elements with partial occupancies.
    This function collects all unique element symbols across all sites.

    Args:
        structure (Structure): The input structure, possibly containing site disorder.

    Returns:
        set: A set of element symbols (e.g., {'Li', 'Fe', 'P', 'O'})
    """
    elements = set()
    for site in structure:
        # site.species is a Composition object representing the elements and their occupancies at this site
        for element in site.species:
            elements.add(
                element.symbol
            )  # Use element symbol (e.g., 'Fe') to avoid isotopes or oxidation states
    return elements
