"""
Handler for computing CrystalNNFingerprint on disordered structures using weighted average approach.
This script doesn't modify the original fingerprint.py but provides wrapper functions.
"""

import numpy as np
import itertools
from pymatgen.core import Structure, Site
from collections import defaultdict
import warnings


def create_ordered_approximation(disordered_structure):
    from dmflow.data.disorder_utils import adjust_to_sum_one

    """
    Create a fully ordered approximate structure from a disordered structure via random sampling.

    This function goes through each site in the disordered structure. If the site is already ordered,
    it keeps the atom as-is. For disordered sites (with mixed species), it randomly selects one 
    species based on their occupancy probabilities.

    Args:
        disordered_structure (Structure): A disordered structure object from pymatgen.

    Returns:
        Structure: A randomly generated, fully ordered structure.
    """
    new_species = []
    for site in disordered_structure:
        # Keep the species if the site is already ordered
        if site.is_ordered:
            new_species.append(site.specie)
        else:
            # For disordered sites, randomly select an element based on occupancy probabilities
            elements = list(site.species.keys())
            probabilities = list(site.species.values())
            probabilities = adjust_to_sum_one(probabilities)
            if len(probabilities) == 1:
                chosen_element = elements[0]
            else:
                chosen_element = np.random.choice(elements, p=probabilities)
            new_species.append(chosen_element)

    # Construct a new ordered structure using the same lattice and fractional coordinates
    return Structure(
        disordered_structure.lattice, new_species, disordered_structure.frac_coords
    )


def get_fingerprints_by_ensemble_averaging(structure, featurizer, n_samples=10):
    """
    Compute site-wise crystal fingerprints for a (possibly disordered) structure using ensemble averaging.

    This method handles disordered structures by generating multiple random ordered approximations,
    computing fingerprints for each, and then averaging them across samples to produce robust,
    probabilistically informed fingerprints.

    Args:
        structure (Structure): A pymatgen Structure object (can be disordered).
        featurizer (CrystalNNFingerprint): An initialized featurization tool (e.g., CrystalNNFingerprint).
        n_samples (int): Number of random ordered structures to generate for averaging. Higher values
                         improve stability but increase computation time.

    Returns:
        list: A list of averaged fingerprint vectors, one for each site in the structure.
              Each fingerprint is a list of floats (length depends on featurizer, e.g., 61 for CrystalNNFP).
    """
    # If the structure is already ordered, compute fingerprints directly
    if structure.is_ordered:
        return [featurizer.featurize(structure, i) for i in range(len(structure))]

    all_samples_fps = []
    for _ in range(n_samples):
        # create a random ordered approximation of the disordered structure
        ordered_approx_structure = create_ordered_approximation(structure)

        # Compute fingerprints for all sites in this ordered approximation
        ## [n_sites, n_features (61)]
        sample_fps = [
            featurizer.featurize(ordered_approx_structure, i)
            for i in range(len(ordered_approx_structure))
        ]
        all_samples_fps.append(sample_fps)

    # Convert to numpy array; missing/failed values should become NaN (if any)
    all_samples_fps = np.array(all_samples_fps, dtype=float)
    # Average over the sample dimension (axis=0), ignoring NaN values
    avg_fps = np.nanmean(all_samples_fps, axis=0)

    return avg_fps.tolist()
