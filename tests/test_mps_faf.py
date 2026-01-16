#!/usr/bin/env python3
"""
Test MPS-based Fermionic Anti-Flatness calculation
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ff.ff_random_states import quimb_random_mps, quimb_zero_state
from ff.ff_distance_measures import FAF


def test_mps_faf():
    """Test FAF calculation with random MPS states."""
    
    n_sites = 10
    max_bond_dims = [2, 10, 25, 50]
    
    print(f"Testing FAF with {n_sites}-qubit MPS states")
    
    # Zero state
    zero_state = quimb_zero_state(n_sites)
    faf_zero = FAF(zero_state)
    print(f"Zero state FAF: {faf_zero:.4f}")
    
    # Random MPS states with different bond dimensions
    for max_bond in max_bond_dims:
        random_mps = quimb_random_mps(n_sites, max_bond)
        faf_value = FAF(random_mps, k=2)
        print(f"Bond dim {max_bond}: FAF = {faf_value:.4f}")

if __name__ == "__main__":
    test_mps_faf()