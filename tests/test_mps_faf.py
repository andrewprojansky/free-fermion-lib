#!/usr/bin/env python3
"""
Test MPS-based Fermionic Anti-Flatness calculation
"""

import numpy as np
import sys
import os
import quimb.tensor as qtn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ff.ff_random_states import quimb_random_mps, quimb_zero_state, apply_random_matchgate_brickwork
from ff.ff_distance_measures import FAF
from ff.ff_encodings import One_Local_encoding, Ternary_Tree_encoding


def test_mps_faf():
    """Test FAF calculation with random MPS states."""
    
    n_sites = 10
    max_bond_dims = [2, 10, 25, 50]
    
    print(f"Testing FAF with {n_sites}-qubit MPS states")
    
    # Zero state
    zero_state = quimb_zero_state(n_sites)
    faf_zero = FAF(zero_state, encoding = "balanced")
    print(f"Zero state FAF: {faf_zero:.4f}")
    
    # Random MPS states with different bond dimensions
    for max_bond in max_bond_dims:
        random_mps = quimb_random_mps(n_sites, max_bond)
        faf_value = FAF(random_mps, k=2, encoding = "1-local")
        print(f"Bond dim {max_bond}: FAF = {faf_value:.4f}")

def test_values_correct():
    """Check FAF values against known results.
    
    1. Gaussian State should have FAF 0 
    2. 4 qubit entangled should have FAF from paper 
    3. State with non-zero FAF should remain invariant under FF rotation"""
    n_sites = 6
    #1. Gaussian states should have FAF = 0 under JW
    zero_state = quimb_zero_state(n_sites)
    apply_random_matchgate_brickwork(zero_state, depth=10)
    faf_zero = FAF(zero_state, encoding = "Jordan-Wigner")
    print(f"FAF: {faf_zero}")
    #CHCEK MARK! 

    #2. Specific magic state has specific FAF via JW encoding 
    print("\nTest 2: 4-qubit entangled state via circuit construction")
    
    # Random angle for ZZ-rotation
    theta = np.random.rand() * 2 * np.pi
    print(f"Using ZZ-rotation angle θ = {theta:.4f}")
    
    # 1. Initialize 4-qubit zero state
    mps_state = quimb_zero_state(4)

    matchgate_H = np.array([
        [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)],      # |00⟩ -> |00⟩
        [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],   # |01⟩ -> (|01⟩ + |10⟩)/√2
        [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],  # |10⟩ -> (|01⟩ - |10⟩)/√2  
        [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]       # |11⟩ -> |11⟩
    ])
    
    # Apply to qubits (0,1) and (2,3)
    mps_state.gate_split(matchgate_H, (0, 1), inplace=True)
    mps_state.gate_split(matchgate_H, (2, 3), inplace=True)
    # 3. Apply ZZ-rotation between qubits 1 and 2 (middle qubits)
    # ZZ rotation: exp(-i*theta/2 * Z⊗Z)
    zz_rotation = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(-1j * theta)]
    ])
    mps_state.gate_split(zz_rotation, (1, 2), inplace=True)
    #print the dense state to test 
    faf_comp = FAF(mps_state, k=2, encoding="Jordan-Wigner")
    print(f"FAF Comp: {faf_comp}, FAF Expected: { 4*(1 - np.cos(theta/2)**4):.6f}")

    #3. Invariance under free fermion rotation via JW
    random_mps = quimb_random_mps(n_sites, 10)
    faf_before = FAF(random_mps, encoding = "Jordan-Wigner")
    apply_random_matchgate_brickwork(random_mps, depth=10)
    faf_after = FAF(random_mps, encoding = "Jordan-Wigner")
    print(f"FAF before: {faf_before}, FAF after: {faf_after}")
    #CHECK! 

def test_encoding():

    n_sites = 8
    print(Ternary_Tree_encoding(n_sites))

if __name__ == "__main__":
    test_encoding()