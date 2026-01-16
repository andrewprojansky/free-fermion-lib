"""
Fermion to Qubit Encodings and Symbolic Pauli Representations

This module contains custom built in a variety of fermion to qubit encodings
and updates the Pauli/spin support with the flexibility to represent Pauli
operators purely symbolically, or through various efficient decompositions. 

Built In Encodings:
 - Jordan-Wigner 
 - Bravyi-Kitaev
 - Ternary Tree
 - Serpinski Tree
 - Balanced Jordan-Wigner 
 - 1-"local"

The main utility is in the ability to do different encodings within the Fermionic 
Anti Flatness calculation. All encodings are written symbolically, though can be 
made into a callable list of matrices stored as separate matrices 

This relies on using quimb as a state simulator and being able to elevate 
the symbolic matrices to properly indexed tensors

Copyright 2025 andrewprojansky@gmail.com
"""

from unittest import result
from numpy import ceil
import numpy as np

def Jordan_Wigner_encoding(num_qubits):
    """Generate the Jordan-Wigner encoding for a given number of qubits."""
    encoding = []
    for i in range(num_qubits):
        JW_right_core = "Z" * i
        JW_left_core = "I" * int(num_qubits - i - 1)
        
        encoding.append(JW_right_core + "X" + JW_left_core)
        encoding.append(JW_right_core + "Y" + JW_left_core)
    return encoding

def Bravyi_Kitaev_encoding(num_qubits):
    """Generate the Bravyi-Kitaev encoding for a given number of qubits."""
    encoding = []
    for i in range(num_qubits):
        pass
    return encoding

def Ternary_Tree_encoding(num_qubits):
    """Generate the Ternary Tree encoding for a given number of qubits."""
    encoding = []
    for i in range(num_qubits):
        pass
    return encoding

def Serpinski_Tree_encoding(num_qubits):
    """Generate the Serpinski Tree encoding for a given number of qubits."""
    encoding = []
    for i in range(num_qubits):
        pass
    return encoding

def Balanced_Jordan_Wigner_encoding(num_qubits):
    """Generate the Balanced Jordan-Wigner encoding for a given number of qubits.
    
    Based off of CZ conjugated encoding; just listing X majoranas, no Y, 
    
    X I Z I Z I Z ...
    Z X I Z I Z I ...
    I Z X I Z I Z ... 
    Z I Z X I Z I """
    ZI_string = "ZI" * (num_qubits // 2)
    IZ_string = "IZ" * (num_qubits // 2)
    encoding = []
    for i in range(num_qubits):
        if i % 2 == 0:
            encoding.append(IZ_string[:i] + "X" + IZ_string[:num_qubits-i-1])
            encoding.append(IZ_string[:i] + "Y" + IZ_string[:num_qubits-i-1])
        else:
            encoding.append(ZI_string[:i] + "X" + IZ_string[:num_qubits-i-1])
            encoding.append(ZI_string[:i] + "Y" + IZ_string[:num_qubits-i-1])
    return encoding

def One_Local_encoding(num_qubits):
    """Generate the 1-"local" encoding for a given number of qubits. 
    
    NOTE: This encoding is not 1-1."""
    encoding = []
    for i in range(num_qubits//2):
        left_core_1 = "I"* 2*i
        right_core_1 = "I" * (num_qubits - 2*i - 1)
        left_core_2 = "I" * (2*i + 1)
        right_core_2 = "I" * (num_qubits - 2*i - 2)
        JW_right_core = "Z" * i
        JW_left_core = "I" * int((ceil(num_qubits/2)) - i - 1)
        
        encoding.append(left_core_1 + "X" + right_core_1 + JW_right_core + "X" + JW_left_core)
        encoding.append(left_core_1 + "Y" + right_core_1 + JW_right_core + "X" + JW_left_core)
        encoding.append(left_core_1 + "Z" + right_core_1 + JW_right_core + "X" + JW_left_core)
        encoding.append(left_core_2 + "X" + right_core_2 + JW_right_core + "Y" + JW_left_core)
        encoding.append(left_core_2 + "Y" + right_core_2 + JW_right_core + "Y" + JW_left_core)
        encoding.append(left_core_2 + "Z" + right_core_2 + JW_right_core + "Y" + JW_left_core)
    return encoding

def symbolic_to_matrix(symbolic_Pauli):
    """Convert a symbolic Pauli string to its corresponding matrix representation."""
    pauli_matrices = {
        'I': [[1, 0], [0, 1]],
        'X': [[0, 1], [1, 0]],
        'Y': [[0, -1j], [1j, 0]],
        'Z': [[1, 0], [0, -1]]
    }
    mat_list = [[pauli_matrices[char] for char in symbolic_pauli] for symbolic_pauli in symbolic_Pauli]
    return mat_list

def kpa(symbolic_pauli):
    """Take the Kronecker product of all 2x2 complex matrices in the list.
    
    Args:
        symbolic_pauli: List of 2x2 complex matrices
        
    Returns:
        numpy.ndarray: The Kronecker product of all matrices
    """
    if not symbolic_pauli:
        raise ValueError("Input list cannot be empty")
    
    result = np.array(symbolic_pauli[0])
    for matrix in symbolic_pauli[1:]:
        result = np.kron(result, np.array(matrix))
    
    return result

def multiply_paulis(p1, p2, phase=False):
    """Multiply two single-qubit Pauli operators.

    Args:
        p1 (str): First Pauli operator ('I', 'X', 'Y', or 'Z')
        p2 (str): Second Pauli operator ('I', 'X', 'Y', or 'Z')
        phase (bool): Whether to return the phase factor

    Returns:
        tuple: (phase, result) where phase is 1, -1, 1j, or -1j and result is the Pauli string
    """
    # Identity cases
    if phase==False:
        if p1 == 'I':
            return p2
        if p2 == 'I':
            return p1

        # Same Pauli -> Identity
        if p1 == p2:
            return 'I'
    else:
        if p1 == 'I':
            return 1, p2
        if p2 == 'I':
            return 1,p1

        # Same Pauli -> Identity
        if p1 == p2:
            return 1, 'I'       

    # Different Paulis - need to track phase
    # X*Y = iZ, Y*X = -iZ
    # Y*Z = iX, Z*Y = -iX
    # Z*X = iY, X*Z = -iY

    pauli_mult = {
        ('X', 'Y'): (1j, 'Z'),
        ('Y', 'X'): (-1j, 'Z'),
        ('Y', 'Z'): (1j, 'X'),
        ('Z', 'Y'): (-1j, 'X'),
        ('Z', 'X'): (1j, 'Y'),
        ('X', 'Z'): (-1j, 'Y'),
    }

    if phase==False:
        return pauli_mult[(p1, p2)][1]
    else:
        return pauli_mult[(p1, p2)]

def multiply_symbolic_paulis(pauli1, pauli2, return_phase=False):
    """Multiply two symbolic Pauli strings

    Args:
        pauli1 (str): First symbolic Pauli string
        pauli2 (str): Second symbolic Pauli string
        return_phase (bool): Whether to return the phase factor

    Returns:
        str or tuple: If return_phase is False, returns the resulting Pauli string.
                     If return_phase is True, returns (total_phase, result) where 
                     total_phase is the accumulated phase and result is the resulting Pauli string
    """
    if len(pauli1) != len(pauli2):
        raise ValueError("Pauli strings must be of the same length")

    result = ""
    total_phase = 1

    for p1, p2 in zip(pauli1, pauli2):
        if return_phase:
            phase, pauli_result = multiply_paulis(p1, p2, phase=True)
            total_phase *= phase
            result += pauli_result
        else:
            result += multiply_paulis(p1, p2, phase=False)
    
    if return_phase:
        return total_phase, result
    else:
        return result

'''
TO QUIMB operators now from symbolic please
'''

#print(Jordan_Wigner_encoding(4))
#print(One_Local_encoding(4))
#print(Balanced_Jordan_Wigner_encoding(4))
#print(symbolic_to_matrix(Jordan_Wigner_encoding(2)))
#print(kpa((symbolic_to_matrix(Jordan_Wigner_encoding(2)))[0]))

#print(jordan_wigner_majoranas(2))