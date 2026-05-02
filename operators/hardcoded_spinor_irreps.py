"""Auto-generated hardcoded spinor irrep matrices.

Built by scripts/build_hardcoded_spinor_irreps_from_cubic.py using
operators.cubic_rotations.iter_spinor_irrep_matrix_blocks (Oh via
get_spinor_irrep_matrix; other little groups via spin-j extraction).
Do not hand-edit unless you know what you are doing.

Rotation keys are condensed symbols (E, C2x, …, I_C4zi) matching
operators.cubic_rotations._POINT_GROUP_NAME_TO_ROTATION; legacy repr() keys are
still accepted on load.
"""

from sympy import *  # noqa: F401,F403

HARD_CODED_SPINOR_IRREP_STR_MATRICES = {
    ('Oh', 'G1g'): {
        'C2a': [['0', 'sqrt(2)*(1 + I)/2'], ['sqrt(2)*(-1 + I)/2', '0']],
        'C2b': [['0', 'sqrt(2)*(-1 + I)/2'], ['sqrt(2)*(1 + I)/2', '0']],
        'C2c': [['sqrt(2)*I/2', 'sqrt(2)*I/2'], ['sqrt(2)*I/2', '-sqrt(2)*I/2']],
        'C2d': [['sqrt(2)*I/2', '-sqrt(2)*I/2'], ['-sqrt(2)*I/2', '-sqrt(2)*I/2']],
        'C2e': [['sqrt(2)*I/2', 'sqrt(2)/2'], ['-sqrt(2)/2', '-sqrt(2)*I/2']],
        'C2f': [['-sqrt(2)*I/2', 'sqrt(2)/2'], ['-sqrt(2)/2', 'sqrt(2)*I/2']],
        'C2x': [['0', 'I'], ['I', '0']],
        'C2y': [['0', '1'], ['-1', '0']],
        'C2z': [['I', '0'], ['0', '-I']],
        'C3a': [['1/2 + I/2', '-1/2 - I/2'], ['1/2 - I/2', '1/2 - I/2']],
        'C3ai': [['1/2 - I/2', '1/2 + I/2'], ['-1/2 + I/2', '1/2 + I/2']],
        'C3b': [['1/2 - I/2', '1/2 - I/2'], ['-1/2 - I/2', '1/2 + I/2']],
        'C3bi': [['1/2 + I/2', '-1/2 + I/2'], ['1/2 + I/2', '1/2 - I/2']],
        'C3c': [['1/2 - I/2', '-1/2 + I/2'], ['1/2 + I/2', '1/2 + I/2']],
        'C3ci': [['1/2 + I/2', '1/2 - I/2'], ['-1/2 - I/2', '1/2 - I/2']],
        'C3d': [['1/2 + I/2', '1/2 + I/2'], ['-1/2 + I/2', '1/2 - I/2']],
        'C3di': [['1/2 - I/2', '-1/2 - I/2'], ['1/2 - I/2', '1/2 + I/2']],
        'C4x': [['sqrt(2)/2', 'sqrt(2)*I/2'], ['sqrt(2)*I/2', 'sqrt(2)/2']],
        'C4xi': [['sqrt(2)/2', '-sqrt(2)*I/2'], ['-sqrt(2)*I/2', 'sqrt(2)/2']],
        'C4y': [['sqrt(2)/2', 'sqrt(2)/2'], ['-sqrt(2)/2', 'sqrt(2)/2']],
        'C4yi': [['sqrt(2)/2', '-sqrt(2)/2'], ['sqrt(2)/2', 'sqrt(2)/2']],
        'C4z': [['sqrt(2)*(1 + I)/2', '0'], ['0', 'sqrt(2)*(1 - I)/2']],
        'C4zi': [['sqrt(2)*(1 - I)/2', '0'], ['0', 'sqrt(2)*(1 + I)/2']],
        'E': [['1', '0'], ['0', '1']],
        'I_C2a': [['0', 'sqrt(2)*(1 + I)/2'], ['sqrt(2)*(-1 + I)/2', '0']],
        'I_C2b': [['0', 'sqrt(2)*(-1 + I)/2'], ['sqrt(2)*(1 + I)/2', '0']],
        'I_C2c': [['sqrt(2)*I/2', 'sqrt(2)*I/2'], ['sqrt(2)*I/2', '-sqrt(2)*I/2']],
        'I_C2d': [['sqrt(2)*I/2', '-sqrt(2)*I/2'], ['-sqrt(2)*I/2', '-sqrt(2)*I/2']],
        'I_C2e': [['sqrt(2)*I/2', 'sqrt(2)/2'], ['-sqrt(2)/2', '-sqrt(2)*I/2']],
        'I_C2f': [['-sqrt(2)*I/2', 'sqrt(2)/2'], ['-sqrt(2)/2', 'sqrt(2)*I/2']],
        'I_C2x': [['0', 'I'], ['I', '0']],
        'I_C2y': [['0', '1'], ['-1', '0']],
        'I_C2z': [['I', '0'], ['0', '-I']],
        'I_C3a': [['1/2 + I/2', '-1/2 - I/2'], ['1/2 - I/2', '1/2 - I/2']],
        'I_C3ai': [['1/2 - I/2', '1/2 + I/2'], ['-1/2 + I/2', '1/2 + I/2']],
        'I_C3b': [['1/2 - I/2', '1/2 - I/2'], ['-1/2 - I/2', '1/2 + I/2']],
        'I_C3bi': [['1/2 + I/2', '-1/2 + I/2'], ['1/2 + I/2', '1/2 - I/2']],
        'I_C3c': [['1/2 - I/2', '-1/2 + I/2'], ['1/2 + I/2', '1/2 + I/2']],
        'I_C3ci': [['1/2 + I/2', '1/2 - I/2'], ['-1/2 - I/2', '1/2 - I/2']],
        'I_C3d': [['1/2 + I/2', '1/2 + I/2'], ['-1/2 + I/2', '1/2 - I/2']],
        'I_C3di': [['1/2 - I/2', '-1/2 - I/2'], ['1/2 - I/2', '1/2 + I/2']],
        'I_C4x': [['sqrt(2)/2', 'sqrt(2)*I/2'], ['sqrt(2)*I/2', 'sqrt(2)/2']],
        'I_C4xi': [['sqrt(2)/2', '-sqrt(2)*I/2'], ['-sqrt(2)*I/2', 'sqrt(2)/2']],
        'I_C4y': [['sqrt(2)/2', 'sqrt(2)/2'], ['-sqrt(2)/2', 'sqrt(2)/2']],
        'I_C4yi': [['sqrt(2)/2', '-sqrt(2)/2'], ['sqrt(2)/2', 'sqrt(2)/2']],
        'I_C4z': [['sqrt(2)*(1 + I)/2', '0'], ['0', 'sqrt(2)*(1 - I)/2']],
        'I_C4zi': [['sqrt(2)*(1 - I)/2', '0'], ['0', 'sqrt(2)*(1 + I)/2']],
        'Is': [['1', '0'], ['0', '1']],
    },
}
