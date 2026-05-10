# spinor irrep matrices for each little group and irrep.

from sympy import Matrix, I, sqrt, exp, pi, Rational

HARD_CODED_SPINOR_IRREP_MATRICES = {
    ('C2v', 'G'): {
        'C2e': Matrix([
            [sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, -sqrt(2)*I/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2f': Matrix([
            [-sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)*I/2],
        ]),
        'I_C2x': Matrix([
            [0, I],
            [I, 0],
        ]),
    },
    ('C3v', 'F1'): {
        'C3d': Matrix([
            [-1],
        ]),
        'C3di': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [I],
        ]),
        'I_C2d': Matrix([
            [I],
        ]),
        'I_C2f': Matrix([
            [I],
        ]),
    },
    ('C3v', 'F2'): {
        'C3d': Matrix([
            [-1],
        ]),
        'C3di': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [-I],
        ]),
        'I_C2d': Matrix([
            [-I],
        ]),
        'I_C2f': Matrix([
            [-I],
        ]),
    },
    ('C3v', 'G'): {
        'C3d': Matrix([
            [Rational(1, 2) + I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(1, 2) - I/2],
        ]),
        'C3di': Matrix([
            [Rational(1, 2) - I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(1, 2) + I/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2b': Matrix([
            [0, sqrt(2)*(-1 + I)/2],
            [sqrt(2)*(1 + I)/2, 0],
        ]),
        'I_C2d': Matrix([
            [sqrt(2)*I/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, -sqrt(2)*I/2],
        ]),
        'I_C2f': Matrix([
            [-sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)*I/2],
        ]),
    },
    ('C4v', 'G1'): {
        'C2z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(1 - I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(1 - I)/2, 0],
            [0, sqrt(2)*(1 + I)/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [0, sqrt(2)*(1 + I)/2],
            [sqrt(2)*(-1 + I)/2, 0],
        ]),
        'I_C2b': Matrix([
            [0, sqrt(2)*(-1 + I)/2],
            [sqrt(2)*(1 + I)/2, 0],
        ]),
        'I_C2x': Matrix([
            [0, I],
            [I, 0],
        ]),
        'I_C2y': Matrix([
            [0, 1],
            [-1, 0],
        ]),
    },
    ('C4v', 'G2'): {
        'C2z': Matrix([
            [-I, 0],
            [0, I],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(-1 - I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(-1 - I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [0, sqrt(2)*(-1 - I)/2],
            [sqrt(2)*(1 - I)/2, 0],
        ]),
        'I_C2b': Matrix([
            [0, sqrt(2)*(1 - I)/2],
            [sqrt(2)*(-1 - I)/2, 0],
        ]),
        'I_C2x': Matrix([
            [0, I],
            [I, 0],
        ]),
        'I_C2y': Matrix([
            [0, 1],
            [-1, 0],
        ]),
    },
    ('CS', 'F1'): {
        'E': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [I],
        ]),
    },
    ('CS', 'F2'): {
        'E': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [-I],
        ]),
    },
    ('Oh', 'G1g'): {
        'C2a': Matrix([
            [0, sqrt(2)*(1 + I)/2],
            [sqrt(2)*(-1 + I)/2, 0],
        ]),
        'C2b': Matrix([
            [0, sqrt(2)*(-1 + I)/2],
            [sqrt(2)*(1 + I)/2, 0],
        ]),
        'C2c': Matrix([
            [sqrt(2)*I/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, -sqrt(2)*I/2],
        ]),
        'C2d': Matrix([
            [sqrt(2)*I/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, -sqrt(2)*I/2],
        ]),
        'C2e': Matrix([
            [sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, -sqrt(2)*I/2],
        ]),
        'C2f': Matrix([
            [-sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)*I/2],
        ]),
        'C2x': Matrix([
            [0, I],
            [I, 0],
        ]),
        'C2y': Matrix([
            [0, 1],
            [-1, 0],
        ]),
        'C2z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'C3a': Matrix([
            [Rational(1, 2) + I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(1, 2) - I/2],
        ]),
        'C3ai': Matrix([
            [Rational(1, 2) - I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(1, 2) + I/2],
        ]),
        'C3b': Matrix([
            [Rational(1, 2) - I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(1, 2) + I/2],
        ]),
        'C3bi': Matrix([
            [Rational(1, 2) + I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(1, 2) - I/2],
        ]),
        'C3c': Matrix([
            [Rational(1, 2) - I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(1, 2) + I/2],
        ]),
        'C3ci': Matrix([
            [Rational(1, 2) + I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(1, 2) - I/2],
        ]),
        'C3d': Matrix([
            [Rational(1, 2) + I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(1, 2) - I/2],
        ]),
        'C3di': Matrix([
            [Rational(1, 2) - I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(1, 2) + I/2],
        ]),
        'C4x': Matrix([
            [sqrt(2)/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, sqrt(2)/2],
        ]),
        'C4xi': Matrix([
            [sqrt(2)/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, sqrt(2)/2],
        ]),
        'C4y': Matrix([
            [sqrt(2)/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)/2],
        ]),
        'C4yi': Matrix([
            [sqrt(2)/2, -sqrt(2)/2],
            [sqrt(2)/2, sqrt(2)/2],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(1 - I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(1 - I)/2, 0],
            [0, sqrt(2)*(1 + I)/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [0, sqrt(2)*(1 + I)/2],
            [sqrt(2)*(-1 + I)/2, 0],
        ]),
        'I_C2b': Matrix([
            [0, sqrt(2)*(-1 + I)/2],
            [sqrt(2)*(1 + I)/2, 0],
        ]),
        'I_C2c': Matrix([
            [sqrt(2)*I/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, -sqrt(2)*I/2],
        ]),
        'I_C2d': Matrix([
            [sqrt(2)*I/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, -sqrt(2)*I/2],
        ]),
        'I_C2e': Matrix([
            [sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, -sqrt(2)*I/2],
        ]),
        'I_C2f': Matrix([
            [-sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)*I/2],
        ]),
        'I_C2x': Matrix([
            [0, I],
            [I, 0],
        ]),
        'I_C2y': Matrix([
            [0, 1],
            [-1, 0],
        ]),
        'I_C2z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'I_C3a': Matrix([
            [Rational(1, 2) + I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(1, 2) - I/2],
        ]),
        'I_C3ai': Matrix([
            [Rational(1, 2) - I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(1, 2) + I/2],
        ]),
        'I_C3b': Matrix([
            [Rational(1, 2) - I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(1, 2) + I/2],
        ]),
        'I_C3bi': Matrix([
            [Rational(1, 2) + I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(1, 2) - I/2],
        ]),
        'I_C3c': Matrix([
            [Rational(1, 2) - I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(1, 2) + I/2],
        ]),
        'I_C3ci': Matrix([
            [Rational(1, 2) + I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(1, 2) - I/2],
        ]),
        'I_C3d': Matrix([
            [Rational(1, 2) + I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(1, 2) - I/2],
        ]),
        'I_C3di': Matrix([
            [Rational(1, 2) - I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(1, 2) + I/2],
        ]),
        'I_C4x': Matrix([
            [sqrt(2)/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, sqrt(2)/2],
        ]),
        'I_C4xi': Matrix([
            [sqrt(2)/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, sqrt(2)/2],
        ]),
        'I_C4y': Matrix([
            [sqrt(2)/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)/2],
        ]),
        'I_C4yi': Matrix([
            [sqrt(2)/2, -sqrt(2)/2],
            [sqrt(2)/2, sqrt(2)/2],
        ]),
        'I_C4z': Matrix([
            [sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(1 - I)/2],
        ]),
        'I_C4zi': Matrix([
            [sqrt(2)*(1 - I)/2, 0],
            [0, sqrt(2)*(1 + I)/2],
        ]),
        'Is': Matrix([
            [1, 0],
            [0, 1],
        ]),
    },
    ('Oh', 'G1u'): {
        'C2a': Matrix([
            [0, sqrt(2)*(1 + I)/2],
            [sqrt(2)*(-1 + I)/2, 0],
        ]),
        'C2b': Matrix([
            [0, sqrt(2)*(-1 + I)/2],
            [sqrt(2)*(1 + I)/2, 0],
        ]),
        'C2c': Matrix([
            [sqrt(2)*I/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, -sqrt(2)*I/2],
        ]),
        'C2d': Matrix([
            [sqrt(2)*I/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, -sqrt(2)*I/2],
        ]),
        'C2e': Matrix([
            [sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, -sqrt(2)*I/2],
        ]),
        'C2f': Matrix([
            [-sqrt(2)*I/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)*I/2],
        ]),
        'C2x': Matrix([
            [0, I],
            [I, 0],
        ]),
        'C2y': Matrix([
            [0, 1],
            [-1, 0],
        ]),
        'C2z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'C3a': Matrix([
            [Rational(1, 2) + I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(1, 2) - I/2],
        ]),
        'C3ai': Matrix([
            [Rational(1, 2) - I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(1, 2) + I/2],
        ]),
        'C3b': Matrix([
            [Rational(1, 2) - I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(1, 2) + I/2],
        ]),
        'C3bi': Matrix([
            [Rational(1, 2) + I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(1, 2) - I/2],
        ]),
        'C3c': Matrix([
            [Rational(1, 2) - I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(1, 2) + I/2],
        ]),
        'C3ci': Matrix([
            [Rational(1, 2) + I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(1, 2) - I/2],
        ]),
        'C3d': Matrix([
            [Rational(1, 2) + I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(1, 2) - I/2],
        ]),
        'C3di': Matrix([
            [Rational(1, 2) - I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(1, 2) + I/2],
        ]),
        'C4x': Matrix([
            [sqrt(2)/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, sqrt(2)/2],
        ]),
        'C4xi': Matrix([
            [sqrt(2)/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, sqrt(2)/2],
        ]),
        'C4y': Matrix([
            [sqrt(2)/2, sqrt(2)/2],
            [-sqrt(2)/2, sqrt(2)/2],
        ]),
        'C4yi': Matrix([
            [sqrt(2)/2, -sqrt(2)/2],
            [sqrt(2)/2, sqrt(2)/2],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(1 - I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(1 - I)/2, 0],
            [0, sqrt(2)*(1 + I)/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [0, sqrt(2)*(-1 - I)/2],
            [sqrt(2)*(1 - I)/2, 0],
        ]),
        'I_C2b': Matrix([
            [0, sqrt(2)*(1 - I)/2],
            [sqrt(2)*(-1 - I)/2, 0],
        ]),
        'I_C2c': Matrix([
            [-sqrt(2)*I/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, sqrt(2)*I/2],
        ]),
        'I_C2d': Matrix([
            [-sqrt(2)*I/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, sqrt(2)*I/2],
        ]),
        'I_C2e': Matrix([
            [-sqrt(2)*I/2, -sqrt(2)/2],
            [sqrt(2)/2, sqrt(2)*I/2],
        ]),
        'I_C2f': Matrix([
            [sqrt(2)*I/2, -sqrt(2)/2],
            [sqrt(2)/2, -sqrt(2)*I/2],
        ]),
        'I_C2x': Matrix([
            [0, -I],
            [-I, 0],
        ]),
        'I_C2y': Matrix([
            [0, -1],
            [1, 0],
        ]),
        'I_C2z': Matrix([
            [-I, 0],
            [0, I],
        ]),
        'I_C3a': Matrix([
            [Rational(-1, 2) - I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(-1, 2) + I/2],
        ]),
        'I_C3ai': Matrix([
            [Rational(-1, 2) + I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(-1, 2) - I/2],
        ]),
        'I_C3b': Matrix([
            [Rational(-1, 2) + I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(-1, 2) - I/2],
        ]),
        'I_C3bi': Matrix([
            [Rational(-1, 2) - I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(-1, 2) + I/2],
        ]),
        'I_C3c': Matrix([
            [Rational(-1, 2) + I/2, Rational(1, 2) - I/2],
            [Rational(-1, 2) - I/2, Rational(-1, 2) - I/2],
        ]),
        'I_C3ci': Matrix([
            [Rational(-1, 2) - I/2, Rational(-1, 2) + I/2],
            [Rational(1, 2) + I/2, Rational(-1, 2) + I/2],
        ]),
        'I_C3d': Matrix([
            [Rational(-1, 2) - I/2, Rational(-1, 2) - I/2],
            [Rational(1, 2) - I/2, Rational(-1, 2) + I/2],
        ]),
        'I_C3di': Matrix([
            [Rational(-1, 2) + I/2, Rational(1, 2) + I/2],
            [Rational(-1, 2) + I/2, Rational(-1, 2) - I/2],
        ]),
        'I_C4x': Matrix([
            [-sqrt(2)/2, -sqrt(2)*I/2],
            [-sqrt(2)*I/2, -sqrt(2)/2],
        ]),
        'I_C4xi': Matrix([
            [-sqrt(2)/2, sqrt(2)*I/2],
            [sqrt(2)*I/2, -sqrt(2)/2],
        ]),
        'I_C4y': Matrix([
            [-sqrt(2)/2, -sqrt(2)/2],
            [sqrt(2)/2, -sqrt(2)/2],
        ]),
        'I_C4yi': Matrix([
            [-sqrt(2)/2, sqrt(2)/2],
            [-sqrt(2)/2, -sqrt(2)/2],
        ]),
        'I_C4z': Matrix([
            [sqrt(2)*(-1 - I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2],
        ]),
        'I_C4zi': Matrix([
            [sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(-1 - I)/2],
        ]),
        'Is': Matrix([
            [-1, 0],
            [0, -1],
        ]),
    },
    ('Oh', 'G2g'): {
        'C2a': Matrix([
            [0, sqrt(10)*(1 - I)/2],
            [sqrt(10)*(-1 - I)/10, 0],
        ]),
        'C2b': Matrix([
            [0, sqrt(10)*(-1 - I)/2],
            [sqrt(10)*(1 - I)/10, 0],
        ]),
        'C2c': Matrix([
            [-sqrt(2)*I/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, sqrt(2)*I/2],
        ]),
        'C2d': Matrix([
            [-sqrt(2)*I/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, sqrt(2)*I/2],
        ]),
        'C2e': Matrix([
            [-sqrt(2)*I/2, sqrt(10)/2],
            [-sqrt(10)/10, sqrt(2)*I/2],
        ]),
        'C2f': Matrix([
            [sqrt(2)*I/2, sqrt(10)/2],
            [-sqrt(10)/10, -sqrt(2)*I/2],
        ]),
        'C2x': Matrix([
            [0, sqrt(5)*I],
            [sqrt(5)*I/5, 0],
        ]),
        'C2y': Matrix([
            [0, -sqrt(5)],
            [sqrt(5)/5, 0],
        ]),
        'C2z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'C3a': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(1, 2) - I/2],
        ]),
        'C3ai': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(1, 2) + I/2],
        ]),
        'C3b': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(1, 2) + I/2],
        ]),
        'C3bi': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(1, 2) - I/2],
        ]),
        'C3c': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(1, 2) + I/2],
        ]),
        'C3ci': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(1, 2) - I/2],
        ]),
        'C3d': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(1, 2) - I/2],
        ]),
        'C3di': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(1, 2) + I/2],
        ]),
        'C4x': Matrix([
            [-sqrt(2)/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, -sqrt(2)/2],
        ]),
        'C4xi': Matrix([
            [-sqrt(2)/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, -sqrt(2)/2],
        ]),
        'C4y': Matrix([
            [-sqrt(2)/2, sqrt(10)/2],
            [-sqrt(10)/10, -sqrt(2)/2],
        ]),
        'C4yi': Matrix([
            [-sqrt(2)/2, -sqrt(10)/2],
            [sqrt(10)/10, -sqrt(2)/2],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(-1 - I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(-1 - I)/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [0, sqrt(10)*(1 - I)/2],
            [sqrt(10)*(-1 - I)/10, 0],
        ]),
        'I_C2b': Matrix([
            [0, sqrt(10)*(-1 - I)/2],
            [sqrt(10)*(1 - I)/10, 0],
        ]),
        'I_C2c': Matrix([
            [-sqrt(2)*I/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, sqrt(2)*I/2],
        ]),
        'I_C2d': Matrix([
            [-sqrt(2)*I/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, sqrt(2)*I/2],
        ]),
        'I_C2e': Matrix([
            [-sqrt(2)*I/2, sqrt(10)/2],
            [-sqrt(10)/10, sqrt(2)*I/2],
        ]),
        'I_C2f': Matrix([
            [sqrt(2)*I/2, sqrt(10)/2],
            [-sqrt(10)/10, -sqrt(2)*I/2],
        ]),
        'I_C2x': Matrix([
            [0, sqrt(5)*I],
            [sqrt(5)*I/5, 0],
        ]),
        'I_C2y': Matrix([
            [0, -sqrt(5)],
            [sqrt(5)/5, 0],
        ]),
        'I_C2z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'I_C3a': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(1, 2) - I/2],
        ]),
        'I_C3ai': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(1, 2) + I/2],
        ]),
        'I_C3b': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(1, 2) + I/2],
        ]),
        'I_C3bi': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(1, 2) - I/2],
        ]),
        'I_C3c': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(1, 2) + I/2],
        ]),
        'I_C3ci': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(1, 2) - I/2],
        ]),
        'I_C3d': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(1, 2) - I/2],
        ]),
        'I_C3di': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(1, 2) + I/2],
        ]),
        'I_C4x': Matrix([
            [-sqrt(2)/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, -sqrt(2)/2],
        ]),
        'I_C4xi': Matrix([
            [-sqrt(2)/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, -sqrt(2)/2],
        ]),
        'I_C4y': Matrix([
            [-sqrt(2)/2, sqrt(10)/2],
            [-sqrt(10)/10, -sqrt(2)/2],
        ]),
        'I_C4yi': Matrix([
            [-sqrt(2)/2, -sqrt(10)/2],
            [sqrt(10)/10, -sqrt(2)/2],
        ]),
        'I_C4z': Matrix([
            [sqrt(2)*(-1 - I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2],
        ]),
        'I_C4zi': Matrix([
            [sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(-1 - I)/2],
        ]),
        'Is': Matrix([
            [1, 0],
            [0, 1],
        ]),
    },
    ('Oh', 'G2u'): {
        'C2a': Matrix([
            [0, sqrt(10)*(1 - I)/2],
            [sqrt(10)*(-1 - I)/10, 0],
        ]),
        'C2b': Matrix([
            [0, sqrt(10)*(-1 - I)/2],
            [sqrt(10)*(1 - I)/10, 0],
        ]),
        'C2c': Matrix([
            [-sqrt(2)*I/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, sqrt(2)*I/2],
        ]),
        'C2d': Matrix([
            [-sqrt(2)*I/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, sqrt(2)*I/2],
        ]),
        'C2e': Matrix([
            [-sqrt(2)*I/2, sqrt(10)/2],
            [-sqrt(10)/10, sqrt(2)*I/2],
        ]),
        'C2f': Matrix([
            [sqrt(2)*I/2, sqrt(10)/2],
            [-sqrt(10)/10, -sqrt(2)*I/2],
        ]),
        'C2x': Matrix([
            [0, sqrt(5)*I],
            [sqrt(5)*I/5, 0],
        ]),
        'C2y': Matrix([
            [0, -sqrt(5)],
            [sqrt(5)/5, 0],
        ]),
        'C2z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'C3a': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(1, 2) - I/2],
        ]),
        'C3ai': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(1, 2) + I/2],
        ]),
        'C3b': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(1, 2) + I/2],
        ]),
        'C3bi': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(1, 2) - I/2],
        ]),
        'C3c': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(1, 2) + I/2],
        ]),
        'C3ci': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(1, 2) - I/2],
        ]),
        'C3d': Matrix([
            [Rational(1, 2) + I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(1, 2) - I/2],
        ]),
        'C3di': Matrix([
            [Rational(1, 2) - I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(1, 2) + I/2],
        ]),
        'C4x': Matrix([
            [-sqrt(2)/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, -sqrt(2)/2],
        ]),
        'C4xi': Matrix([
            [-sqrt(2)/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, -sqrt(2)/2],
        ]),
        'C4y': Matrix([
            [-sqrt(2)/2, sqrt(10)/2],
            [-sqrt(10)/10, -sqrt(2)/2],
        ]),
        'C4yi': Matrix([
            [-sqrt(2)/2, -sqrt(10)/2],
            [sqrt(10)/10, -sqrt(2)/2],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(-1 - I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(-1 - I)/2],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [0, sqrt(10)*(-1 + I)/2],
            [sqrt(10)*(1 + I)/10, 0],
        ]),
        'I_C2b': Matrix([
            [0, sqrt(10)*(1 + I)/2],
            [sqrt(10)*(-1 + I)/10, 0],
        ]),
        'I_C2c': Matrix([
            [sqrt(2)*I/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, -sqrt(2)*I/2],
        ]),
        'I_C2d': Matrix([
            [sqrt(2)*I/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, -sqrt(2)*I/2],
        ]),
        'I_C2e': Matrix([
            [sqrt(2)*I/2, -sqrt(10)/2],
            [sqrt(10)/10, -sqrt(2)*I/2],
        ]),
        'I_C2f': Matrix([
            [-sqrt(2)*I/2, -sqrt(10)/2],
            [sqrt(10)/10, sqrt(2)*I/2],
        ]),
        'I_C2x': Matrix([
            [0, -sqrt(5)*I],
            [-sqrt(5)*I/5, 0],
        ]),
        'I_C2y': Matrix([
            [0, sqrt(5)],
            [-sqrt(5)/5, 0],
        ]),
        'I_C2z': Matrix([
            [-I, 0],
            [0, I],
        ]),
        'I_C3a': Matrix([
            [Rational(-1, 2) - I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(-1, 2) + I/2],
        ]),
        'I_C3ai': Matrix([
            [Rational(-1, 2) + I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(-1, 2) - I/2],
        ]),
        'I_C3b': Matrix([
            [Rational(-1, 2) + I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(-1, 2) - I/2],
        ]),
        'I_C3bi': Matrix([
            [Rational(-1, 2) - I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(-1, 2) + I/2],
        ]),
        'I_C3c': Matrix([
            [Rational(-1, 2) + I/2, sqrt(5)*(-1 - I)/2],
            [sqrt(5)*(1 - I)/10, Rational(-1, 2) - I/2],
        ]),
        'I_C3ci': Matrix([
            [Rational(-1, 2) - I/2, sqrt(5)*(1 + I)/2],
            [sqrt(5)*(-1 + I)/10, Rational(-1, 2) + I/2],
        ]),
        'I_C3d': Matrix([
            [Rational(-1, 2) - I/2, sqrt(5)*(1 - I)/2],
            [sqrt(5)*(-1 - I)/10, Rational(-1, 2) + I/2],
        ]),
        'I_C3di': Matrix([
            [Rational(-1, 2) + I/2, sqrt(5)*(-1 + I)/2],
            [sqrt(5)*(1 + I)/10, Rational(-1, 2) - I/2],
        ]),
        'I_C4x': Matrix([
            [sqrt(2)/2, sqrt(10)*I/2],
            [sqrt(10)*I/10, sqrt(2)/2],
        ]),
        'I_C4xi': Matrix([
            [sqrt(2)/2, -sqrt(10)*I/2],
            [-sqrt(10)*I/10, sqrt(2)/2],
        ]),
        'I_C4y': Matrix([
            [sqrt(2)/2, -sqrt(10)/2],
            [sqrt(10)/10, sqrt(2)/2],
        ]),
        'I_C4yi': Matrix([
            [sqrt(2)/2, sqrt(10)/2],
            [-sqrt(10)/10, sqrt(2)/2],
        ]),
        'I_C4z': Matrix([
            [sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(1 - I)/2],
        ]),
        'I_C4zi': Matrix([
            [sqrt(2)*(1 - I)/2, 0],
            [0, sqrt(2)*(1 + I)/2],
        ]),
        'Is': Matrix([
            [-1, 0],
            [0, -1],
        ]),
    },
    ('Oh', 'Hg'): {
        'C2a': Matrix([
            [0, 0, 0, sqrt(2)*(-1 - I)/2],
            [0, 0, sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(1 + I)/2, 0, 0],
            [sqrt(2)*(1 - I)/2, 0, 0, 0],
        ]),
        'C2b': Matrix([
            [0, 0, 0, sqrt(2)*(1 - I)/2],
            [0, 0, sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2, 0, 0],
            [sqrt(2)*(-1 - I)/2, 0, 0, 0],
        ]),
        'C2c': Matrix([
            [-sqrt(2)*I/4, sqrt(6)*I/4, -sqrt(6)*I/4, sqrt(2)*I/4],
            [sqrt(6)*I/4, -sqrt(2)*I/4, -sqrt(2)*I/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)*I/4, sqrt(2)*I/4, sqrt(6)*I/4],
            [sqrt(2)*I/4, sqrt(6)*I/4, sqrt(6)*I/4, sqrt(2)*I/4],
        ]),
        'C2d': Matrix([
            [-sqrt(2)*I/4, -sqrt(6)*I/4, -sqrt(6)*I/4, -sqrt(2)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)*I/4, sqrt(2)*I/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, sqrt(2)*I/4, sqrt(2)*I/4, -sqrt(6)*I/4],
            [-sqrt(2)*I/4, sqrt(6)*I/4, -sqrt(6)*I/4, sqrt(2)*I/4],
        ]),
        'C2e': Matrix([
            [-sqrt(2)*I/4, -sqrt(6)/4, sqrt(6)*I/4, sqrt(2)/4],
            [sqrt(6)/4, -sqrt(2)*I/4, sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(6)*I/4, -sqrt(2)/4, sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(2)/4, -sqrt(6)*I/4, sqrt(6)/4, sqrt(2)*I/4],
        ]),
        'C2f': Matrix([
            [sqrt(2)*I/4, -sqrt(6)/4, -sqrt(6)*I/4, sqrt(2)/4],
            [sqrt(6)/4, sqrt(2)*I/4, sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)/4, -sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(2)/4, sqrt(6)*I/4, sqrt(6)/4, -sqrt(2)*I/4],
        ]),
        'C2x': Matrix([
            [0, 0, 0, I],
            [0, 0, I, 0],
            [0, I, 0, 0],
            [I, 0, 0, 0],
        ]),
        'C2y': Matrix([
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ]),
        'C2z': Matrix([
            [-I, 0, 0, 0],
            [0, I, 0, 0],
            [0, 0, -I, 0],
            [0, 0, 0, I],
        ]),
        'C3a': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 - I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3ai': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4],
        ]),
        'C3b': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4],
        ]),
        'C3bi': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 + I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3c': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 + I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4],
        ]),
        'C3ci': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3d': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3di': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 - I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4],
        ]),
        'C4x': Matrix([
            [sqrt(2)/4, -sqrt(6)*I/4, -sqrt(6)/4, sqrt(2)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)/4, -sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(6)/4, -sqrt(2)*I/4, -sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(2)*I/4, -sqrt(6)/4, -sqrt(6)*I/4, sqrt(2)/4],
        ]),
        'C4xi': Matrix([
            [sqrt(2)/4, sqrt(6)*I/4, -sqrt(6)/4, -sqrt(2)*I/4],
            [sqrt(6)*I/4, -sqrt(2)/4, sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(6)/4, sqrt(2)*I/4, -sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(2)*I/4, -sqrt(6)/4, sqrt(6)*I/4, sqrt(2)/4],
        ]),
        'C4y': Matrix([
            [sqrt(2)/4, sqrt(6)/4, sqrt(6)/4, sqrt(2)/4],
            [-sqrt(6)/4, -sqrt(2)/4, sqrt(2)/4, sqrt(6)/4],
            [sqrt(6)/4, -sqrt(2)/4, -sqrt(2)/4, sqrt(6)/4],
            [-sqrt(2)/4, sqrt(6)/4, -sqrt(6)/4, sqrt(2)/4],
        ]),
        'C4yi': Matrix([
            [sqrt(2)/4, -sqrt(6)/4, sqrt(6)/4, -sqrt(2)/4],
            [sqrt(6)/4, -sqrt(2)/4, -sqrt(2)/4, sqrt(6)/4],
            [sqrt(6)/4, sqrt(2)/4, -sqrt(2)/4, -sqrt(6)/4],
            [sqrt(2)/4, sqrt(6)/4, sqrt(6)/4, sqrt(2)/4],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(-1 + I)/2, 0, 0, 0],
            [0, sqrt(2)*(1 + I)/2, 0, 0],
            [0, 0, sqrt(2)*(1 - I)/2, 0],
            [0, 0, 0, sqrt(2)*(-1 - I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(-1 - I)/2, 0, 0, 0],
            [0, sqrt(2)*(1 - I)/2, 0, 0],
            [0, 0, sqrt(2)*(1 + I)/2, 0],
            [0, 0, 0, sqrt(2)*(-1 + I)/2],
        ]),
        'E': Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
        'I_C2a': Matrix([
            [0, 0, 0, sqrt(2)*(-1 - I)/2],
            [0, 0, sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(1 + I)/2, 0, 0],
            [sqrt(2)*(1 - I)/2, 0, 0, 0],
        ]),
        'I_C2b': Matrix([
            [0, 0, 0, sqrt(2)*(1 - I)/2],
            [0, 0, sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2, 0, 0],
            [sqrt(2)*(-1 - I)/2, 0, 0, 0],
        ]),
        'I_C2c': Matrix([
            [-sqrt(2)*I/4, sqrt(6)*I/4, -sqrt(6)*I/4, sqrt(2)*I/4],
            [sqrt(6)*I/4, -sqrt(2)*I/4, -sqrt(2)*I/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)*I/4, sqrt(2)*I/4, sqrt(6)*I/4],
            [sqrt(2)*I/4, sqrt(6)*I/4, sqrt(6)*I/4, sqrt(2)*I/4],
        ]),
        'I_C2d': Matrix([
            [-sqrt(2)*I/4, -sqrt(6)*I/4, -sqrt(6)*I/4, -sqrt(2)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)*I/4, sqrt(2)*I/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, sqrt(2)*I/4, sqrt(2)*I/4, -sqrt(6)*I/4],
            [-sqrt(2)*I/4, sqrt(6)*I/4, -sqrt(6)*I/4, sqrt(2)*I/4],
        ]),
        'I_C2e': Matrix([
            [-sqrt(2)*I/4, -sqrt(6)/4, sqrt(6)*I/4, sqrt(2)/4],
            [sqrt(6)/4, -sqrt(2)*I/4, sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(6)*I/4, -sqrt(2)/4, sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(2)/4, -sqrt(6)*I/4, sqrt(6)/4, sqrt(2)*I/4],
        ]),
        'I_C2f': Matrix([
            [sqrt(2)*I/4, -sqrt(6)/4, -sqrt(6)*I/4, sqrt(2)/4],
            [sqrt(6)/4, sqrt(2)*I/4, sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)/4, -sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(2)/4, sqrt(6)*I/4, sqrt(6)/4, -sqrt(2)*I/4],
        ]),
        'I_C2x': Matrix([
            [0, 0, 0, I],
            [0, 0, I, 0],
            [0, I, 0, 0],
            [I, 0, 0, 0],
        ]),
        'I_C2y': Matrix([
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ]),
        'I_C2z': Matrix([
            [-I, 0, 0, 0],
            [0, I, 0, 0],
            [0, 0, -I, 0],
            [0, 0, 0, I],
        ]),
        'I_C3a': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 - I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4],
        ]),
        'I_C3ai': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4],
        ]),
        'I_C3b': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4],
        ]),
        'I_C3bi': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 + I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4],
        ]),
        'I_C3c': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 + I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4],
        ]),
        'I_C3ci': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4],
        ]),
        'I_C3d': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4],
        ]),
        'I_C3di': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 - I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4],
        ]),
        'I_C4x': Matrix([
            [sqrt(2)/4, -sqrt(6)*I/4, -sqrt(6)/4, sqrt(2)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)/4, -sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(6)/4, -sqrt(2)*I/4, -sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(2)*I/4, -sqrt(6)/4, -sqrt(6)*I/4, sqrt(2)/4],
        ]),
        'I_C4xi': Matrix([
            [sqrt(2)/4, sqrt(6)*I/4, -sqrt(6)/4, -sqrt(2)*I/4],
            [sqrt(6)*I/4, -sqrt(2)/4, sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(6)/4, sqrt(2)*I/4, -sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(2)*I/4, -sqrt(6)/4, sqrt(6)*I/4, sqrt(2)/4],
        ]),
        'I_C4y': Matrix([
            [sqrt(2)/4, sqrt(6)/4, sqrt(6)/4, sqrt(2)/4],
            [-sqrt(6)/4, -sqrt(2)/4, sqrt(2)/4, sqrt(6)/4],
            [sqrt(6)/4, -sqrt(2)/4, -sqrt(2)/4, sqrt(6)/4],
            [-sqrt(2)/4, sqrt(6)/4, -sqrt(6)/4, sqrt(2)/4],
        ]),
        'I_C4yi': Matrix([
            [sqrt(2)/4, -sqrt(6)/4, sqrt(6)/4, -sqrt(2)/4],
            [sqrt(6)/4, -sqrt(2)/4, -sqrt(2)/4, sqrt(6)/4],
            [sqrt(6)/4, sqrt(2)/4, -sqrt(2)/4, -sqrt(6)/4],
            [sqrt(2)/4, sqrt(6)/4, sqrt(6)/4, sqrt(2)/4],
        ]),
        'I_C4z': Matrix([
            [sqrt(2)*(-1 + I)/2, 0, 0, 0],
            [0, sqrt(2)*(1 + I)/2, 0, 0],
            [0, 0, sqrt(2)*(1 - I)/2, 0],
            [0, 0, 0, sqrt(2)*(-1 - I)/2],
        ]),
        'I_C4zi': Matrix([
            [sqrt(2)*(-1 - I)/2, 0, 0, 0],
            [0, sqrt(2)*(1 - I)/2, 0, 0],
            [0, 0, sqrt(2)*(1 + I)/2, 0],
            [0, 0, 0, sqrt(2)*(-1 + I)/2],
        ]),
        'Is': Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
    },
    ('Oh', 'Hu'): {
        'C2a': Matrix([
            [0, 0, 0, sqrt(2)*(-1 - I)/2],
            [0, 0, sqrt(2)*(-1 + I)/2, 0],
            [0, sqrt(2)*(1 + I)/2, 0, 0],
            [sqrt(2)*(1 - I)/2, 0, 0, 0],
        ]),
        'C2b': Matrix([
            [0, 0, 0, sqrt(2)*(1 - I)/2],
            [0, 0, sqrt(2)*(1 + I)/2, 0],
            [0, sqrt(2)*(-1 + I)/2, 0, 0],
            [sqrt(2)*(-1 - I)/2, 0, 0, 0],
        ]),
        'C2c': Matrix([
            [-sqrt(2)*I/4, sqrt(6)*I/4, -sqrt(6)*I/4, sqrt(2)*I/4],
            [sqrt(6)*I/4, -sqrt(2)*I/4, -sqrt(2)*I/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)*I/4, sqrt(2)*I/4, sqrt(6)*I/4],
            [sqrt(2)*I/4, sqrt(6)*I/4, sqrt(6)*I/4, sqrt(2)*I/4],
        ]),
        'C2d': Matrix([
            [-sqrt(2)*I/4, -sqrt(6)*I/4, -sqrt(6)*I/4, -sqrt(2)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)*I/4, sqrt(2)*I/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, sqrt(2)*I/4, sqrt(2)*I/4, -sqrt(6)*I/4],
            [-sqrt(2)*I/4, sqrt(6)*I/4, -sqrt(6)*I/4, sqrt(2)*I/4],
        ]),
        'C2e': Matrix([
            [-sqrt(2)*I/4, -sqrt(6)/4, sqrt(6)*I/4, sqrt(2)/4],
            [sqrt(6)/4, -sqrt(2)*I/4, sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(6)*I/4, -sqrt(2)/4, sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(2)/4, -sqrt(6)*I/4, sqrt(6)/4, sqrt(2)*I/4],
        ]),
        'C2f': Matrix([
            [sqrt(2)*I/4, -sqrt(6)/4, -sqrt(6)*I/4, sqrt(2)/4],
            [sqrt(6)/4, sqrt(2)*I/4, sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)/4, -sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(2)/4, sqrt(6)*I/4, sqrt(6)/4, -sqrt(2)*I/4],
        ]),
        'C2x': Matrix([
            [0, 0, 0, I],
            [0, 0, I, 0],
            [0, I, 0, 0],
            [I, 0, 0, 0],
        ]),
        'C2y': Matrix([
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ]),
        'C2z': Matrix([
            [-I, 0, 0, 0],
            [0, I, 0, 0],
            [0, 0, -I, 0],
            [0, 0, 0, I],
        ]),
        'C3a': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 - I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3ai': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4],
        ]),
        'C3b': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4],
        ]),
        'C3bi': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 + I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3c': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 + I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4],
        ]),
        'C3ci': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3d': Matrix([
            [Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4],
        ]),
        'C3di': Matrix([
            [Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 - I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4],
        ]),
        'C4x': Matrix([
            [sqrt(2)/4, -sqrt(6)*I/4, -sqrt(6)/4, sqrt(2)*I/4],
            [-sqrt(6)*I/4, -sqrt(2)/4, -sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(6)/4, -sqrt(2)*I/4, -sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(2)*I/4, -sqrt(6)/4, -sqrt(6)*I/4, sqrt(2)/4],
        ]),
        'C4xi': Matrix([
            [sqrt(2)/4, sqrt(6)*I/4, -sqrt(6)/4, -sqrt(2)*I/4],
            [sqrt(6)*I/4, -sqrt(2)/4, sqrt(2)*I/4, -sqrt(6)/4],
            [-sqrt(6)/4, sqrt(2)*I/4, -sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(2)*I/4, -sqrt(6)/4, sqrt(6)*I/4, sqrt(2)/4],
        ]),
        'C4y': Matrix([
            [sqrt(2)/4, sqrt(6)/4, sqrt(6)/4, sqrt(2)/4],
            [-sqrt(6)/4, -sqrt(2)/4, sqrt(2)/4, sqrt(6)/4],
            [sqrt(6)/4, -sqrt(2)/4, -sqrt(2)/4, sqrt(6)/4],
            [-sqrt(2)/4, sqrt(6)/4, -sqrt(6)/4, sqrt(2)/4],
        ]),
        'C4yi': Matrix([
            [sqrt(2)/4, -sqrt(6)/4, sqrt(6)/4, -sqrt(2)/4],
            [sqrt(6)/4, -sqrt(2)/4, -sqrt(2)/4, sqrt(6)/4],
            [sqrt(6)/4, sqrt(2)/4, -sqrt(2)/4, -sqrt(6)/4],
            [sqrt(2)/4, sqrt(6)/4, sqrt(6)/4, sqrt(2)/4],
        ]),
        'C4z': Matrix([
            [sqrt(2)*(-1 + I)/2, 0, 0, 0],
            [0, sqrt(2)*(1 + I)/2, 0, 0],
            [0, 0, sqrt(2)*(1 - I)/2, 0],
            [0, 0, 0, sqrt(2)*(-1 - I)/2],
        ]),
        'C4zi': Matrix([
            [sqrt(2)*(-1 - I)/2, 0, 0, 0],
            [0, sqrt(2)*(1 - I)/2, 0, 0],
            [0, 0, sqrt(2)*(1 + I)/2, 0],
            [0, 0, 0, sqrt(2)*(-1 + I)/2],
        ]),
        'E': Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
        'I_C2a': Matrix([
            [0, 0, 0, sqrt(2)*(1 + I)/2],
            [0, 0, sqrt(2)*(1 - I)/2, 0],
            [0, sqrt(2)*(-1 - I)/2, 0, 0],
            [sqrt(2)*(-1 + I)/2, 0, 0, 0],
        ]),
        'I_C2b': Matrix([
            [0, 0, 0, sqrt(2)*(-1 + I)/2],
            [0, 0, sqrt(2)*(-1 - I)/2, 0],
            [0, sqrt(2)*(1 - I)/2, 0, 0],
            [sqrt(2)*(1 + I)/2, 0, 0, 0],
        ]),
        'I_C2c': Matrix([
            [sqrt(2)*I/4, -sqrt(6)*I/4, sqrt(6)*I/4, -sqrt(2)*I/4],
            [-sqrt(6)*I/4, sqrt(2)*I/4, sqrt(2)*I/4, -sqrt(6)*I/4],
            [sqrt(6)*I/4, sqrt(2)*I/4, -sqrt(2)*I/4, -sqrt(6)*I/4],
            [-sqrt(2)*I/4, -sqrt(6)*I/4, -sqrt(6)*I/4, -sqrt(2)*I/4],
        ]),
        'I_C2d': Matrix([
            [sqrt(2)*I/4, sqrt(6)*I/4, sqrt(6)*I/4, sqrt(2)*I/4],
            [sqrt(6)*I/4, sqrt(2)*I/4, -sqrt(2)*I/4, -sqrt(6)*I/4],
            [sqrt(6)*I/4, -sqrt(2)*I/4, -sqrt(2)*I/4, sqrt(6)*I/4],
            [sqrt(2)*I/4, -sqrt(6)*I/4, sqrt(6)*I/4, -sqrt(2)*I/4],
        ]),
        'I_C2e': Matrix([
            [sqrt(2)*I/4, sqrt(6)/4, -sqrt(6)*I/4, -sqrt(2)/4],
            [-sqrt(6)/4, sqrt(2)*I/4, -sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(6)*I/4, sqrt(2)/4, -sqrt(2)*I/4, sqrt(6)/4],
            [sqrt(2)/4, sqrt(6)*I/4, -sqrt(6)/4, -sqrt(2)*I/4],
        ]),
        'I_C2f': Matrix([
            [-sqrt(2)*I/4, sqrt(6)/4, sqrt(6)*I/4, -sqrt(2)/4],
            [-sqrt(6)/4, -sqrt(2)*I/4, -sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(6)*I/4, sqrt(2)/4, sqrt(2)*I/4, sqrt(6)/4],
            [sqrt(2)/4, -sqrt(6)*I/4, -sqrt(6)/4, sqrt(2)*I/4],
        ]),
        'I_C2x': Matrix([
            [0, 0, 0, -I],
            [0, 0, -I, 0],
            [0, -I, 0, 0],
            [-I, 0, 0, 0],
        ]),
        'I_C2y': Matrix([
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
        ]),
        'I_C2z': Matrix([
            [I, 0, 0, 0],
            [0, -I, 0, 0],
            [0, 0, I, 0],
            [0, 0, 0, -I],
        ]),
        'I_C3a': Matrix([
            [Rational(1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 + I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(1 - I)/4, Rational(1, 4) + I/4, Rational(1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 + I)/4, Rational(1, 4) + I/4],
        ]),
        'I_C3ai': Matrix([
            [Rational(1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 + I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(1, 4) + I/4, Rational(1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 + I)/4, Rational(1, 4) - I/4],
        ]),
        'I_C3b': Matrix([
            [Rational(1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 - I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(-1 - I)/4, Rational(1, 4) - I/4, Rational(-1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(1, 4) - I/4, Rational(1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 + I)/4, Rational(1, 4) - I/4],
        ]),
        'I_C3bi': Matrix([
            [Rational(1, 4) - I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(1 - I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(-1 - I)/4, Rational(1, 4) + I/4, Rational(1, 4) + I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) - I/4, sqrt(3)*(1 - I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(1 + I)/4, Rational(1, 4) + I/4],
        ]),
        'I_C3c': Matrix([
            [Rational(1, 4) + I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(-1 - I)/4, Rational(-1, 4) + I/4],
            [sqrt(3)*(1 + I)/4, Rational(1, 4) - I/4, Rational(1, 4) + I/4, sqrt(3)*(1 - I)/4],
            [sqrt(3)*(1 + I)/4, Rational(-1, 4) + I/4, Rational(1, 4) + I/4, sqrt(3)*(-1 + I)/4],
            [Rational(1, 4) + I/4, sqrt(3)*(-1 + I)/4, sqrt(3)*(-1 - I)/4, Rational(1, 4) - I/4],
        ]),
        'I_C3ci': Matrix([
            [Rational(1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 - I)/4, Rational(1, 4) - I/4],
            [sqrt(3)*(1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) - I/4, sqrt(3)*(-1 - I)/4],
            [sqrt(3)*(-1 + I)/4, Rational(1, 4) - I/4, Rational(1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [Rational(-1, 4) - I/4, sqrt(3)*(1 + I)/4, sqrt(3)*(-1 - I)/4, Rational(1, 4) + I/4],
        ]),
        'I_C3d': Matrix([
            [Rational(1, 4) - I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(-1 + I)/4, Rational(1, 4) + I/4],
            [sqrt(3)*(-1 + I)/4, Rational(1, 4) + I/4, Rational(-1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [sqrt(3)*(1 - I)/4, Rational(1, 4) + I/4, Rational(1, 4) - I/4, sqrt(3)*(1 + I)/4],
            [Rational(-1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 - I)/4, Rational(1, 4) + I/4],
        ]),
        'I_C3di': Matrix([
            [Rational(1, 4) + I/4, sqrt(3)*(-1 - I)/4, sqrt(3)*(1 + I)/4, Rational(-1, 4) - I/4],
            [sqrt(3)*(-1 + I)/4, Rational(1, 4) - I/4, Rational(1, 4) - I/4, sqrt(3)*(-1 + I)/4],
            [sqrt(3)*(-1 - I)/4, Rational(-1, 4) - I/4, Rational(1, 4) + I/4, sqrt(3)*(1 + I)/4],
            [Rational(1, 4) - I/4, sqrt(3)*(1 - I)/4, sqrt(3)*(1 - I)/4, Rational(1, 4) - I/4],
        ]),
        'I_C4x': Matrix([
            [-sqrt(2)/4, sqrt(6)*I/4, sqrt(6)/4, -sqrt(2)*I/4],
            [sqrt(6)*I/4, sqrt(2)/4, sqrt(2)*I/4, sqrt(6)/4],
            [sqrt(6)/4, sqrt(2)*I/4, sqrt(2)/4, sqrt(6)*I/4],
            [-sqrt(2)*I/4, sqrt(6)/4, sqrt(6)*I/4, -sqrt(2)/4],
        ]),
        'I_C4xi': Matrix([
            [-sqrt(2)/4, -sqrt(6)*I/4, sqrt(6)/4, sqrt(2)*I/4],
            [-sqrt(6)*I/4, sqrt(2)/4, -sqrt(2)*I/4, sqrt(6)/4],
            [sqrt(6)/4, -sqrt(2)*I/4, sqrt(2)/4, -sqrt(6)*I/4],
            [sqrt(2)*I/4, sqrt(6)/4, -sqrt(6)*I/4, -sqrt(2)/4],
        ]),
        'I_C4y': Matrix([
            [-sqrt(2)/4, -sqrt(6)/4, -sqrt(6)/4, -sqrt(2)/4],
            [sqrt(6)/4, sqrt(2)/4, -sqrt(2)/4, -sqrt(6)/4],
            [-sqrt(6)/4, sqrt(2)/4, sqrt(2)/4, -sqrt(6)/4],
            [sqrt(2)/4, -sqrt(6)/4, sqrt(6)/4, -sqrt(2)/4],
        ]),
        'I_C4yi': Matrix([
            [-sqrt(2)/4, sqrt(6)/4, -sqrt(6)/4, sqrt(2)/4],
            [-sqrt(6)/4, sqrt(2)/4, sqrt(2)/4, -sqrt(6)/4],
            [-sqrt(6)/4, -sqrt(2)/4, sqrt(2)/4, sqrt(6)/4],
            [-sqrt(2)/4, -sqrt(6)/4, -sqrt(6)/4, -sqrt(2)/4],
        ]),
        'I_C4z': Matrix([
            [sqrt(2)*(1 - I)/2, 0, 0, 0],
            [0, sqrt(2)*(-1 - I)/2, 0, 0],
            [0, 0, sqrt(2)*(-1 + I)/2, 0],
            [0, 0, 0, sqrt(2)*(1 + I)/2],
        ]),
        'I_C4zi': Matrix([
            [sqrt(2)*(1 + I)/2, 0, 0, 0],
            [0, sqrt(2)*(-1 + I)/2, 0, 0],
            [0, 0, sqrt(2)*(-1 - I)/2, 0],
            [0, 0, 0, sqrt(2)*(1 - I)/2],
        ]),
        'Is': Matrix([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ]),
    },
}
