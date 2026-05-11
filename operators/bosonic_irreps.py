from sympy import Matrix, I, sqrt, Rational

HARD_CODED_BOSONIC_IRREP_MATRICES = {
    ('C2v', 'A1'): {
        'C2e': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2f': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [1],
        ]),
    },
    ('C2v', 'A2'): {
        'C2e': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2f': Matrix([
            [-1],
        ]),
        'I_C2x': Matrix([
            [-1],
        ]),
    },
    ('C2v', 'B1'): {
        'C2e': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2f': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [-1],
        ]),
    },
    ('C2v', 'B2'): {
        'C2e': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2f': Matrix([
            [-1],
        ]),
        'I_C2x': Matrix([
            [1],
        ]),
    },
    ('C3v', 'A1'): {
        'C3d': Matrix([
            [1],
        ]),
        'C3di': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [1],
        ]),
        'I_C2d': Matrix([
            [1],
        ]),
        'I_C2f': Matrix([
            [1],
        ]),
    },
    ('C3v', 'A2'): {
        'C3d': Matrix([
            [1],
        ]),
        'C3di': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [-1],
        ]),
        'I_C2d': Matrix([
            [-1],
        ]),
        'I_C2f': Matrix([
            [-1],
        ]),
    },
    ('C3v', 'E'): {
        'C3d': Matrix([
            [I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3 - I*(-I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 - I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2)/3 + I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/3 + (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2 - sqrt(2)*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2), (-sqrt(2)/6 - sqrt(2)*I/6)*(I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 + I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2) + sqrt(2)*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/3 - sqrt(2)*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3 + (sqrt(2)/6 - sqrt(2)*I/6)*(-I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 - I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2)],
            [-sqrt(2)*(-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2 - sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/3 - I*(-sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 - I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2)/3 + I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3, -sqrt(2)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3 + (sqrt(2)/6 - sqrt(2)*I/6)*(-sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 - I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2) + (-sqrt(2)/6 - sqrt(2)*I/6)*(-sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 + I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2)],
        ]),
        'C3di': Matrix([
            [I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3 + (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)*I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 + sqrt(2)*I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2) - I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/3 - I*(I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 - I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2)/3, sqrt(2)*I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3 + (sqrt(2)/6 - sqrt(2)*I/6)*(I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 - I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2) + (-sqrt(2)/6 - sqrt(2)*I/6)*(I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 - I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2) + sqrt(2)*I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/3],
            [sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/3 + sqrt(2)*I*(-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2 + I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3 - I*(-sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 + I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2)/3, (-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 + I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2) + (sqrt(2)/6 - sqrt(2)*I/6)*(-sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 + I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2) + sqrt(2)*I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3],
        ]),
        'E': Matrix([
            [2*(-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/3 - I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3 + Rational(8, 9), (-sqrt(2)/6 - sqrt(2)*I/6)*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3)) + (sqrt(2)/6 - sqrt(2)*I/6)*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)],
            [-I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3 + (-sqrt(2)/6 + sqrt(2)*I/6)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3)), 2*(-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2)/3 + (sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3) + Rational(8, 9)],
        ]),
        'I_C2b': Matrix([
            [(-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/3 + Rational(4, 9) - 2*I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3, -I*(-sqrt(2)/6 - sqrt(2)*I/6)*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3) + I*(sqrt(2)/6 - sqrt(2)*I/6)*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))],
            [-2*I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3 + (Rational(-4, 3) - (-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2))*(-sqrt(2)/6 + sqrt(2)*I/6), Rational(-8, 9) - I*(-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3) - 2*(-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2)/3],
        ]),
        'I_C2d': Matrix([
            [Rational(-4, 9) - (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/3 + (-sqrt(2)/6 + sqrt(2)*I/6)*(-sqrt(2)*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2 + sqrt(2)*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2) - 2*I/9 - (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/3 - I*(Rational(-2, 3) - (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2 - I/3 - (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2)/3, -sqrt(2)*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/3 + (sqrt(2)/6 - sqrt(2)*I/6)*(Rational(-2, 3) - (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2 - I/3 - (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2) + (-sqrt(2)/6 - sqrt(2)*I/6)*(Rational(-2, 3) - (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2 - I/3 - (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2) + sqrt(2)*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3],
            [-sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/3 + sqrt(2)*(-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2 - sqrt(2)/9 - sqrt(2)*I/9 - I*(-sqrt(2)/6 + sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 - sqrt(2)*I/6 - I*(sqrt(2)/2 - sqrt(2)*I/2)/6)/3 - I*(sqrt(2)/2 - sqrt(2)*I/2)/9, (sqrt(2)/6 - sqrt(2)*I/6)*(-sqrt(2)/6 + sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 - sqrt(2)*I/6 - I*(sqrt(2)/2 - sqrt(2)*I/2)/6) + (-sqrt(2)/6 - sqrt(2)*I/6)*(-sqrt(2)*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2 - sqrt(2)/6 - sqrt(2)*I/6 - I*(sqrt(2)/2 - sqrt(2)*I/2)/6) + sqrt(2)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3],
        ]),
        'I_C2f': Matrix([
            [Rational(-4, 9) - (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/3 - I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2 + Rational(2, 3) - I/3 - (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2)/3 + (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/3 + 2*I/9 + (-sqrt(2)/6 + sqrt(2)*I/6)*(-sqrt(2)*I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/2 - sqrt(2)*I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/2), (-sqrt(2)/6 - sqrt(2)*I/6)*(Rational(-2, 3) - (-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2 + (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2 + I/3) - sqrt(2)*I*((sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + 2*I/3)/3 - sqrt(2)*I*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2) + Rational(4, 3))/3 + (sqrt(2)/6 - sqrt(2)*I/6)*((-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2 + Rational(2, 3) - I/3 - (sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/2 + sqrt(2)*I/2)/2)],
            [sqrt(2)/9 + I*(sqrt(2)/2 - sqrt(2)*I/2)/9 - I*(-sqrt(2)/6 - sqrt(2)*I/6 - I*(sqrt(2)/2 - sqrt(2)*I/2)/6 + sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2)/3 + sqrt(2)*I/9 - sqrt(2)*I*(-sqrt(2)/6 + sqrt(2)*I/6)*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/2 + sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/3, -sqrt(2)*I*(sqrt(2)/3 + I*(sqrt(2)/2 - sqrt(2)*I/2)/3 + sqrt(2)*I/3)/3 + (-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/6 + I*(sqrt(2)/2 - sqrt(2)*I/2)/6 + sqrt(2)*I/6 + sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2) + (sqrt(2)/6 - sqrt(2)*I/6)*(-sqrt(2)/6 - sqrt(2)*I/6 - I*(sqrt(2)/2 - sqrt(2)*I/2)/6 + sqrt(2)*I*((-sqrt(2)/6 - sqrt(2)*I/6)*(sqrt(2)/2 - sqrt(2)*I/2) + Rational(4, 3))/2)],
        ]),
    },
    ('C4v', 'A1'): {
        'C2z': Matrix([
            [1],
        ]),
        'C4z': Matrix([
            [1],
        ]),
        'C4zi': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [1],
        ]),
        'I_C2y': Matrix([
            [1],
        ]),
    },
    ('C4v', 'A2'): {
        'C2z': Matrix([
            [1],
        ]),
        'C4z': Matrix([
            [1],
        ]),
        'C4zi': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [-1],
        ]),
        'I_C2b': Matrix([
            [-1],
        ]),
        'I_C2x': Matrix([
            [-1],
        ]),
        'I_C2y': Matrix([
            [-1],
        ]),
    },
    ('C4v', 'B1'): {
        'C2z': Matrix([
            [1],
        ]),
        'C4z': Matrix([
            [-1],
        ]),
        'C4zi': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [-1],
        ]),
        'I_C2b': Matrix([
            [-1],
        ]),
        'I_C2x': Matrix([
            [1],
        ]),
        'I_C2y': Matrix([
            [1],
        ]),
    },
    ('C4v', 'B2'): {
        'C2z': Matrix([
            [1],
        ]),
        'C4z': Matrix([
            [-1],
        ]),
        'C4zi': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [-1],
        ]),
        'I_C2y': Matrix([
            [-1],
        ]),
    },
    ('C4v', 'E'): {
        'C2z': Matrix([
            [-1, 0],
            [0, -1],
        ]),
        'C4z': Matrix([
            [I, 0],
            [0, -I],
        ]),
        'C4zi': Matrix([
            [-I, 0],
            [0, I],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [0, -I],
            [I, 0],
        ]),
        'I_C2b': Matrix([
            [0, I],
            [-I, 0],
        ]),
        'I_C2x': Matrix([
            [0, -1],
            [-1, 0],
        ]),
        'I_C2y': Matrix([
            [0, 1],
            [1, 0],
        ]),
    },
    ('CS', 'A1'): {
        'E': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [1],
        ]),
    },
    ('CS', 'A2'): {
        'E': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [-1],
        ]),
        'I_C2x': Matrix([
            [-1],
        ]),
    },
    ('Oh', 'A1g'): {
        'C2a': Matrix([
            [1],
        ]),
        'C2b': Matrix([
            [1],
        ]),
        'C2c': Matrix([
            [1],
        ]),
        'C2d': Matrix([
            [1],
        ]),
        'C2e': Matrix([
            [1],
        ]),
        'C2f': Matrix([
            [1],
        ]),
        'C2x': Matrix([
            [1],
        ]),
        'C2y': Matrix([
            [1],
        ]),
        'C2z': Matrix([
            [1],
        ]),
        'C3a': Matrix([
            [1],
        ]),
        'C3ai': Matrix([
            [1],
        ]),
        'C3b': Matrix([
            [1],
        ]),
        'C3bi': Matrix([
            [1],
        ]),
        'C3c': Matrix([
            [1],
        ]),
        'C3ci': Matrix([
            [1],
        ]),
        'C3d': Matrix([
            [1],
        ]),
        'C3di': Matrix([
            [1],
        ]),
        'C4x': Matrix([
            [1],
        ]),
        'C4xi': Matrix([
            [1],
        ]),
        'C4y': Matrix([
            [1],
        ]),
        'C4yi': Matrix([
            [1],
        ]),
        'C4z': Matrix([
            [1],
        ]),
        'C4zi': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [1],
        ]),
        'I_C2c': Matrix([
            [1],
        ]),
        'I_C2d': Matrix([
            [1],
        ]),
        'I_C2e': Matrix([
            [1],
        ]),
        'I_C2f': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [1],
        ]),
        'I_C2y': Matrix([
            [1],
        ]),
        'I_C2z': Matrix([
            [1],
        ]),
        'I_C3a': Matrix([
            [1],
        ]),
        'I_C3ai': Matrix([
            [1],
        ]),
        'I_C3b': Matrix([
            [1],
        ]),
        'I_C3bi': Matrix([
            [1],
        ]),
        'I_C3c': Matrix([
            [1],
        ]),
        'I_C3ci': Matrix([
            [1],
        ]),
        'I_C3d': Matrix([
            [1],
        ]),
        'I_C3di': Matrix([
            [1],
        ]),
        'I_C4x': Matrix([
            [1],
        ]),
        'I_C4xi': Matrix([
            [1],
        ]),
        'I_C4y': Matrix([
            [1],
        ]),
        'I_C4yi': Matrix([
            [1],
        ]),
        'I_C4z': Matrix([
            [1],
        ]),
        'I_C4zi': Matrix([
            [1],
        ]),
        'Is': Matrix([
            [1],
        ]),
    },
    ('Oh', 'A1u'): {
        'C2a': Matrix([
            [1],
        ]),
        'C2b': Matrix([
            [1],
        ]),
        'C2c': Matrix([
            [1],
        ]),
        'C2d': Matrix([
            [1],
        ]),
        'C2e': Matrix([
            [1],
        ]),
        'C2f': Matrix([
            [1],
        ]),
        'C2x': Matrix([
            [1],
        ]),
        'C2y': Matrix([
            [1],
        ]),
        'C2z': Matrix([
            [1],
        ]),
        'C3a': Matrix([
            [1],
        ]),
        'C3ai': Matrix([
            [1],
        ]),
        'C3b': Matrix([
            [1],
        ]),
        'C3bi': Matrix([
            [1],
        ]),
        'C3c': Matrix([
            [1],
        ]),
        'C3ci': Matrix([
            [1],
        ]),
        'C3d': Matrix([
            [1],
        ]),
        'C3di': Matrix([
            [1],
        ]),
        'C4x': Matrix([
            [1],
        ]),
        'C4xi': Matrix([
            [1],
        ]),
        'C4y': Matrix([
            [1],
        ]),
        'C4yi': Matrix([
            [1],
        ]),
        'C4z': Matrix([
            [1],
        ]),
        'C4zi': Matrix([
            [1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [-1],
        ]),
        'I_C2b': Matrix([
            [-1],
        ]),
        'I_C2c': Matrix([
            [-1],
        ]),
        'I_C2d': Matrix([
            [-1],
        ]),
        'I_C2e': Matrix([
            [-1],
        ]),
        'I_C2f': Matrix([
            [-1],
        ]),
        'I_C2x': Matrix([
            [-1],
        ]),
        'I_C2y': Matrix([
            [-1],
        ]),
        'I_C2z': Matrix([
            [-1],
        ]),
        'I_C3a': Matrix([
            [-1],
        ]),
        'I_C3ai': Matrix([
            [-1],
        ]),
        'I_C3b': Matrix([
            [-1],
        ]),
        'I_C3bi': Matrix([
            [-1],
        ]),
        'I_C3c': Matrix([
            [-1],
        ]),
        'I_C3ci': Matrix([
            [-1],
        ]),
        'I_C3d': Matrix([
            [-1],
        ]),
        'I_C3di': Matrix([
            [-1],
        ]),
        'I_C4x': Matrix([
            [-1],
        ]),
        'I_C4xi': Matrix([
            [-1],
        ]),
        'I_C4y': Matrix([
            [-1],
        ]),
        'I_C4yi': Matrix([
            [-1],
        ]),
        'I_C4z': Matrix([
            [-1],
        ]),
        'I_C4zi': Matrix([
            [-1],
        ]),
        'Is': Matrix([
            [-1],
        ]),
    },
    ('Oh', 'A2g'): {
        'C2a': Matrix([
            [-1],
        ]),
        'C2b': Matrix([
            [-1],
        ]),
        'C2c': Matrix([
            [-1],
        ]),
        'C2d': Matrix([
            [-1],
        ]),
        'C2e': Matrix([
            [-1],
        ]),
        'C2f': Matrix([
            [-1],
        ]),
        'C2x': Matrix([
            [1],
        ]),
        'C2y': Matrix([
            [1],
        ]),
        'C2z': Matrix([
            [1],
        ]),
        'C3a': Matrix([
            [1],
        ]),
        'C3ai': Matrix([
            [1],
        ]),
        'C3b': Matrix([
            [1],
        ]),
        'C3bi': Matrix([
            [1],
        ]),
        'C3c': Matrix([
            [1],
        ]),
        'C3ci': Matrix([
            [1],
        ]),
        'C3d': Matrix([
            [1],
        ]),
        'C3di': Matrix([
            [1],
        ]),
        'C4x': Matrix([
            [-1],
        ]),
        'C4xi': Matrix([
            [-1],
        ]),
        'C4y': Matrix([
            [-1],
        ]),
        'C4yi': Matrix([
            [-1],
        ]),
        'C4z': Matrix([
            [-1],
        ]),
        'C4zi': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [-1],
        ]),
        'I_C2b': Matrix([
            [-1],
        ]),
        'I_C2c': Matrix([
            [-1],
        ]),
        'I_C2d': Matrix([
            [-1],
        ]),
        'I_C2e': Matrix([
            [-1],
        ]),
        'I_C2f': Matrix([
            [-1],
        ]),
        'I_C2x': Matrix([
            [1],
        ]),
        'I_C2y': Matrix([
            [1],
        ]),
        'I_C2z': Matrix([
            [1],
        ]),
        'I_C3a': Matrix([
            [1],
        ]),
        'I_C3ai': Matrix([
            [1],
        ]),
        'I_C3b': Matrix([
            [1],
        ]),
        'I_C3bi': Matrix([
            [1],
        ]),
        'I_C3c': Matrix([
            [1],
        ]),
        'I_C3ci': Matrix([
            [1],
        ]),
        'I_C3d': Matrix([
            [1],
        ]),
        'I_C3di': Matrix([
            [1],
        ]),
        'I_C4x': Matrix([
            [-1],
        ]),
        'I_C4xi': Matrix([
            [-1],
        ]),
        'I_C4y': Matrix([
            [-1],
        ]),
        'I_C4yi': Matrix([
            [-1],
        ]),
        'I_C4z': Matrix([
            [-1],
        ]),
        'I_C4zi': Matrix([
            [-1],
        ]),
        'Is': Matrix([
            [1],
        ]),
    },
    ('Oh', 'A2u'): {
        'C2a': Matrix([
            [-1],
        ]),
        'C2b': Matrix([
            [-1],
        ]),
        'C2c': Matrix([
            [-1],
        ]),
        'C2d': Matrix([
            [-1],
        ]),
        'C2e': Matrix([
            [-1],
        ]),
        'C2f': Matrix([
            [-1],
        ]),
        'C2x': Matrix([
            [1],
        ]),
        'C2y': Matrix([
            [1],
        ]),
        'C2z': Matrix([
            [1],
        ]),
        'C3a': Matrix([
            [1],
        ]),
        'C3ai': Matrix([
            [1],
        ]),
        'C3b': Matrix([
            [1],
        ]),
        'C3bi': Matrix([
            [1],
        ]),
        'C3c': Matrix([
            [1],
        ]),
        'C3ci': Matrix([
            [1],
        ]),
        'C3d': Matrix([
            [1],
        ]),
        'C3di': Matrix([
            [1],
        ]),
        'C4x': Matrix([
            [-1],
        ]),
        'C4xi': Matrix([
            [-1],
        ]),
        'C4y': Matrix([
            [-1],
        ]),
        'C4yi': Matrix([
            [-1],
        ]),
        'C4z': Matrix([
            [-1],
        ]),
        'C4zi': Matrix([
            [-1],
        ]),
        'E': Matrix([
            [1],
        ]),
        'I_C2a': Matrix([
            [1],
        ]),
        'I_C2b': Matrix([
            [1],
        ]),
        'I_C2c': Matrix([
            [1],
        ]),
        'I_C2d': Matrix([
            [1],
        ]),
        'I_C2e': Matrix([
            [1],
        ]),
        'I_C2f': Matrix([
            [1],
        ]),
        'I_C2x': Matrix([
            [-1],
        ]),
        'I_C2y': Matrix([
            [-1],
        ]),
        'I_C2z': Matrix([
            [-1],
        ]),
        'I_C3a': Matrix([
            [-1],
        ]),
        'I_C3ai': Matrix([
            [-1],
        ]),
        'I_C3b': Matrix([
            [-1],
        ]),
        'I_C3bi': Matrix([
            [-1],
        ]),
        'I_C3c': Matrix([
            [-1],
        ]),
        'I_C3ci': Matrix([
            [-1],
        ]),
        'I_C3d': Matrix([
            [-1],
        ]),
        'I_C3di': Matrix([
            [-1],
        ]),
        'I_C4x': Matrix([
            [1],
        ]),
        'I_C4xi': Matrix([
            [1],
        ]),
        'I_C4y': Matrix([
            [1],
        ]),
        'I_C4yi': Matrix([
            [1],
        ]),
        'I_C4z': Matrix([
            [1],
        ]),
        'I_C4zi': Matrix([
            [1],
        ]),
        'Is': Matrix([
            [-1],
        ]),
    },
    ('Oh', 'Eg'): {
        'C2a': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'C2b': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'C2c': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2d': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2e': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2f': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2x': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'C2y': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'C2z': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'C3a': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3ai': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3b': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3bi': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3c': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3ci': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3d': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3di': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4x': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4xi': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4y': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4yi': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4z': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'C4zi': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'I_C2b': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'I_C2c': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C2d': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C2e': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C2f': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C2x': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2y': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2z': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C3a': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C3ai': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C3b': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C3bi': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C3c': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C3ci': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C3d': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C3di': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C4x': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C4xi': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C4y': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C4yi': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'I_C4z': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'I_C4zi': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'Is': Matrix([
            [1, 0],
            [0, 1],
        ]),
    },
    ('Oh', 'Eu'): {
        'C2a': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'C2b': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'C2c': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2d': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2e': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2f': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C2x': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'C2y': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'C2z': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'C3a': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3ai': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3b': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3bi': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3c': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3ci': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3d': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C3di': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4x': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4xi': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4y': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4yi': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(-1, 2)],
        ]),
        'C4z': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'C4zi': Matrix([
            [-1, 0],
            [0, 1],
        ]),
        'E': Matrix([
            [1, 0],
            [0, 1],
        ]),
        'I_C2a': Matrix([
            [1, 0],
            [0, -1],
        ]),
        'I_C2b': Matrix([
            [1, 0],
            [0, -1],
        ]),
        'I_C2c': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C2d': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C2e': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C2f': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C2x': Matrix([
            [-1, 0],
            [0, -1],
        ]),
        'I_C2y': Matrix([
            [-1, 0],
            [0, -1],
        ]),
        'I_C2z': Matrix([
            [-1, 0],
            [0, -1],
        ]),
        'I_C3a': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C3ai': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C3b': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C3bi': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C3c': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C3ci': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C3d': Matrix([
            [Rational(1, 2), -sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C3di': Matrix([
            [Rational(1, 2), sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C4x': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C4xi': Matrix([
            [Rational(-1, 2), sqrt(6)/2],
            [sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C4y': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C4yi': Matrix([
            [Rational(-1, 2), -sqrt(6)/2],
            [-sqrt(6)/4, Rational(1, 2)],
        ]),
        'I_C4z': Matrix([
            [1, 0],
            [0, -1],
        ]),
        'I_C4zi': Matrix([
            [1, 0],
            [0, -1],
        ]),
        'Is': Matrix([
            [-1, 0],
            [0, -1],
        ]),
    },
    ('Oh', 'T1g'): {
        'C2a': Matrix([
            [0, 0, -I],
            [0, -1, 0],
            [I, 0, 0],
        ]),
        'C2b': Matrix([
            [0, 0, I],
            [0, -1, 0],
            [-I, 0, 0],
        ]),
        'C2c': Matrix([
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
        ]),
        'C2d': Matrix([
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
        ]),
        'C2e': Matrix([
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'C2f': Matrix([
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'C2x': Matrix([
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
        ]),
        'C2y': Matrix([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ]),
        'C2z': Matrix([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ]),
        'C3a': Matrix([
            [I/2, -sqrt(2)/2, -I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [I/2, sqrt(2)/2, -I/2],
        ]),
        'C3ai': Matrix([
            [-I/2, -sqrt(2)*I/2, -I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [I/2, -sqrt(2)*I/2, I/2],
        ]),
        'C3b': Matrix([
            [-I/2, sqrt(2)/2, I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [-I/2, -sqrt(2)/2, I/2],
        ]),
        'C3bi': Matrix([
            [I/2, -sqrt(2)*I/2, I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [-I/2, -sqrt(2)*I/2, -I/2],
        ]),
        'C3c': Matrix([
            [-I/2, -sqrt(2)/2, I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [-I/2, sqrt(2)/2, I/2],
        ]),
        'C3ci': Matrix([
            [I/2, sqrt(2)*I/2, I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [-I/2, sqrt(2)*I/2, -I/2],
        ]),
        'C3d': Matrix([
            [I/2, sqrt(2)/2, -I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [I/2, -sqrt(2)/2, -I/2],
        ]),
        'C3di': Matrix([
            [-I/2, sqrt(2)*I/2, -I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [I/2, sqrt(2)*I/2, I/2],
        ]),
        'C4x': Matrix([
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'C4xi': Matrix([
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'C4y': Matrix([
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
        ]),
        'C4yi': Matrix([
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
        ]),
        'C4z': Matrix([
            [I, 0, 0],
            [0, 1, 0],
            [0, 0, -I],
        ]),
        'C4zi': Matrix([
            [-I, 0, 0],
            [0, 1, 0],
            [0, 0, I],
        ]),
        'E': Matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        'I_C2a': Matrix([
            [0, 0, -I],
            [0, -1, 0],
            [I, 0, 0],
        ]),
        'I_C2b': Matrix([
            [0, 0, I],
            [0, -1, 0],
            [-I, 0, 0],
        ]),
        'I_C2c': Matrix([
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
        ]),
        'I_C2d': Matrix([
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
        ]),
        'I_C2e': Matrix([
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'I_C2f': Matrix([
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'I_C2x': Matrix([
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
        ]),
        'I_C2y': Matrix([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ]),
        'I_C2z': Matrix([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ]),
        'I_C3a': Matrix([
            [I/2, -sqrt(2)/2, -I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [I/2, sqrt(2)/2, -I/2],
        ]),
        'I_C3ai': Matrix([
            [-I/2, -sqrt(2)*I/2, -I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [I/2, -sqrt(2)*I/2, I/2],
        ]),
        'I_C3b': Matrix([
            [-I/2, sqrt(2)/2, I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [-I/2, -sqrt(2)/2, I/2],
        ]),
        'I_C3bi': Matrix([
            [I/2, -sqrt(2)*I/2, I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [-I/2, -sqrt(2)*I/2, -I/2],
        ]),
        'I_C3c': Matrix([
            [-I/2, -sqrt(2)/2, I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [-I/2, sqrt(2)/2, I/2],
        ]),
        'I_C3ci': Matrix([
            [I/2, sqrt(2)*I/2, I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [-I/2, sqrt(2)*I/2, -I/2],
        ]),
        'I_C3d': Matrix([
            [I/2, sqrt(2)/2, -I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [I/2, -sqrt(2)/2, -I/2],
        ]),
        'I_C3di': Matrix([
            [-I/2, sqrt(2)*I/2, -I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [I/2, sqrt(2)*I/2, I/2],
        ]),
        'I_C4x': Matrix([
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'I_C4xi': Matrix([
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'I_C4y': Matrix([
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
        ]),
        'I_C4yi': Matrix([
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
        ]),
        'I_C4z': Matrix([
            [I, 0, 0],
            [0, 1, 0],
            [0, 0, -I],
        ]),
        'I_C4zi': Matrix([
            [-I, 0, 0],
            [0, 1, 0],
            [0, 0, I],
        ]),
        'Is': Matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
    },
    ('Oh', 'T1u'): {
        'C2a': Matrix([
            [0, 0, -I],
            [0, -1, 0],
            [I, 0, 0],
        ]),
        'C2b': Matrix([
            [0, 0, I],
            [0, -1, 0],
            [-I, 0, 0],
        ]),
        'C2c': Matrix([
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
        ]),
        'C2d': Matrix([
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
        ]),
        'C2e': Matrix([
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'C2f': Matrix([
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'C2x': Matrix([
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
        ]),
        'C2y': Matrix([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ]),
        'C2z': Matrix([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ]),
        'C3a': Matrix([
            [I/2, -sqrt(2)/2, -I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [I/2, sqrt(2)/2, -I/2],
        ]),
        'C3ai': Matrix([
            [-I/2, -sqrt(2)*I/2, -I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [I/2, -sqrt(2)*I/2, I/2],
        ]),
        'C3b': Matrix([
            [-I/2, sqrt(2)/2, I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [-I/2, -sqrt(2)/2, I/2],
        ]),
        'C3bi': Matrix([
            [I/2, -sqrt(2)*I/2, I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [-I/2, -sqrt(2)*I/2, -I/2],
        ]),
        'C3c': Matrix([
            [-I/2, -sqrt(2)/2, I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [-I/2, sqrt(2)/2, I/2],
        ]),
        'C3ci': Matrix([
            [I/2, sqrt(2)*I/2, I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [-I/2, sqrt(2)*I/2, -I/2],
        ]),
        'C3d': Matrix([
            [I/2, sqrt(2)/2, -I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [I/2, -sqrt(2)/2, -I/2],
        ]),
        'C3di': Matrix([
            [-I/2, sqrt(2)*I/2, -I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [I/2, sqrt(2)*I/2, I/2],
        ]),
        'C4x': Matrix([
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'C4xi': Matrix([
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'C4y': Matrix([
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
        ]),
        'C4yi': Matrix([
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
        ]),
        'C4z': Matrix([
            [I, 0, 0],
            [0, 1, 0],
            [0, 0, -I],
        ]),
        'C4zi': Matrix([
            [-I, 0, 0],
            [0, 1, 0],
            [0, 0, I],
        ]),
        'E': Matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        'I_C2a': Matrix([
            [0, 0, I],
            [0, 1, 0],
            [-I, 0, 0],
        ]),
        'I_C2b': Matrix([
            [0, 0, -I],
            [0, 1, 0],
            [I, 0, 0],
        ]),
        'I_C2c': Matrix([
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
        ]),
        'I_C2d': Matrix([
            [Rational(1, 2), sqrt(2)/2, Rational(1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(1, 2), -sqrt(2)/2, Rational(1, 2)],
        ]),
        'I_C2e': Matrix([
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'I_C2f': Matrix([
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
        ]),
        'I_C2x': Matrix([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]),
        'I_C2y': Matrix([
            [0, 0, -1],
            [0, 1, 0],
            [-1, 0, 0],
        ]),
        'I_C2z': Matrix([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]),
        'I_C3a': Matrix([
            [-I/2, sqrt(2)/2, I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [-I/2, -sqrt(2)/2, I/2],
        ]),
        'I_C3ai': Matrix([
            [I/2, sqrt(2)*I/2, I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [-I/2, sqrt(2)*I/2, -I/2],
        ]),
        'I_C3b': Matrix([
            [I/2, -sqrt(2)/2, -I/2],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [I/2, sqrt(2)/2, -I/2],
        ]),
        'I_C3bi': Matrix([
            [-I/2, sqrt(2)*I/2, -I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [I/2, sqrt(2)*I/2, I/2],
        ]),
        'I_C3c': Matrix([
            [I/2, sqrt(2)/2, -I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [I/2, -sqrt(2)/2, -I/2],
        ]),
        'I_C3ci': Matrix([
            [-I/2, -sqrt(2)*I/2, -I/2],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [I/2, -sqrt(2)*I/2, I/2],
        ]),
        'I_C3d': Matrix([
            [-I/2, -sqrt(2)/2, I/2],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [-I/2, sqrt(2)/2, I/2],
        ]),
        'I_C3di': Matrix([
            [I/2, -sqrt(2)*I/2, I/2],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [-I/2, -sqrt(2)*I/2, -I/2],
        ]),
        'I_C4x': Matrix([
            [Rational(-1, 2), sqrt(2)*I/2, Rational(1, 2)],
            [sqrt(2)*I/2, 0, sqrt(2)*I/2],
            [Rational(1, 2), sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'I_C4xi': Matrix([
            [Rational(-1, 2), -sqrt(2)*I/2, Rational(1, 2)],
            [-sqrt(2)*I/2, 0, -sqrt(2)*I/2],
            [Rational(1, 2), -sqrt(2)*I/2, Rational(-1, 2)],
        ]),
        'I_C4y': Matrix([
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
            [sqrt(2)/2, 0, -sqrt(2)/2],
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
        ]),
        'I_C4yi': Matrix([
            [Rational(-1, 2), sqrt(2)/2, Rational(-1, 2)],
            [-sqrt(2)/2, 0, sqrt(2)/2],
            [Rational(-1, 2), -sqrt(2)/2, Rational(-1, 2)],
        ]),
        'I_C4z': Matrix([
            [-I, 0, 0],
            [0, -1, 0],
            [0, 0, I],
        ]),
        'I_C4zi': Matrix([
            [I, 0, 0],
            [0, -1, 0],
            [0, 0, -I],
        ]),
        'Is': Matrix([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]),
    },
    ('Oh', 'T2g'): {
        'C2a': Matrix([
            [1, 0, 0],
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I), -(-1)*I],
            [0, -I, -I*(-I/8 + Rational(-1, 8)*(-1)*I)],
        ]),
        'C2b': Matrix([
            [1, 0, 0],
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I), -I],
            [0, -(-1)*I, I*(-I/8 + Rational(-1, 8)*(-1)*I)],
        ]),
        'C2c': Matrix([
            [0, -1, -1],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(-1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'C2d': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'C2e': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [-I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'C2f': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [-I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'C2x': Matrix([
            [-1, 0, 0],
            [0, -I/8 + Rational(-1, 8)*(-1)*I, 1],
            [0, 1, -I/8 + Rational(-1, 8)*(-1)*I],
        ]),
        'C2y': Matrix([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
        ]),
        'C2z': Matrix([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]),
        'C3a': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [Rational(-1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [Rational(-1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'C3ai': Matrix([
            [0, -1, -1],
            [I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [-I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'C3b': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [Rational(1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [Rational(1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'C3bi': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [-I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'C3c': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [Rational(-1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [Rational(-1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'C3ci': Matrix([
            [0, -1, -1],
            [-I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'C3d': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [Rational(1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [Rational(1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'C3di': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [-I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'C4x': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [-I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'C4xi': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [-I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'C4y': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [Rational(-1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'C4yi': Matrix([
            [0, -1, -1],
            [Rational(1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'C4z': Matrix([
            [-1, 0, 0],
            [0, I, I*(-I/8 + Rational(-1, 8)*(-1)*I)],
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I), -I],
        ]),
        'C4zi': Matrix([
            [-1, 0, 0],
            [0, -I, -I*(-I/8 + Rational(-1, 8)*(-1)*I)],
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I), I],
        ]),
        'E': Matrix([
            [1, 0, 0],
            [0, 1, -I/8 + Rational(-1, 8)*(-1)*I],
            [0, -I/8 + Rational(-1, 8)*(-1)*I, 1],
        ]),
        'I_C2a': Matrix([
            [1, 0, 0],
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I), -(-1)*I],
            [0, -I, -I*(-I/8 + Rational(-1, 8)*(-1)*I)],
        ]),
        'I_C2b': Matrix([
            [1, 0, 0],
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I), -I],
            [0, -(-1)*I, I*(-I/8 + Rational(-1, 8)*(-1)*I)],
        ]),
        'I_C2c': Matrix([
            [0, -1, -1],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(-1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'I_C2d': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'I_C2e': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [-I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'I_C2f': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [-I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
            [I/2, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'I_C2x': Matrix([
            [-1, 0, 0],
            [0, -I/8 + Rational(-1, 8)*(-1)*I, 1],
            [0, 1, -I/8 + Rational(-1, 8)*(-1)*I],
        ]),
        'I_C2y': Matrix([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
        ]),
        'I_C2z': Matrix([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]),
        'I_C3a': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [Rational(-1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [Rational(-1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'I_C3ai': Matrix([
            [0, -1, -1],
            [I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [-I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'I_C3b': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [Rational(1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [Rational(1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'I_C3bi': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [-I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'I_C3c': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [Rational(-1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [Rational(-1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'I_C3ci': Matrix([
            [0, -1, -1],
            [-I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
            [I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'I_C3d': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [Rational(1, 2), -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [Rational(1, 2), I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2],
        ]),
        'I_C3di': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [-I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
            [I/2, -I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 + I/2, I*(-I/8 + Rational(-1, 8)*(-1)*I)/2 - I/2],
        ]),
        'I_C4x': Matrix([
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I],
            [-I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'I_C4xi': Matrix([
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I) + I, I*(-I/8 + Rational(-1, 8)*(-1)*I) - I],
            [I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [-I/2, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) + (Rational(1, 16))*(-1)*I + I/16],
        ]),
        'I_C4y': Matrix([
            [0, 1 - I/8 + Rational(-1, 8)*(-1)*I, 1 - I/8 + Rational(-1, 8)*(-1)*I],
            [Rational(-1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'I_C4yi': Matrix([
            [0, -1, -1],
            [Rational(1, 2), Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I, Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 16))*(-1)*I + I/16, Rational(-1, 2) - I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'I_C4z': Matrix([
            [-1, 0, 0],
            [0, I, I*(-I/8 + Rational(-1, 8)*(-1)*I)],
            [0, -I*(-I/8 + Rational(-1, 8)*(-1)*I), -I],
        ]),
        'I_C4zi': Matrix([
            [-1, 0, 0],
            [0, -I, -I*(-I/8 + Rational(-1, 8)*(-1)*I)],
            [0, I*(-I/8 + Rational(-1, 8)*(-1)*I), I],
        ]),
        'Is': Matrix([
            [1, 0, 0],
            [0, 1, -I/8 + Rational(-1, 8)*(-1)*I],
            [0, -I/8 + Rational(-1, 8)*(-1)*I, 1],
        ]),
    },
    ('Oh', 'T2u'): {
        'C2a': Matrix([
            [1, 0, 0],
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I), -(-1)*I],
            [0, -I, -I*(-I/16 + Rational(-1, 16)*(-1)*I)],
        ]),
        'C2b': Matrix([
            [1, 0, 0],
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I), -I],
            [0, -(-1)*I, I*(-I/16 + Rational(-1, 16)*(-1)*I)],
        ]),
        'C2c': Matrix([
            [0, -1, -1],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [Rational(-1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'C2d': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [Rational(1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'C2e': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [-I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'C2f': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [-I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'C2x': Matrix([
            [-1, 0, 0],
            [0, -I/16 + Rational(-1, 16)*(-1)*I, 1],
            [0, 1, -I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'C2y': Matrix([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
        ]),
        'C2z': Matrix([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]),
        'C3a': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [Rational(-1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [Rational(-1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'C3ai': Matrix([
            [0, -1, -1],
            [I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [-I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'C3b': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [Rational(1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [Rational(1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'C3bi': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [-I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'C3c': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [Rational(-1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [Rational(-1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'C3ci': Matrix([
            [0, -1, -1],
            [-I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'C3d': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [Rational(1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [Rational(1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'C3di': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [-I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'C4x': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [-I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'C4xi': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [-I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'C4y': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(-1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'C4yi': Matrix([
            [0, -1, -1],
            [Rational(1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'C4z': Matrix([
            [-1, 0, 0],
            [0, I, I*(-I/16 + Rational(-1, 16)*(-1)*I)],
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I), -I],
        ]),
        'C4zi': Matrix([
            [-1, 0, 0],
            [0, -I, -I*(-I/16 + Rational(-1, 16)*(-1)*I)],
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I), I],
        ]),
        'E': Matrix([
            [1, 0, 0],
            [0, 1, -I/16 + Rational(-1, 16)*(-1)*I],
            [0, -I/16 + Rational(-1, 16)*(-1)*I, 1],
        ]),
        'I_C2a': Matrix([
            [-1, 0, 0],
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I), -I],
            [0, I, I*(-I/16 + Rational(-1, 16)*(-1)*I)],
        ]),
        'I_C2b': Matrix([
            [-1, 0, 0],
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I), I],
            [0, -I, -I*(-I/16 + Rational(-1, 16)*(-1)*I)],
        ]),
        'I_C2c': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'I_C2d': Matrix([
            [0, -1, -1],
            [Rational(-1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'I_C2e': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [-I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'I_C2f': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
            [-I/2, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'I_C2x': Matrix([
            [1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
        ]),
        'I_C2y': Matrix([
            [1, 0, 0],
            [0, -I/16 + Rational(-1, 16)*(-1)*I, 1],
            [0, 1, -I/16 + Rational(-1, 16)*(-1)*I],
        ]),
        'I_C2z': Matrix([
            [-1, 0, 0],
            [0, 1, -I/16 + Rational(-1, 16)*(-1)*I],
            [0, -I/16 + Rational(-1, 16)*(-1)*I, 1],
        ]),
        'I_C3a': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [Rational(1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [Rational(1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'I_C3ai': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [-I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'I_C3b': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [Rational(-1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [Rational(-1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'I_C3bi': Matrix([
            [0, -1, -1],
            [-I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'I_C3c': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [Rational(1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [Rational(1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'I_C3ci': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
            [-I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'I_C3d': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [Rational(-1, 2), I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [Rational(-1, 2), -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2],
        ]),
        'I_C3di': Matrix([
            [0, -1, -1],
            [I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
            [-I/2, I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 - I/2, -I*(-I/16 + Rational(-1, 16)*(-1)*I)/2 + I/2],
        ]),
        'I_C4x': Matrix([
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I],
            [I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [-I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'I_C4xi': Matrix([
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I) - I, -I*(-I/16 + Rational(-1, 16)*(-1)*I) + I],
            [-I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [I/2, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
        ]),
        'I_C4y': Matrix([
            [0, -1, -1],
            [Rational(1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [Rational(1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'I_C4yi': Matrix([
            [0, 1 - I/16 + Rational(-1, 16)*(-1)*I, 1 - I/16 + Rational(-1, 16)*(-1)*I],
            [Rational(-1, 2), Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32, Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I],
            [Rational(-1, 2), Rational(-1, 2) - I/32 + Rational(-1, 32)*(-1)*I, Rational(1, 2) + (Rational(1, 32))*(-1)*I + I/32],
        ]),
        'I_C4z': Matrix([
            [1, 0, 0],
            [0, -I, -I*(-I/16 + Rational(-1, 16)*(-1)*I)],
            [0, I*(-I/16 + Rational(-1, 16)*(-1)*I), I],
        ]),
        'I_C4zi': Matrix([
            [1, 0, 0],
            [0, I, I*(-I/16 + Rational(-1, 16)*(-1)*I)],
            [0, -I*(-I/16 + Rational(-1, 16)*(-1)*I), -I],
        ]),
        'Is': Matrix([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]),
    },
}
