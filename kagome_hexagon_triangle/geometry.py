"""27-spin blocking, in the old Julia library's physical-site coordinates.

A(n,m) is a six-spin hexagon. B(n,m) groups three spins which are NOT
nearest neighbours. Virtual leg k of A connects B at shifts (0,0),(-1,0),
(0,-1). Leg k of B connects A at the opposite shift. The cell periods
3*a1 and 3*a2 become (2,-1) and (1,1) in the coarse u,v coordinates.
Consequently each kind has three colours c=(n-m)%3.
"""
import numpy as np

A1 = np.array([1., 0.])
A2 = np.array([.5, np.sqrt(3)/2])
U, V = A1+A2, -A1+2*A2
C0, DELTA = (A1-A2)/2, (U+V)/3
SHIFTS = ((0, 0), (-1, 0), (0, -1))
OFFSETS = {
    'A': .5*np.column_stack((np.cos(np.arange(6)*np.pi/3), np.sin(np.arange(6)*np.pi/3))),
    'B': .5*np.column_stack((np.cos([-np.pi/3, np.pi/3, np.pi]),
                            np.sin([-np.pi/3, np.pi/3, np.pi]))),
}
REPS = {'A': ((0, 0), (1, 0), (1, -1)), 'B': ((0, 0), (0, -1), (1, -1))}


def points(kind, n=0, m=0):
    return C0+n*U+m*V+(DELTA if kind == 'B' else 0)+OFFSETS[kind]


def edge_bonds(k):
    n, m = SHIFTS[k]
    a, b = points('A'), points('B', n, m)
    return [(i, j) for i in range(6) for j in range(3)
            if abs(np.linalg.norm(a[i]-b[j])-.5) < 1e-9]


def edge_support(k):
    bonds = edge_bonds(k)
    aa = sorted({i for i, _ in bonds})
    bb = sorted({j for _, j in bonds})
    internal = [(i, j) for i in aa for j in aa if i < j
                and abs(np.linalg.norm(OFFSETS['A'][i]-OFFSETS['A'][j])-.5) < 1e-9]
    return aa, bb, internal, bonds


def neighbour(c, k):
    n, m = SHIFTS[k]
    return (c+n-m) % 3


def face_nodes(c):
    # Clockwise: a_top,b_upper_right,a_lower_right,b_bottom,a_lower_left,b_upper_left.
    return (('A', (c+1)%3), ('B', (c+2)%3), ('A', (c+2)%3),
            ('B', (c+1)%3), ('A', c), ('B', c))


def audit():
    assert all(len(edge_bonds(k)) == 4 for k in range(3))
    internal = [e for k in range(3) for e in edge_support(k)[2]]
    assert len(internal) == len(set(internal)) == 6
    assert all(len(edge_support(k)[0]) == 3 and len(edge_support(k)[1]) == 2 for k in range(3))
    assert all(abs(np.linalg.norm(x-y)-.5) > 1e-9
               for i, x in enumerate(OFFSETS['B']) for y in OFFSETS['B'][i+1:])
    ps = np.concatenate([points(kind, *q) for kind in ('A', 'B') for q in REPS[kind]])
    period = np.column_stack((3*A1, 3*A2))
    keys = [tuple(np.round(np.linalg.solve(period, x) % 1, 8) % 1) for x in ps]
    assert len(set(keys)) == 27
    # Independent periodic nearest-neighbour enumeration, not the block list.
    bonds = []
    for i, x in enumerate(ps):
        for j, y in enumerate(ps):
            for s in range(-1, 2):
                for t in range(-1, 2):
                    if abs(np.linalg.norm(x-y-s*3*A1-t*3*A2)-.5) < 1e-9:
                        bonds.append((i, j, s, t))
    assert len(bonds) == 108  # undirected count = 54
    # Explicit 27-site Hamiltonian term audit, including periodic copies of
    # B blocks. No use of the CTM observable summation in this check.
    lookup = {key:i for i, key in enumerate(keys)}
    def index(p):
        key = tuple(np.round(np.linalg.solve(period, p) % 1, 8) % 1)
        return lookup[key]
    field_count = np.zeros(27)
    blocked_bonds = []
    for c, (n, m) in enumerate(REPS['A']):
        pa = points('A', n, m)
        for k, (sn, sm) in enumerate(SHIFTS):
            pb = points('B', n+sn, m+sm)
            aa, bb, internal, cross = edge_support(k)
            blocked_bonds += [tuple(sorted((index(pa[i]), index(pa[j])))) for i, j in internal]
            blocked_bonds += [tuple(sorted((index(pa[i]), index(pb[j])))) for i, j in cross]
            for i in aa:
                field_count[index(pa[i])] += .5 if i % 2 == 0 else 1.
            for j in bb:
                field_count[index(pb[j])] += .5
    independently_counted = {tuple(sorted((i, j))) for i, j, s, t in bonds}
    assert len(blocked_bonds) == len(set(blocked_bonds)) == 54
    assert set(blocked_bonds) == independently_counted
    assert np.array_equal(field_count, np.ones(27))
    return dict(physical_sites=27, internal_hexagon_bonds=18, cross_block_bonds=36,
                total_bonds=54, coarse_edges=9, field_multiplicity_min=float(field_count.min()),
                field_multiplicity_max=float(field_count.max()))
