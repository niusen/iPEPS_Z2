"""Complete real-state observables, using the existing honeycomb environment.

S_i are spin 1/2, H=sum S.S-h*sum Sz. Coordinates use the original Julia
B1(R)=R-a1/2, B2(R)=R, B3(R)=R-a2/2 convention. x,y are one based.
YY is evaluated through real Sp/Sm combinations; no imaginary operator is
cast into the real double-layer routine. Sy and scalar chirality vanish
identically for this real-valued ansatz and are labelled accordingly.
"""
from functools import lru_cache
import numpy as np
import torch
from ..geometry import A1, A2, REPS, SHIFTS, points, neighbour, edge_support, face_nodes
from ..simple_update import SZ, SP, SM, embed
from ..honeycomb_ctm import layers, double_layer, FaceMeasurement

PERIOD = np.column_stack((3*A1, 3*A2))
SX = (SP+SM)/2


def position_key(p):
    return tuple(np.round(np.linalg.solve(PERIOD, p) % 1, 8) % 1)


@lru_cache(None)
def site_layout():
    original = {}
    for x in range(3):
        for y in range(3):
            for s, basis in enumerate((-A1/2, np.zeros(2), -A2/2), 1):
                original[position_key(x*A1+y*A2+basis)] = (x+1, y+1, f'B{s}')
    layout = {}
    for kind, p in (('A', 6), ('B', 3)):
        for c in range(3):
            for i, pos in enumerate(points(kind, *REPS[kind][c])):
                x, y, sub = original[position_key(pos)]
                layout[kind, c, i] = dict(site_id=f'{x},{y},{sub}', x=x, y=y, sublattice=sub,
                                          cartesian_x=float(pos[0]), cartesian_y=float(pos[1]))
    assert len({r['site_id'] for r in layout.values()}) == 27
    return layout


def edge_faces(c, k):
    b = neighbour(c, k)
    matches = []
    for f in range(3):
        ns = face_nodes(f)
        for e, leg in enumerate((2, 0, 1, 2, 0, 1)):
            if leg == k and {ns[e], ns[(e+1)%6]} == {('A', c), ('B', b)}:
                matches.append(f)
    assert len(matches) == 2
    return matches


def triangle_catalogue():
    rows = []
    for c, (n, m) in enumerate(REPS['A']):
        for k, (sn, sm) in enumerate(SHIFTS):
            aa, bb, internal, cross = edge_support(k)
            b = neighbour(c, k)
            pa, pb = points('A', n, m), points('B', n+sn, m+sm)
            for i, j in internal:
                for t in bb:
                    if (i, t) not in cross or (j, t) not in cross:
                        continue
                    pp = np.array([pa[i], pa[j], pb[t]])
                    # Original Tup points down in these Cartesian axes;
                    # Tdn points up. Name the tensor convention, not a view.
                    yy = np.sort(pp[:, 1])
                    orientation = 'Tup' if abs(yy[1]-yy[2]) < 1e-8 else 'Tdn'
                    rows.append(dict(a=c, k=k, b=b, i=i, j=j, t=t,
                                     orientation=orientation,
                                     centre_x=float(pp[:, 0].mean()), centre_y=float(pp[:, 1].mean())))
    assert len(rows) == 18
    assert sum(r['orientation'] == 'Tup' for r in rows) == 9
    return rows


@torch.no_grad()
def measure_complete(state, env, progress=None, max_corner_cache_bytes=256*1024**2):
    if any(t.is_complex() for t in state.A+state.B):
        raise ValueError('These full observables are for the real-valued blocked ansatz')
    local = layers(state)
    measurements = [FaceMeasurement(env.faces[c], local, c) for c in range(3)]
    layout = site_layout()
    tensor_cache, site_map, bonds, triangles, consistency = {}, {}, [], [], []

    def impurity(kind, c, axes):
        axes = tuple(axes)
        key = kind, c, axes
        if key not in tensor_cache:
            n = 6 if kind == 'A' else 3
            op = torch.eye(2**n, dtype=torch.float64)
            for i, axis in axes:
                op = op@embed({'z':SZ, 'p':SP, 'm':SM, 'x':SX}[axis], i, n)
            tensor_cache[key] = double_layer(state.dressed(kind, c), op)
        return tensor_cache[key]

    def average(replacements, faces=range(3)):
        values = [measurements[f].expectation(replacements) for f in faces]
        consistency.append(max(values)-min(values))
        used = sum(t.numel()*t.element_size() for m in measurements for t in m.impurity_corners.values())
        if used > max_corner_cache_bytes:
            clear_corners()
        return sum(values)/len(values)

    def clear_corners():
        for m in measurements:
            m.impurity_corners.clear()
            m.impurity_references.clear()

    for kind, n in (('A', 6), ('B', 3)):
        for c in range(3):
            for i in range(n):
                z = average({(kind, c):impurity(kind, c, ((i, 'z'),))})
                x = average({(kind, c):impurity(kind, c, ((i, 'x'),))})
                site_map[kind, c, i] = dict(kind=kind, colour=c, site=i, **layout[kind, c, i],
                                           sz=z, sx=x, sy=0., m_normalized=2*z,
                                           spin_length=float(np.hypot(x, z)),
                                           purity=.5+2*(x*x+z*z))
    if progress:
        progress('27 site magnetizations measured')

    def append_bond(kind, c, k, i, b, j, xx, yy, zz):
        left = site_map['A', c, i]
        right = site_map['A' if kind == 'internal' else 'B', b, j]
        sxprod, szprod = left['sx']*right['sx'], left['sz']*right['sz']
        bonds.append(dict(kind=kind, a=c, k=k, i=i, b=b, j=j,
                          site_id_1=left['site_id'], site_id_2=right['site_id'],
                          sxx=xx, syy=yy, szz=zz, energy=xx+yy+zz,
                          connected_sxx=xx-sxprod, connected_syy=yy,
                          connected_szz=zz-szprod, connected_dot=xx+yy+zz-sxprod-szprod))

    for c in range(3):
        for i in range(6):
            j = (i+1)%6
            zz = average({('A', c):impurity('A', c, ((i, 'z'), (j, 'z')))})
            xx = average({('A', c):impurity('A', c, ((i, 'x'), (j, 'x')))})
            # (Sy_i Sy_j) is a REAL two-spin operator on a real state.
            yy = 0.
            for ai, aj, weight in (('p','m',.25), ('m','p',.25), ('p','p',-.25), ('m','m',-.25)):
                yy += weight*average({('A', c):impurity('A', c, ((i, ai), (j, aj)))})
            append_bond('internal', c, -1, i, c, j, xx, yy, zz)
        for k in range(3):
            b, faces = neighbour(c, k), edge_faces(c, k)
            aa, bb, internal, cross = edge_support(k)
            for i, j in cross:
                terms = {}
                for ai, aj in (('z','z'), ('p','p'), ('p','m'), ('m','p'), ('m','m')):
                    terms[ai, aj] = average({('A', c):impurity('A', c, ((i, ai),)),
                                            ('B', b):impurity('B', b, ((j, aj),))}, faces)
                xx = .25*sum(terms[a] for a in (('p','p'), ('p','m'), ('m','p'), ('m','m')))
                yy = .25*(terms['p','m']+terms['m','p']-terms['p','p']-terms['m','m'])
                append_bond('cross', c, k, i, b, j, xx, yy, terms['z','z'])
            # Every elementary triangle spans exactly one A and one B.
            for tri in (r for r in triangle_catalogue() if r['a'] == c and r['k'] == k):
                i, j, t = tri['i'], tri['j'], tri['t']
                zzz = average({('A', c):impurity('A', c, ((i,'z'), (j,'z'))),
                               ('B', b):impurity('B', b, ((t,'z'),))}, faces)
                triangles.append(tri | dict(szszsz=zzz))
            clear_corners()
            if progress:
                progress(f'coarse edge A{c}/leg{k}: bonds and triangles measured ({3*c+k+1}/9)')
        # Internal two-spin operators are not needed after finishing A_c.
        tensor_cache = {key:value for key,value in tensor_cache.items() if len(key[2]) == 1}

    bond_lookup = {frozenset((r['site_id_1'], r['site_id_2'])):r for r in bonds}
    for t in triangles:
        ids = [layout['A', t['a'], t['i']]['site_id'], layout['A', t['a'], t['j']]['site_id'],
               layout['B', t['b'], t['t']]['site_id']]
        sites_by_id = {r['site_id']:r for r in site_map.values()}
        zs = [sites_by_id[i]['sz'] for i in ids]
        pairs = [bond_lookup[frozenset((ids[i], ids[j]))] for i,j in ((0,1),(0,2),(1,2))]
        anchor = next(sites_by_id[s] for s in ids if sites_by_id[s]['sublattice'] == 'B2')
        t.update(triangle_id=f"{t['orientation']}({anchor['x']},{anchor['y']})", x=anchor['x'], y=anchor['y'],
                 site_id_1=ids[0], site_id_2=ids[1], site_id_3=ids[2],
                 sum_sz=sum(zs), mean_sz=sum(zs)/3, m_normalized=2*sum(zs)/3,
                 exchange_energy=sum(p['energy'] for p in pairs),
                 sum_sxx=sum(p['sxx'] for p in pairs), sum_syy=sum(p['syy'] for p in pairs),
                 sum_szz=sum(p['szz'] for p in pairs),
                 sum_sz_variance=.75+2*sum(p['szz'] for p in pairs)-sum(zs)**2,
                 connected_szszsz=t['szszsz']-zs[0]*pairs[2]['szz']-zs[1]*pairs[1]['szz']-
                                     zs[2]*pairs[0]['szz']+2*zs[0]*zs[1]*zs[2])
    sites = list(site_map.values())
    assert len(sites) == 27 and len(bonds) == 54 and len(triangles) == 18
    assert len(bond_lookup) == 54
    eex = sum(r['energy'] for r in bonds)/27
    assert abs(sum(r['exchange_energy'] for r in triangles)/27-eex) < 1e-12
    mz = sum(r['sz'] for r in sites)/27
    spectra = []
    for c in range(3):
        for k in range(3):
            probabilities = state.weights[c][k].square()
            entropy = float(-(probabilities*probabilities.clamp_min(1e-300).log()).sum())
            for j, (w, p) in enumerate(zip(state.weights[c][k], probabilities)):
                spectra.append(dict(a=c, k=k, index=j, schmidt_weight=float(w), probability=float(p),
                                    bond_entropy=entropy))
    corners = [C for face in env.faces for C in face.C]
    ranks = []
    for C in corners:
        s = torch.linalg.svdvals(C)
        ranks.append(int((s > 1e-12*s[0]).sum()))
    return dict(D=state.D, field=state.field, seed=state.seed, steps=state.steps,
                energy_exchange=eex, energy_field=-state.field*mz, energy_total=eex-state.field*mz,
                mean_sz=mz, magnetization_normalized=2*mz, distance_to_one_ninth=2*mz-1/9,
                max_abs_sx=max(abs(r['sx']) for r in sites), face_consistency_max=max(consistency),
                ctm_converged=env.converged, ctm_iterations=env.iterations, ctm_residual=env.residual,
                corner_dimension_min=min(C.shape[0] for C in corners),
                corner_dimension_max=max(C.shape[0] for C in corners),
                corner_numerical_rank_min=min(ranks), corner_numerical_rank_max=max(ranks),
                real_tensor_ansatz=True, scalar_chirality_fixed_by_real_ansatz=0.,
                sites=sites, bonds=bonds, triangles=triangles, schmidt_spectra=spectra)
