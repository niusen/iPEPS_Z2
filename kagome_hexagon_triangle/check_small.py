"""Small independent checks; no production scan is launched by this file."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import opt_einsum as oe
from .geometry import audit, edge_support, face_nodes, SHIFTS, neighbour
from .simple_update import BlockState, edge_hamiltonian, gate_factors, apply_local_factors, update_bond, sweep, rotate_tensor
from .honeycomb_ctm import HoneycombEnvironment, FaceEnvironment, layers, double_layer
from .observables import measure


def direct_patch(local, nodes):
    # Independent geometric contraction: connect physical coarse neighbours
    # by coordinate, not the CTM update index equations. Open bonds get I on
    # the single-layer virtual space (identity boundary of the double layer).
    links, label, args = {}, 0, []
    for kind, n, m in nodes:
        for k, (sn, sm) in enumerate(SHIFTS):
            other = ('B', n+sn, m+sm) if kind == 'A' else ('A', n-sn, m-sm)
            key = tuple(sorted(((kind, n, m), other)))
            if key not in links:
                links[key] = label
                label += 1
    for kind, n, m in nodes:
        legs = []
        for k, (sn, sm) in enumerate(SHIFTS):
            other = ('B', n+sn, m+sm) if kind == 'A' else ('A', n-sn, m-sm)
            key = tuple(sorted(((kind, n, m), other)))
            legs.append(links[key])
            if other not in nodes:
                d = int(round(local[(kind, (n-m)%3)].shape[0]**.5))
                args.extend([torch.eye(d, dtype=torch.float64).flatten()/d**.5, [links[key]]])
        args.extend([local[(kind, (n-m)%3)], legs])
    args.append([])
    return oe.contract(*args, backend='torch', optimize='auto-hq')


def run():
    torch.set_num_threads(10)
    result = dict(geometry=audit())
    # Operator Schmidt factors must reconstruct the exact five-spin gate.
    for k in range(3):
        ga, gb = gate_factors(k, .38, .03)
        rebuilt = torch.einsum('sij,skl->ikjl', ga, gb).reshape(32, 32)
        assert torch.allclose(rebuilt, torch.matrix_exp(-.03*edge_hamiltonian(k, .38)), atol=2e-14)
    result['five_spin_gate_reconstruction'] = 'passed'
    # Compare factor application to a full 64-dimensional embedded operator.
    from .simple_update import embed, SZ
    for k in range(3):
        aa, _, _, _ = edge_support(k)
        op = torch.kron(torch.kron(SZ, torch.eye(2)), SZ)[None]
        t = torch.randn((64, 2, 2, 2), dtype=torch.float64)
        y = apply_local_factors(t, op, aa, 6)[..., 0]
        expected = torch.einsum('pq,qabc->pabc', embed(SZ, aa[0], 6)@embed(SZ, aa[2], 6), t)
        assert torch.allclose(y, expected, atol=1e-13)
    result['physical_axis_gate_application'] = 'passed'
    # The truncated QR core must give the SAME retained spectrum as a
    # direct dense SVD of the fully evolved two-block matrix.
    st = BlockState.random(2, seed=980)
    c, k = 1, 2
    b = neighbour(c, k)
    order = [0]+[j+1 for j in range(3) if j != k]+[k+1]
    a, t = st.A[c].clone(), st.B[b].clone()
    for kind, colour, x in (('A', c, a), ('B', b, t)):
        for leg, w in enumerate(st.lambdas(kind, colour)):
            if leg != k:
                shape = [1]*4
                shape[leg+1] = len(w)
                x *= w.reshape(shape)
    ga, gb = gate_factors(k, st.field, .01)
    aa, bb, _, _ = edge_support(k)
    x = apply_local_factors(a.permute(order), ga, aa, 6)
    y = apply_local_factors(t.permute(order), gb, bb, 3)
    w = st.weights[c][k].sqrt().reshape(1, 1, 1, -1, 1)
    x = (x*w).reshape(256, -1)
    y = (y*w).reshape(32, -1)
    singular = torch.linalg.svdvals(x@y.T)[:2]
    update_bond(st, c, k, .01)
    assert torch.allclose(st.weights[c][k], singular/singular.norm(), atol=2e-13)
    result['QR_update_vs_direct_full_SVD'] = 'passed'
    # Check C3 constraints both before and after evolution. The rotated
    # copies include the physical spins, virtual axes AND bond weights.
    st = BlockState.random(2, seed=983, c3=True)
    for _ in range(3):
        for kind in ('A', 'B'):
            group = st.A if kind == 'A' else st.B
            shift = 1 if kind == 'A' else -1
            for c in range(3):
                assert torch.allclose(rotate_tensor(group[c], kind), group[(c+shift)%3], atol=1e-13)
                assert torch.allclose(rotate_tensor(st.dressed(kind, c), kind), st.dressed(kind, (c+shift)%3), atol=1e-13)
        sweep(st, .03)
    result['C3_physical_virtual_and_weight_rotations'] = 'passed'
    # Fully polarized product: every bond is +1/4, Eex/site=1/2, m=1.
    st = BlockState.random(1)
    for group in (st.A, st.B):
        for t in group:
            t.zero_()
            t[0, 0, 0, 0] = 1.
    loc = layers(st)
    env = HoneycombEnvironment.initial(1).run(loc, chi=4)
    obs = measure(st, env)
    assert abs(obs['energy_exchange']-.5) < 1e-12
    assert abs(obs['magnetization_normalized']-1) < 1e-12
    assert abs(obs['energy_total']-(.5-.38/2)) < 1e-12
    result['polarized_27_site_energy_and_field'] = 'passed'
    # A random D1 state is a product of six internally entangled clusters.
    # Its complete energy can be computed WITHOUT a tensor-network
    # environment, including factorized inter-block spin expectations.
    st1 = BlockState.random(1, seed=979)
    from .simple_update import SP, SM
    moments = {}
    for kind, p, n in (('A', 64, 6), ('B', 8, 3)):
        for c in range(3):
            vec = (st1.A if kind == 'A' else st1.B)[c].flatten()
            for i in range(n):
                for axis, op in enumerate((SZ, SP, SM)):
                    moments[kind, c, i, axis] = float(vec@(embed(op, i, n)@vec))
    exact_energy = 0.
    for c in range(3):
        vec = st1.A[c].flatten()
        for i in range(6):
            j = (i+1)%6
            h = embed(SZ, i, 6)@embed(SZ, j, 6)
            h += .5*(embed(SP, i, 6)@embed(SM, j, 6)+embed(SM, i, 6)@embed(SP, j, 6))
            exact_energy += float(vec@h@vec)
        for k in range(3):
            b = neighbour(c, k)
            for i, j in edge_support(k)[3]:
                exact_energy += sum(weight*moments['A', c, i, a]*moments['B', b, j, z]
                                    for a, z, weight in ((0, 0, 1.), (1, 2, .5), (2, 1, .5)))
    env1 = HoneycombEnvironment.initial(1).run(layers(st1), chi=4)
    obs1 = measure(st1, env1)
    assert abs(obs1['energy_exchange']-exact_energy/27) < 2e-13
    result['random_cluster_product_energy_independent_check'] = 'passed'
    # Random D=2, exact finite 18-node patch versus ONE untruncated CTM
    # growth: tests all six tensor colours, permutations, corner and edge
    # updates, and enlarged-cell environment translation together.
    st = BlockState.random(2, seed=981)
    loc = layers(st)
    nodes = {('A', 1, 0), ('B', 1, -1), ('A', 1, -1), ('B', 0, -1), ('A', 0, 0), ('B', 0, 0),
             ('B', 1, 0), ('A', 2, -1), ('B', 1, -2), ('A', 0, -1), ('B', -1, 0), ('A', 0, 1),
             ('A', 2, 0), ('B', 2, -1), ('A', 1, -2), ('B', 0, -2), ('A', -1, 1), ('B', -1, 1)}
    # CTM tensor normalization cancels in observable ratios, so test an
    # impurity rather than the partition function's arbitrary scale.
    env = HoneycombEnvironment.initial(2)
    env.run(loc, chi=16, max_iterations=1)
    from .honeycomb_ctm import FaceMeasurement
    op = embed(SZ, 0, 6)
    imp = double_layer(st.dressed('A', 1), op)
    ct = FaceMeasurement(env.faces[0], loc, 0).expectation({('A', 1): imp})
    # A colour repeats in this patch. Insert at the central A(1,0) only,
    # by giving each coordinate a distinct temporary colour to the helper.
    def exact_with_impurity(replacements=None):
        replacements = replacements or {}
        indexed = {}
        args, labels, nextlabel = [], {}, 0
        for kind, n, m in sorted(nodes):
            legs = []
            for k, (sn, sm) in enumerate(SHIFTS):
                other = ('B', n+sn, m+sm) if kind == 'A' else ('A', n-sn, m-sm)
                key = tuple(sorted(((kind, n, m), other)))
                if key not in labels:
                    labels[key] = nextlabel
                    nextlabel += 1
                legs.append(labels[key])
                if other not in nodes:
                    args.extend([torch.eye(2, dtype=torch.float64).flatten()/2**.5, [labels[key]]])
            tensor = replacements.get((kind, n, m), loc[(kind, (n-m)%3)])
            args.extend([tensor, legs])
        return oe.contract(*args, [], backend='torch', optimize='auto-hq')
    norm_exact = exact_with_impurity()
    exact = float(exact_with_impurity({('A', 1, 0):imp})/norm_exact)
    result['native_ctm_vs_exact_18_node_patch_error'] = abs(ct-exact)
    assert abs(ct-exact) < 2e-10, (ct, exact)
    bimp = double_layer(st.dressed('B', 2), embed(SZ, 1, 3))
    aimp2 = double_layer(st.dressed('A', 2), embed(SZ, 3, 6))
    measurement = FaceMeasurement(env.faces[0], loc, 0)
    errors = []
    for replacements, coordinates in (({('A', 1):imp, ('B', 2):bimp}, {('A', 1, 0):imp, ('B', 1, -1):bimp}),
                                      ({('A', 2):aimp2, ('B', 2):bimp}, {('A', 1, -1):aimp2, ('B', 1, -1):bimp})):
        ctm_value = measurement.expectation(replacements)
        exact_value = float(exact_with_impurity(coordinates)/norm_exact)
        errors.append(abs(ctm_value-exact_value))
    assert max(errors) < 2e-10
    result['same_and_different_corner_bond_impurity_errors'] = errors
    # A second independent finite-patch check, after two CTM growths, catches
    # errors hidden by the initially one-dimensional environment legs.
    from functools import lru_cache
    outward = ((1, 0), (0, -1), (-1, 1))
    def central(n, m):
        return (('A', n+1, m), ('B', n+1, m-1), ('A', n+1, m-1),
                ('B', n, m-1), ('A', n, m), ('B', n, m))
    @lru_cache(None)
    def support(part, i, n, m, depth):
        if depth == 0:
            return frozenset()
        which = (i+1)%3 if part == 'Tb' else i
        sn, sm = outward[which]
        qn, qm = n+sn, m+sm
        ring = central(qn, qm)
        if part == 'C':
            return (support('C', i, qn, qm, depth-1) |
                    support('Ta', i, qn, qm, depth-1) |
                    support('Tb', (i-1)%3, qn, qm, depth-1) |
                    {ring[2*i], ring[2*i+1]})
        if part == 'Ta':
            return support('Tb', i, qn, qm, depth-1) | {ring[(2*i+2)%6]}
        return support('Ta', i, qn, qm, depth-1) | {ring[2*i+1]}
    nodes = set(central(0, 0))
    for part in ('C', 'Ta', 'Tb'):
        for i in range(3):
            nodes |= support(part, i, 0, 0, 2)
    assert len(nodes) == 36
    env2 = HoneycombEnvironment.initial(2).run(loc, chi=64, max_iterations=2)
    ct2 = FaceMeasurement(env2.faces[0], loc, 0).expectation({('A', 1):imp})
    ex2 = float(exact_with_impurity({('A', 1, 0):imp})/exact_with_impurity())
    result['native_ctm_vs_exact_36_node_patch_error'] = abs(ct2-ex2)
    assert abs(ct2-ex2) < 2e-10, (ct2, ex2)
    # Small timed evolution; full accuracy/truncation test below compares
    # QR/SVD update with a direct large matrix SVD through its singular values.
    st = BlockState.random(2, seed=982)
    t0 = time.perf_counter()
    for _ in range(4):
        sweep(st, .03)
    result['D2_seconds_per_symmetric_sweep'] = (time.perf_counter()-t0)/4
    assert all(torch.isfinite(t).all() for t in st.A+st.B)
    result['small_evolution'] = 'passed'
    out = Path(__file__).parent/'results'/'checks.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    run()
