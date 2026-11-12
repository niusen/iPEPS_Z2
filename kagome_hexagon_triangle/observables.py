"""Local magnetization and every physical bond, from native honeycomb CTM."""
import torch
from .geometry import edge_bonds, face_nodes, neighbour
from .simple_update import SZ, SP, SM, embed
from .honeycomb_ctm import layers, double_layer, FaceMeasurement


@torch.no_grad()
def measure(state, env):
    local = layers(state)
    measurements = [FaceMeasurement(env.faces[c], local, c) for c in range(3)]
    sites, bonds, repeated = [], [], []
    impurity_cache = {}

    def impurity(kind, c, i, axis):
        key = (kind, c, i, axis)
        if key not in impurity_cache:
            op = embed((SZ, SP, SM)[axis], i, 6 if kind == 'A' else 3)
            impurity_cache[key] = double_layer(state.dressed(kind, c), op)
        return impurity_cache[key]

    # Measure each block once for the reported value, but compare its Sz in
    # all three face environments as an independent consistency diagnostic.
    for kind in ('A', 'B'):
        for c in range(3):
            for i in range(6 if kind == 'A' else 3):
                values = [m.expectation({(kind, c): impurity(kind, c, i, 0)}) for m in measurements]
                sx = sum(.5*(m.expectation({(kind, c): impurity(kind, c, i, 1)})+
                             m.expectation({(kind, c): impurity(kind, c, i, 2)})) for m in measurements)/3
                sites.append(dict(kind=kind, colour=c, site=i, sz=sum(values)/3, sx=sx))
                repeated.append(max(values)-min(values))
    for c in range(3):
        # Six hexagon-internal bonds, counted once each.
        for i in range(6):
            j = (i+1)%6
            op = embed(SZ, i, 6)@embed(SZ, j, 6)
            op += .5*(embed(SP, i, 6)@embed(SM, j, 6)+embed(SM, i, 6)@embed(SP, j, 6))
            imp = double_layer(state.dressed('A', c), op)
            value = sum(m.expectation({('A', c): imp}) for m in measurements)/3
            bonds.append(dict(kind='internal', a=c, k=-1, i=i, b=c, j=j, energy=value))
        for k in range(3):
            b = neighbour(c, k)
            # This edge must be an actual CENTRAL HEXAGON EDGE of the face.
            # Merely having both nodes in the face is insufficient: its third
            # leg may connect the periodic copy outside the patch.
            faces = []
            for f in range(3):
                nodes = face_nodes(f)
                edge_legs = (2, 0, 1, 2, 0, 1)
                for e in range(6):
                    left, right = nodes[e], nodes[(e+1)%6]
                    if {left, right} == {('A', c), ('B', b)} and edge_legs[e] == k:
                        faces.append(f)
            assert len(faces) == 2
            for i, j in edge_bonds(k):
                values = []
                for f in faces:
                    val = 0.
                    for ia, ib, weight in ((0, 0, 1.), (1, 2, .5), (2, 1, .5)):
                        val += weight*measurements[f].expectation({('A', c): impurity('A', c, i, ia),
                                                                  ('B', b): impurity('B', b, j, ib)})
                    values.append(val)
                bonds.append(dict(kind='cross', a=c, k=k, i=i, b=b, j=j, energy=sum(values)/len(values)))
                repeated.append(max(values)-min(values))
            # Bound memory: reuse impurity corners within this coarse edge,
            # then release them before moving to the next edge.
            for m in measurements:
                m.impurity_corners.clear()
                m.impurity_references.clear()
    assert len(sites) == 27 and len(bonds) == 54
    sz = sum(r['sz'] for r in sites)/27
    exchange = sum(r['energy'] for r in bonds)/27
    return dict(D=state.D, field=state.field, seed=state.seed, steps=state.steps,
                energy_exchange=exchange, energy_total=exchange-state.field*sz,
                mean_sz=sz, magnetization_normalized=2*sz, sites=sites, bonds=bonds,
                max_abs_sx=max(abs(r['sx']) for r in sites),
                face_consistency_max=max(repeated), ctm_iterations=env.iterations,
                ctm_converged=env.converged, ctm_residual=env.residual)
