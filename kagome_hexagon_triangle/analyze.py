"""Plot physical sites/bonds and compare spatial patterns with author data.

Requires numpy and matplotlib; no torch needed. This is a scientific static
figure. The D15 reference is a reconstruction from Fig.S3, not a wavefunction.
"""
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .geometry import A1, A2, C0, REPS, points, edge_bonds, SHIFTS

ROOT = Path(__file__).parent
RESULTS = ROOT/'results'/'D5_D6_c3_h038'
PERIOD = np.column_stack((3*A1, 3*A2))


def canonical(p):
    x = np.linalg.solve(PERIOD, p)
    return tuple(np.round(x % 1, 7) % 1)


def site_position(row):
    return points(row['kind'], *REPS[row['kind']][int(row['colour'])])[int(row['site'])]


def reference(D=15):
    data = json.loads((ROOT/'references'/'paper_D15_reference.json').read_text())
    basis = {'B1':-A1/2, 'B2':np.zeros(2), 'B3':-A2/2}
    ps = [(r['x']-1)*A1+(r['y']-1)*A2+basis[r['site']] for r in data['sites']]
    if D == 15:
        vals = np.array([r['Sz_field_aligned'] for r in data['sites']])
    else:
        from fractions import Fraction
        rows = np.loadtxt(ROOT/'references'/'data_fig_S3_mag_vs_D.dat', dtype=str, skiprows=1)
        row = next(row for row in rows if int(1/Fraction(row[0])) == D)
        vals = np.array([float(row[r['paper_label']]) for r in data['sites']])
    return np.array(ps), vals


def transformed_values(ps, values, dest, theta, reflect=False, shift=(0, 0), centre=C0):
    angle = theta*np.pi/3
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if reflect:
        rot = rot@np.diag([1., -1.])
    transformed = (ps-centre)@rot.T+centre+shift[0]*A1+shift[1]*A2
    mapping = {canonical(p):v for p, v in zip(transformed, values)}
    if len(mapping) != 27 or any(canonical(p) not in mapping for p in dest):
        return None
    return np.array([mapping[canonical(p)] for p in dest])


def compare(obs, reference_D=15):
    ps = np.array([site_position(r) for r in obs['sites']])
    sz = np.array([r['sz'] for r in obs['sites']])
    rp, rv = reference(reference_D)
    best = None
    for theta in range(6):
        for reflect in (False, True):
            for x in range(3):
                for y in range(3):
                    aligned = transformed_values(rp, rv, ps, theta, reflect, (x, y))
                    if aligned is None:
                        continue
                    rms = float(np.sqrt(np.mean((sz-aligned)**2)))
                    if best is None or rms < best[0]:
                        best = (rms, theta, reflect, x, y, aligned)
    assert best is not None
    symmetry = {}
    for label, theta, reflect in (('C3', 2, False), ('mirror', 0, True)):
        residuals = []
        angles = [theta] if label == 'C3' else list(range(6))
        for t in angles:
            for x in range(3):
                for y in range(3):
                    vals = transformed_values(ps, sz, ps, t, reflect, (x, y))
                    if vals is not None:
                        residuals.append(float(np.sqrt(np.mean((sz-vals)**2))))
        symmetry[label+'_best_rms_sz'] = min(residuals)
    info = dict(paper_D15_best_spatial_rms_sz=best[0],
                paper_D15_sorted_rms_sz=float(np.sqrt(np.mean((np.sort(sz)-np.sort(rv))**2))),
                paper_D15_best_spatial_correlation=float(np.corrcoef(sz, best[-1])[0, 1]),
                reference_alignment=dict(rotation_60deg_units=best[1], reflection=best[2], translation=[best[3], best[4]]),
                **symmetry)
    return ps, sz, best[-1], info


def plot_panel(ax, ps, sz, title, bond_map=None):
    # Draw one rectangular 3x3 primitive-cell view, including nearby copies
    # for bonds at the boundary. Every marker value is field-aligned <Sz>.
    coeff = np.linalg.solve(PERIOD, ps.T).T % 1
    coeff[np.isclose(coeff, 1, atol=1e-9)] = 0
    displayed = coeff@PERIOD.T
    copies = []
    for i, p in enumerate(displayed):
        for x in range(-1, 2):
            for y in range(-1, 2):
                copies.append((i, p+x*PERIOD[:, 0]+y*PERIOD[:, 1]))
    for i, p in enumerate(displayed):
        for j, q in copies:
            if abs(np.linalg.norm(p-q)-.5) < 1e-7:
                value = bond_map.get(tuple(sorted((i, j))), -.2) if bond_map else -.2
                ax.plot([p[0], q[0]], [p[1], q[1]], color='#52606c',
                        lw=(.5+4*abs(value)) if bond_map else .65, alpha=.8, zorder=1)
    ax.scatter(displayed[:, 0], displayed[:, 1], c=sz, cmap='coolwarm',
               norm=TwoSlopeNorm(vmin=-.35, vcenter=0, vmax=.35),
               s=60+950*np.abs(sz), edgecolors='#28313a', linewidths=.6, zorder=3)
    for p, v in zip(displayed, sz):
        ax.annotate(f'{v:.3f}', p, xytext=(0, 10), textcoords='offset points', ha='center', fontsize=6)
    corners = np.array([[0, 0], PERIOD[:, 0], PERIOD.sum(axis=1), PERIOD[:, 1], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], '--', color='#8f9da7', lw=.8)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.axis('off')


def bond_symmetry(ps, energies, reflect):
    lookup = {canonical(p):i for i, p in enumerate(ps)}
    best = float('inf')
    for t in (range(6) if reflect else (2,)):
        angle = t*np.pi/3
        mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        if reflect:
            mat = mat@np.diag([1., -1.])
        for x in range(3):
            for y in range(3):
                transformed = (ps-C0)@mat.T+C0+x*A1+y*A2
                if any(canonical(p) not in lookup for p in transformed):
                    continue
                mapping = [lookup[canonical(p)] for p in transformed]
                rms = np.sqrt(np.mean([(v-energies[tuple(sorted((mapping[i], mapping[j])))])**2
                                       for (i, j), v in energies.items()]))
                best = min(best, float(rms))
    return best


def main():
    files = sorted(RESULTS.glob('D*_seed*/*observables_chi*.json'))
    files += sorted((ROOT/'results'/'D5_continuation_h038').glob('D*_seed*/*observables_chi*.json'))
    files += sorted((ROOT/'results'/'D6_continuation_h038').glob('D*_seed*/*observables_chi*.json'))
    runs = {}
    allrows = []
    for f in files:
        obs = json.loads(f.read_text())
        obs['run_directory'] = f.parent.parent.name
        obs['measurement_file'] = f.name
        allrows.append({k:v for k,v in obs.items() if k not in ('sites', 'bonds')})
        key = str(f.parent)
        if key not in runs or obs['chi'] > runs[key]['chi'] or (obs['chi'] == runs[key]['chi'] and f.name.startswith('verified')):
            runs[key] = obs
    # First select the largest chi for EACH saved state; then choose the
    # lowest total energy at this fixed field for the figure. Keep all runs
    # and all chi in the comparison table, including the continuation.
    best_per_D = {}
    for obs in runs.values():
        D = obs['D']
        if D not in best_per_D or obs['energy_total'] < best_per_D[D]['energy_total']:
            best_per_D[D] = obs
    with (RESULTS/'all_measurements.csv').open('w', newline='') as f:
        keys = list(dict.fromkeys(key for r in allrows for key in r))
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(allrows)
    runs = best_per_D
    if not runs:
        raise RuntimeError('No completed measurements yet')
    summaries = []
    fig, axes = plt.subplots(len(runs), 3, figsize=(17, 5*len(runs)), squeeze=False)
    for row, (key, obs) in enumerate(sorted(runs.items())):
        ps, sz, aligned, comparison = compare(obs)
        _, _, aligned7, comparison7 = compare(obs, 7)
        comparison['paper_D7_best_spatial_rms_sz'] = comparison7['paper_D15_best_spatial_rms_sz']
        comparison['paper_D7_best_spatial_correlation'] = comparison7['paper_D15_best_spatial_correlation']
        lookup = {canonical(p):i for i, p in enumerate(ps)}
        bonds = {}
        for r in obs['bonds']:
            pa = points('A', *REPS['A'][r['a']])[r['i']]
            if r['kind'] == 'internal':
                pb = points('A', *REPS['A'][r['a']])[r['j']]
            else:
                n, m = REPS['A'][r['a']]
                sn, sm = SHIFTS[r['k']]
                pb = points('B', n+sn, m+sm)[r['j']]
            bonds[tuple(sorted((lookup[canonical(pa)], lookup[canonical(pb)])))] = r['energy']
        assert len(bonds) == 54
        comparison['C3_best_rms_bond'] = bond_symmetry(ps, bonds, False)
        comparison['mirror_best_rms_bond'] = bond_symmetry(ps, bonds, True)
        plot_panel(axes[row, 0], ps, sz,
                   f"Our blocked iPEPS D={obs['D']}, h={obs['field']}, chi={obs['chi']}\n"
                   f"Eex={obs['energy_exchange']:.8f}, m/ms={obs['magnetization_normalized']:.7f}", bonds)
        plot_panel(axes[row, 1], ps, aligned7, 'Paper D=7, h=0.4: closest available local-Sz data\nBest space-group alignment; uniform bonds for orientation')
        plot_panel(axes[row, 2], ps, aligned, 'Paper D=15, h=0.4: best space-group alignment\nReference local Sz only; bond values unavailable')
        summaries.append({k:v for k,v in obs.items() if k not in ('sites', 'bonds')} | comparison)
    fig.suptitle('Ghosh--Mila blocking: actual site magnetization and nearest-neighbour bonds', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(RESULTS/'patterns.png', dpi=180)
    plt.close(fig)
    (RESULTS/'pattern_comparison.json').write_text(json.dumps(summaries, indent=2)+'\n')
    from fractions import Fraction
    energy_paper = np.loadtxt(ROOT/'references'/'data_fig_2b_energy_vs_D.dat', dtype=str, skiprows=1)
    refs = {int(1/Fraction(d)):float(e) for d,e in energy_paper}
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
                   (r['D'], r['chi'], f"{r['energy_exchange']:.9f}", refs.get(r['D']),
                    f"{r['magnetization_normalized']:.8f}", f"{r.get('max_abs_sx', float('nan')):.5f}", r['ctm_converged'],
                    f"{r['face_consistency_max']:.2e}", f"{r['paper_D15_best_spatial_rms_sz']:.4f}"))+'</tr>' for r in summaries)
    page = '''<!doctype html><meta charset="utf-8"><title>Blocked honeycomb iPEPS experiment</title>
<style>body{max-width:1200px;margin:35px auto;font:16px/1.6 system-ui;color:#243443;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd5dd;text-align:right}img{width:100%}.note{background:#edf2f6;padding:16px}</style>
<h1>六边形＋三自旋块 iPEPS：首次复现测试</h1>
<p>新文件独立实现：27 spins，6 个块，3 个角与 6 个边的原生蜂窝 CTMRG。C3 模式使用两个独立张量的旋转副本，同时旋转物理腿与虚拟腿。没有固定总磁化，没有加入目标局域磁化图案。</p>
<p class="note">交换能量 Eex 不含磁场；本实现 H=ΣS·S−hΣSz，Etot=Eex−h(m/ms)/2。论文参考能量属于 1/9 平台，图案参考为 D7/D15、h=0.4；本次在 h=0.38。若本次磁化偏离 1/9，则能量不能作为同一磁化扇区的直接优劣比较。一个随机种子和有限演化时间不能代表全局最优。横向磁化 Sx 的残留也需要继续检查，不能凭一个磁场点声称得到严格平台。</p>
<table><tr><th>D</th><th>χ</th><th>本次 Eex</th><th>论文 Eex</th><th>m/ms</th><th>max |Sx|</th><th>CTM 收敛</th><th>环境位置误差</th><th>对齐后 Sz RMS</th></tr>'''+table+'''</table>
<img src="patterns.png"><p>圆圈颜色与面积：带符号的局域 Sz；左图键宽：|〈Si·Sj〉|。右图只展示论文局域磁化，不伪造其未提供的逐键数值。空间比较遍历平移、旋转和反射；并非只将数值排序。</p>
<p><a href="all_measurements.csv">各轮与各 χ 的测量表</a> · <a href="pattern_comparison.json">对称性与图案比较</a> · <a href="https://arxiv.org/html/2610.34439v1">目标论文</a> · <a href="https://arxiv.org/html/2306.09046v2">蜂窝 CTMRG 方法</a></p>'''
    (RESULTS/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(summaries, indent=2))
    print(RESULTS/'report.html')


if __name__ == '__main__':
    main()
