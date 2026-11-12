"""Read downloaded CSVs, without loading states or running tensor contractions.

Run as a module from iPEPS_Z2. Raw server data and algorithm files are untouched.
Paper local-Sz values come from the author's data; their site-label geometry
is reconstructed from Fig. S3 and tested over all 108 kagome space operations.
The sorted comparison does not depend on that reconstruction.
"""
import csv
import hashlib
import html
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analyze import (ROOT, PERIOD, canonical, compare, reference,
                      transformed_values, plot_panel)
from .geometry import A1, A2, C0, OFFSETS, points, REPS

INPUT = ROOT/'results'/'server_D4_D10_h038_C3_chain'
OUTPUT = INPUT/'analysis_20261101'
PAPER = 'https://arxiv.org/html/2610.34439v1'
DATA = 'https://doi.org/10.6084/m9.figshare.34731234.v3'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    return float(row[key])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def permutation(ps, theta=0, reflect=False, shift=(0, 0), centre=C0):
    angle = theta*np.pi/3
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    if reflect:
        matrix = matrix@np.diag([1., -1.])
    transformed = (ps-centre)@matrix.T+centre+shift[0]*A1+shift[1]*A2
    lookup = {canonical(p):i for i, p in enumerate(ps)}
    assert len(lookup) == 27
    indices = np.array([lookup[canonical(p)] for p in transformed])
    assert len(set(indices)) == 27
    return indices


def symmetry(ps, sz, bonds):
    records = []
    for theta in range(6):
        for reflect in (False, True):
            for x in range(3):
                for y in range(3):
                    perm = permutation(ps, theta, reflect, (x, y))
                    delta = [v-bonds[tuple(sorted((perm[i], perm[j])))]
                             for (i, j), v in bonds.items()]
                    records.append(dict(theta=theta, reflection=reflect, x=x, y=y,
                                        sz_rms=rms(sz-sz[perm]),
                                        sz_max=float(np.max(np.abs(sz-sz[perm]))),
                                        bond_rms=rms(delta)))
    assert len(records) == 108
    result = {}
    for name, predicate in (
        ('C3', lambda r:r['theta'] == 2 and not r['reflection']),
        ('C6', lambda r:r['theta'] == 1 and not r['reflection']),
        ('mirror', lambda r:r['reflection']),
        ('nontrivial_translation', lambda r:r['theta'] == 0 and not r['reflection']
         and (r['x'] != 0 or r['y'] != 0)),
    ):
        candidates = [r for r in records if predicate(r)]
        best = min(candidates, key=lambda r:r['sz_rms'])
        result[name] = dict(best_by_sz=best,
                            minimum_bond_rms=min(r['bond_rms'] for r in candidates))
    return result


def paper_compare(sites, D):
    _, _, aligned, old = compare({'sites':sites}, D)
    return dict(reference_D=D, reference_field=.4,
                spatial_rms_sz=old['paper_D15_best_spatial_rms_sz'],
                sorted_rms_sz=old['paper_D15_sorted_rms_sz'],
                correlation=old['paper_D15_best_spatial_correlation'],
                alignment=old['reference_alignment']), aligned


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {name:INPUT/f'{name}.csv' for name in
             ('summary', 'site_magnetization', 'bond_energies',
              'triangle_observables', 'schmidt_spectra')}
    fingerprints = {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    source = {name:read_csv(p) for name,p in paths.items()}
    paper_energy = {int(1/Fraction(d)):float(e) for d,e in
                    np.loadtxt(ROOT/'references'/'data_fig_2b_energy_vs_D.dat',
                               dtype=str, skiprows=1)}
    snapshots, table, audit, symmetries, comparisons, hexagons = {}, [], [], [], [], []
    for row in source['summary']:
        D = int(row['D'])
        sites = [dict(r, sz=f(r,'sz')) for r in source['site_magnetization'] if int(r['D']) == D]
        bonds_raw = [r for r in source['bond_energies'] if int(r['D']) == D]
        triangles = [r for r in source['triangle_observables'] if int(r['D']) == D]
        spectra = [r for r in source['schmidt_spectra'] if int(r['D']) == D]
        assert len(sites) == 27 and len(bonds_raw) == 54 and len(triangles) == 18
        ps = np.array([[f(r,'cartesian_x'), f(r,'cartesian_y')] for r in sites])
        sz = np.array([r['sz'] for r in sites])
        for r,p in zip(sites, ps):
            expected = points(r['kind'], *REPS[r['kind']][int(r['colour'])])[int(r['site'])]
            assert canonical(expected) == canonical(p)
        ids = {r['site_id']:i for i,r in enumerate(sites)}
        assert len(ids) == 27
        bonds = {tuple(sorted((ids[r['site_id_1']], ids[r['site_id_2']]))):f(r,'energy')
                 for r in bonds_raw}
        assert len(bonds) == 54
        degree = np.zeros(27, dtype=int)
        for i,j in bonds:
            degree[i] += 1
            degree[j] += 1
            dif = ps[i]-ps[j]
            distance = min(np.linalg.norm(dif+x*PERIOD[:,0]+y*PERIOD[:,1])
                           for x in range(-3,4) for y in range(-3,4))
            assert abs(distance-.5) < 1e-8
        assert np.all(degree == 4)
        triangle_pairs = []
        for r in triangles:
            indices = [ids[r[f'site_id_{i}']] for i in range(1,4)]
            triangle_pairs.extend(tuple(sorted((indices[i],indices[j])))
                                  for i,j in ((0,1),(1,2),(0,2)))
        assert len(set(triangle_pairs)) == 54 and set(triangle_pairs) == set(bonds)
        ere = sum(bonds.values())/27
        ee = f(row,'energy_exchange')
        et = f(row,'energy_total')
        field = f(row,'field')
        weights_norm_error = max(abs(sum(f(r,'probability') for r in spectra
                                        if r['a'] == str(a) and r['k'] == str(k))-1)
                                 for a in range(3) for k in range(3))
        check = dict(D=D, sites=27, unique_bonds=54, triangles=18,
                     energy_from_bonds_minus_summary=ere-ee,
                     energy_from_triangles_minus_summary=sum(f(r,'exchange_energy') for r in triangles)/27-ee,
                     total_energy_identity_error=et-(ee-field*sz.mean()),
                     magnetization_identity_error=2*sz.mean()-f(row,'magnetization_normalized'),
                     bond_component_sum_max=max(abs(f(r,'sxx')+f(r,'syy')+f(r,'szz')-f(r,'energy')) for r in bonds_raw),
                     triangle_sum_sz_max=max(abs(f(r,'sum_sz')-sum(sz[ids[r[f'site_id_{i}']]] for i in range(1,4))) for r in triangles),
                     schmidt_probability_norm_max=weights_norm_error,
                     parent_hash_matches=(D == 4 or row['parent_sha256'] == snapshots[D-1]['summary']['state_sha256']))
        assert max(abs(check[k]) for k in ('energy_from_bonds_minus_summary',
                   'energy_from_triangles_minus_summary','total_energy_identity_error',
                   'magnetization_identity_error','bond_component_sum_max','triangle_sum_sz_max',
                   'schmidt_probability_norm_max')) < 1e-10
        assert check['parent_hash_matches']
        audit.append(check)
        sy = symmetry(ps, sz, bonds)
        symmetries.append(dict(D=D, **sy))
        c15, aligned15 = paper_compare(sites,15)
        same, aligned_same = paper_compare(sites,D) if D >= 7 else (None,None)
        comparisons.append(dict(D=D, paper_D15=c15, paper_same_D=same))
        record = dict(D=D, field=field, chi=int(row['chi']), energy_exchange=ee,
                      paper_energy_exchange=paper_energy[D], delta_exchange=ee-paper_energy[D],
                      energy_total=et, m_normalized=f(row,'magnetization_normalized'),
                      distance_to_one_ninth=f(row,'distance_to_one_ninth'),
                      ctm_converged=row['ctm_converged'], ctm_residual=f(row,'ctm_residual'),
                      face_consistency=f(row,'face_consistency_max'),
                      su_proxy_converged=row['su_proxy_converged'], total_tau=f(row,'total_tau'),
                      max_abs_sx=f(row,'max_abs_sx'),
                      mirror_rms_sz=sy['mirror']['best_by_sz']['sz_rms'],
                      mirror_rms_bond=sy['mirror']['minimum_bond_rms'],
                      C3_rms_sz=sy['C3']['best_by_sz']['sz_rms'],
                      C3_rms_bond=sy['C3']['minimum_bond_rms'],
                      C6_rms_sz=sy['C6']['best_by_sz']['sz_rms'],
                      minimum_nontrivial_translation_rms_sz=sy['nontrivial_translation']['best_by_sz']['sz_rms'],
                      paper15_spatial_rms=c15['spatial_rms_sz'],
                      paper15_sorted_rms=c15['sorted_rms_sz'], paper15_correlation=c15['correlation'],
                      sameD_spatial_rms=same['spatial_rms_sz'] if same else '',
                      sameD_sorted_rms=same['sorted_rms_sz'] if same else '',
                      sameD_correlation=same['correlation'] if same else '')
        table.append(record)
        snapshots[D] = dict(summary=row, sites=sites, ps=ps, sz=sz, bonds=bonds,
                            aligned15=aligned15, aligned_same=aligned_same)
        # The physical kagome cell contains nine hexagons, not just the
        # three hexagonal tensor blocks. Search all centres independently.
        lookup = {canonical(p):i for i,p in enumerate(ps)}
        for x in range(3):
            for y in range(3):
                centre = C0+x*A1+y*A2
                indices = [lookup[canonical(centre+offset)] for offset in OFFSETS['A']]
                vals = sz[indices]
                perm = permutation(ps,theta=2,centre=centre)
                cres = rms(sz-sz[perm])
                hexagons.append(dict(D=D, centre_x=x,centre_y=y,C3_rms_sz=cres,
                                     minimum_sz=float(vals.min()), maximum_sz=float(vals.max()),
                                     even_mean_sz=float(vals[::2].mean()),odd_mean_sz=float(vals[1::2].mean()),
                                     all_six_field_aligned=bool(vals.min()>0)))

    # Keep a single D10-to-paper alignment for all D in the labelled trend
    # figure. This avoids independently rearranging individual sites.
    d10 = next(r for r in comparisons if r['D'] == 10)['paper_same_D']['alignment']
    rp,_ = reference(15)
    paper_map = json.loads((ROOT/'references'/'paper_D15_reference.json').read_text())
    labels = np.array([r['paper_label'] for r in paper_map['sites']])
    values = np.loadtxt(ROOT/'references'/'data_fig_S3_mag_vs_D.dat', dtype=str, skiprows=1)
    pv = {int(1/Fraction(r[0])):np.array(r[1:],dtype=float) for r in values}
    label_rows = []
    for D,obs in snapshots.items():
        aligned_labels = transformed_values(rp,labels,obs['ps'], d10['rotation_60deg_units'],
                                             d10['reflection'],d10['translation']).astype(int)
        assert set(aligned_labels) == set(range(1,10))
        obs['aligned_labels'] = aligned_labels
        for label in range(1,10):
            vals = obs['sz'][aligned_labels == label]
            assert len(vals) == 3
            label_rows.append(dict(D=D,paper_label=label,sz_mean=float(vals.mean()),
                                   C3_orbit_spread=float(np.ptp(vals)),
                                   paper_same_D=pv[D][label-1] if D in pv else '',
                                   paper_D15=pv[15][label-1]))

    # A geometry-only diagnostic of the pattern: alternating polarizations
    # on each of the three inequivalent C3-centred physical hexagons.
    # This comparison is invariant under relabelling the three centres.
    c3_hexagons = []
    for name,values in (('this_D10',snapshots[10]['sz']),
                        ('paper_D10',snapshots[10]['aligned_same'])):
        ps = snapshots[10]['ps']
        lookup = {canonical(p):i for i,p in enumerate(ps)}
        candidates = []
        for x in range(3):
            for y in range(3):
                centre = C0+x*A1+y*A2
                perm = permutation(ps,theta=2,centre=centre)
                if rms(values-values[perm]) > 1e-6:
                    continue
                indices = [lookup[canonical(centre+offset)] for offset in OFFSETS['A']]
                v = values[indices]
                candidates.append(dict(source=name,centre_x=x,centre_y=y,
                                       low_sz=float(v.min()),high_sz=float(v.max())))
        assert len(candidates) == 3
        c3_hexagons.extend(sorted(candidates,key=lambda r:r['low_sz']))

    ds = np.array(list(snapshots))
    fig,ax = plt.subplots(1,3,figsize=(16,4.5))
    ax[0].plot(ds,[r['energy_exchange'] for r in table],'o-',label='This D4 to D10 chain')
    ax[0].plot(ds,[r['paper_energy_exchange'] for r in table],'s--',label='Paper Fig. 2b data')
    ax[0].set(xlabel='D',ylabel='Exchange energy per spin')
    ax[0].legend(fontsize=9)
    ax[1].axhline(0,color='gray',lw=.8)
    ax[1].plot(ds,[1e4*r['delta_exchange'] for r in table],'o-')
    ax[1].set(xlabel='D',ylabel='(This - paper) exchange energy x 10^4')
    ax[2].plot(ds,[1e6*r['distance_to_one_ninth'] for r in table],'o-')
    ax[2].axhline(0,color='gray',lw=.8)
    ax[2].set(xlabel='D',ylabel='(m/ms - 1/9) x 10^6')
    for a in ax:a.grid(alpha=.25)
    fig.suptitle('h=0.38, chi=40; one field point, no plateau-width determination')
    fig.tight_layout()
    fig.savefig(OUTPUT/'energy_magnetization.png',dpi=170)
    plt.close(fig)

    fig,axes = plt.subplots(3,3,figsize=(17,14))
    for i,D in enumerate((4,7,10)):
        obs = snapshots[D]
        plot_panel(axes[i,0],obs['ps'],obs['sz'],f'This chain D={D}, h=0.38',obs['bonds'])
        if D >= 7:
            plot_panel(axes[i,1],obs['ps'],obs['aligned_same'],f'Paper D={D}, h=0.4; best spatial alignment')
        else:
            axes[i,1].axis('off')
            axes[i,1].text(.15,.5,'No author local-Sz data at D=4',transform=axes[i,1].transAxes)
        plot_panel(axes[i,2],obs['ps'],obs['aligned15'],'Paper D=15, h=0.4; best spatial alignment')
    fig.suptitle('Local Sz: shared colour/size scale; bond widths shown only for measured states',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(OUTPUT/'pattern_comparison.png',dpi=165)
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(13,5))
    colours = plt.cm.tab10(np.arange(9))
    for label in range(1,10):
        ours = [r for r in label_rows if r['paper_label'] == label]
        axes[0].plot([r['D'] for r in ours],[r['sz_mean'] for r in ours],'o-',
                     color=colours[label-1],label=f'Label {label}')
        axes[0].plot([d for d in ds if d in pv],[pv[d][label-1] for d in ds if d in pv],
                     '--',color=colours[label-1],alpha=.8)
    axes[0].set(xlabel='D',ylabel='Local <Sz>',title='Fixed D10 spatial alignment; dashed = paper')
    axes[0].legend(ncol=3,fontsize=8)
    obs = snapshots[10]
    axes[1].plot(range(1,28),np.sort(obs['sz']),'o-',label='This D10')
    axes[1].plot(range(1,28),np.sort(obs['aligned_same']),'s--',label='Paper D10')
    axes[1].plot(range(1,28),np.sort(obs['aligned15']),':',label='Paper D15')
    axes[1].set(xlabel='Sorted site rank (spatial positions discarded)',ylabel='Local <Sz>',
                title='Values can be close while their spatial pattern differs')
    axes[1].legend()
    for a in axes:a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT/'local_sz_trends.png',dpi=170)
    plt.close(fig)

    write_csv(OUTPUT/'comparison.csv',table)
    write_csv(OUTPUT/'consistency_checks.csv',audit)
    write_csv(OUTPUT/'labelled_local_sz.csv',label_rows)
    write_csv(OUTPUT/'hexagon_patterns.csv',hexagons)
    write_csv(OUTPUT/'c3_hexagon_comparison.csv',c3_hexagons)
    details = dict(input_sha256=fingerprints,paper=PAPER,author_data=DATA,
                   paper_field_for_local_sz=.4,our_field=.38,
                   alignment_method='108 global kagome operations; no independent site permutation',
                   local_sz_geometry='Reconstructed from labelled Fig. S3; sorted metric is independent of mapping',
                   fixed_alignment_for_label_trends=d10,summary=table,
                   symmetries=symmetries,comparisons=comparisons,consistency=audit)
    (OUTPUT/'analysis.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')

    def htable(headers, rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+\
               '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'
               for x in row)+'</tr>' for row in rows)+'</tbody></table>'

    energy_table = htable(['D','本次 Eex/site','论文 Eex/site','本次−论文','m/ms','SU proxy','CTM'],
        [[r['D'],f"{r['energy_exchange']:.9f}",f"{r['paper_energy_exchange']:.9f}",
          f"{r['delta_exchange']:+.3e}",f"{r['m_normalized']:.9f}",r['su_proxy_converged'],r['ctm_converged']]
         for r in table])
    pattern_table = htable(['D','对论文同D：空间Sz RMS','排序Sz RMS','空间相关系数','最佳镜面Sz RMS','最佳镜面bond RMS'],
        [[r['D'],f"{r['sameD_spatial_rms']:.5f}",f"{r['sameD_sorted_rms']:.5f}",
          f"{r['sameD_correlation']:.4f}",f"{r['mirror_rms_sz']:.5f}",f"{r['mirror_rms_bond']:.5f}"]
         for r in table if r['D']>=7])
    d10labels = [r for r in label_rows if r['D']==10]
    local_table = htable(['论文site标签','本次D10 Sz（每类3点平均）','论文D10 Sz','论文D15 Sz'],
        [[r['paper_label'],f"{r['sz_mean']:.6f}",f"{r['paper_same_D']:.6f}",f"{r['paper_D15']:.6f}"]
         for r in d10labels])
    hex_table = htable(['C3中心六边形类型','本次D10较小Sz','本次D10较大Sz','论文D10较小Sz','论文D10较大Sz'],
        [[kind,f"{c3_hexagons[i]['low_sz']:.6f}",f"{c3_hexagons[i]['high_sz']:.6f}",
          f"{c3_hexagons[i+3]['low_sz']:.6f}",f"{c3_hexagons[i+3]['high_sz']:.6f}"]
         for i,kind in enumerate(('负磁化较强','负磁化较弱','全正磁化'))])
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>服务器D4–D10结果与Ghosh–Mila论文比较</title>
<style>body{{max-width:1180px;margin:35px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243443}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:8px;border-bottom:1px solid #cbd5df;text-align:right}}th{{background:#edf2f6}}img{{max-width:100%}}.note{{background:#fff2d8;padding:16px}}a{{color:#175a9d}}</style>
<h1>服务器 D4–D10：能量接近，局域空间图案尚未吻合</h1>
<p>本次 h=0.38、χ=40，D4随机种子1701，D5–10严格补零继承前一个D。C3旋转关系在ansatz中预先施加，没有固定总磁化。每个D均有27个site、54条最近邻bond和18个三角形的数据。这里只读取下载的5份CSV，没有重新演化、运行CTMRG或改动原始数据。</p>
<p><a href="{PAPER}">Ghosh–Mila论文v1</a> · <a href="{DATA}">作者公开数据v3</a>。论文局域磁化数据为h=0.4，本次为h=0.38。两点位于论文报告的平台内部，但尚未对本实现独立验证平台宽度，不能把两磁场的图案差别完全归因于不同态。</p>
<h2>能量与磁化</h2>
<p>统一比较不含磁场的交换能 Eex=Σbond〈Si·Sj〉/27。采用 H=ΣS·S−hΣSz，Etot=Eex−h·mean(Sz)。1/9时 mean(Sz)=1/18，在h=0.38的Zeeman项为−0.0211111111/site。论文Fig.2b的数值按其不含Zeeman的e(m)数据比较；文中去磁场能量的符号书写与−hΣSz的惯例有歧义，这里明确使用公开表中的交换能数据，不再次减磁场。</p>
{energy_table}<img src="energy_magnetization.png">
<p>D8–10的本次Eex比论文同D略低，D10差−2.10474×10⁻⁴/site。D5–7略高。本次D10仍高于论文D15的−0.422039091。所有D的m/ms距离1/9小于9×10⁻⁷；这是单个磁场点的结果，不能确定平台宽度。</p>
<h2>图案与对称性</h2>
<p>用原始空间坐标逐点比较，遍历9个平移、6个旋转和2种反射，共108个全局操作。没有任意交换site。论文27-site标签位置依据Fig.S3重建；排序比较完全不依赖这一步位置重建。图中颜色/圆圈大小为局域Sz，只有本次数据展示实测bond宽度。作者公开数据没有逐条NN bond表，因此无法作逐bond数值匹配。</p>
{pattern_table}<img src="pattern_comparison.png">
<p>D10与论文同D的排序RMS仅0.01193，但最佳空间RMS仍为0.06722，相关系数0.85349。这说明九类磁化数值的集合已较接近，排列仍有明显差别。对论文D15的最佳空间RMS为0.05268、相关系数0.90234，不能将0.9相关系数解释为同一态。D4–5有很好的近似镜面；D6以后明显偏离，D10最佳镜面Sz RMS约0.049。D10的C3逐点误差约10⁻¹¹，与预先施加的约束一致。C6和更小平移周期均未恢复。</p>
<p>论文有限D图案同样保留C3并破坏镜面，但相同的空间对称性不足以保证局域排列相同。一个不依赖整体坐标对齐的差别是三个C3中心六边形上的交替磁化：每个六边形的两个数值各出现三次。按负磁化的大小给六边形分类后，本次最强正磁化0.253与最强负磁化−0.148交替，而论文最强正磁化0.259与较弱负磁化−0.077交替。</p>
{hex_table}<p>下表在一个固定的D10最佳空间对齐下比较，而非把每类磁化分别排序。</p>
{local_table}<img src="local_sz_trends.png">
<h2>收敛与数据一致性</h2>
<p class="note">所有D的CTMRG迭代收敛标志为True，但χ只有40一个值；这不等于χ收敛。所有D的最终SU proxy均为False，因此不能声称虚时演化充分收敛。D6、8、9、10的总τ=544.012，已跑满当前五stage的上限。CSV没有SU历史、每stage报告和final.pt，无法从这些表确定最终漂移大小或继续演化。</p>
<p>54条唯一最近邻bond均使每个site连接4条bond，且每条bond恰好属于一个基本三角形。bond求和、triangle求和、分量相加、局域Sz求和、Zeeman项与summary互相吻合到浮点精度；SU权重归一化和D4→D10父态hash链也通过。这排除了下载表格汇总中的计数错误，但不是对高D张量收缩算法的独立验证。D7–10的不同环境位置观测量差异约8×10⁻⁶至1.5×10⁻⁵，此误差指标并非严格能量误差界。D9残留max|Sx|≈1.06×10⁻⁴，也需在继续演化时检查。</p>
<h2>下一步</h2><p>优先在保存的D10态上增大χ，仅重测观测量；再延长小dt演化、在同χ下复测局域图案是否继续变化。另保留早先D5/D6独立随机初态得到的较像论文图案的分支，避免只追踪一条D4继承链。若要严格对照论文图案，补一个h=0.4的测量/优化点。本次未启动任何大规模本机计算。</p>
<p><a href="comparison.csv">逐D比较表</a> · <a href="labelled_local_sz.csv">固定空间对齐的局域磁化</a> · <a href="hexagon_patterns.csv">全部六边形检查</a> · <a href="consistency_checks.csv">汇总一致性</a> · <a href="analysis.json">完整指标与来源hash</a></p></html>'''
    (OUTPUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(summary=table,checks=audit,output=str(OUTPUT/'report.html')),indent=2))


if __name__ == '__main__':
    main()
