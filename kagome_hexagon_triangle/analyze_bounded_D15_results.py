"""Read-only comparison of the downloaded D5-origin continuation with author data.

No evolution or contraction. All 108 global space operations are allowed;
individual site exchanges are never used for alignment.
"""
import hashlib
import html
import json
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analyze import ROOT, reference
from .analyze_seeded_server_results import load_runs, align
from .analyze_server_results import rms, symmetry, permutation, write_csv
from .plot_local_D5_D6_vs_paper import panel

INPUT = ROOT/'results/server_from_local_D5_D9_D15_h038_C3_bounded'
OUT = INPUT/'analysis_20261112'
PAPER = 'https://arxiv.org/html/2610.34439v1'
DATA = 'https://doi.org/10.6084/m9.figshare.34731234.v3'


def mirror_error(ps, sz):
    return min(rms(sz-sz[permutation(ps, t, True, (x, y))])
               for t in range(6) for x in range(3) for y in range(3))


def main():
    OUT.mkdir(exist_ok=True)
    sources = list(INPUT.glob('*.csv')) + [ROOT/'references'/name for name in (
        'data_fig_2b_energy_vs_D.dat', 'data_fig_S3_mag_vs_D.dat',
        'paper_D15_reference.json')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert hashlib.md5(sources[-3].read_bytes()).hexdigest() == '9e0fb678ea72578b807a6667fa8cc4f8'
    assert hashlib.md5(sources[-2].read_bytes()).hexdigest() == '5c612a9ea4b72b06e6f4738576011acf'
    _, runs = load_runs(INPUT)
    assert sorted(runs) == list(range(5, 16))
    energies = {int(1/Fraction(d)): float(e) for d, e in np.loadtxt(
        ROOT/'references/data_fig_2b_energy_vs_D.dat', dtype=str, skiprows=1)}
    labels = np.array([s['paper_label'] for s in json.loads(
        (ROOT/'references/paper_D15_reference.json').read_text())['sites']])
    available = {7, 8, 9, 10, 11, 12, 14, 15}
    records, classes, details = [], [], []
    snapshots = {}
    for D, run in sorted(runs.items()):
        s = run['summary']
        if D > 9:
            assert s['parent_sha256'] == runs[D-1]['summary']['state_sha256']
        E = float(s['energy_exchange'])
        sy = symmetry(run['ps'], run['sz'], run['energies'])
        refD = D if D in available else 15
        ps, rv = reference(refD)
        a = align(run['ps'], run['sz'], ps, rv)
        _, ref15 = reference(15)
        a15 = align(run['ps'], run['sz'], ps, ref15)
        record = dict(D=D, field=float(s['field']), chi=int(s['chi']),
            exchange_energy=E, paper_energy=energies[D], delta_energy=E-energies[D],
            magnetization_normalized=float(s['magnetization_normalized']),
            SU_proxy=s['su_proxy_converged'], CTM_converged=s['ctm_converged'],
            ctm_residual=float(s['ctm_residual']), face_consistency=float(s['face_consistency_max']),
            added_steps=s.get('added_steps_this_job', ''),
            pattern_reference_D=refD, same_D_pattern_reference=D in available,
            spatial_rms=a['rms'], correlation=a['correlation'],
            sorted_rms=rms(np.sort(run['sz'])-np.sort(rv)),
            spatial_rms_to_D15=a15['rms'],
            C3_sz_rms=sy['C3']['best_by_sz']['sz_rms'],
            mirror_sz_rms=sy['mirror']['best_by_sz']['sz_rms'],
            mirror_bond_rms=sy['mirror']['minimum_bond_rms'],
            translation_sz_rms=sy['nontrivial_translation']['best_by_sz']['sz_rms'],
            paper_mirror_sz_rms=mirror_error(ps, rv))
        records.append(record)
        snapshots[D] = dict(ps=ps, paper=rv, ours=a['values'], record=record)
        details.append(dict(D=D, checks=run['checks'], symmetry=sy,
                            alignment={k:v for k,v in a.items() if k!='values'}))
        for label in range(1, 10):
            vals = a['values'][labels==label]
            classes.append(dict(D=D, reference_D=refD, label=label,
                paper_sz=float(rv[labels==label].mean()), ours_sz=float(vals.mean()),
                ours_class_range=float(np.ptp(vals))))
        print('RESULT', record)
    write_csv(OUT/'comparison.csv', records)
    write_csv(OUT/'local_classes.csv', classes)
    (OUT/'analysis.json').write_text(json.dumps(dict(
        paper=PAPER, author_data=DATA, sources_sha256=hashes, records=records,
        checks_and_symmetries=details,
        limitations=[
            'Our field is 0.38; author magnetization references use 0.4.',
            'Compare field-removed exchange energy to author Fig2b, not total energy.',
            'Author spatial site labels are reconstructed from FigS3; sorted RMS is mapping independent.',
            'No author bond-resolved numeric reference is available in the downloaded data.',
            'All D9-D15 SU proxies are false; one chi only. CTM convergence is not chi convergence.',
            'Existing branch uses five-spin SU, not the later physical-NN kernel.'])
        , indent=2)+'\n', encoding='utf-8')

    ds = np.array([r['D'] for r in records])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    axes[0].plot(ds, [r['exchange_energy'] for r in records], 'o-', label='D5-origin chain, h=0.38')
    axes[0].plot(ds, [r['paper_energy'] for r in records], 's--', label='Author Fig.2b data')
    axes[0].set(xlabel='Bond dimension D', ylabel='Exchange energy per spin', title='Raw finite-D data; no extrapolation')
    axes[0].legend(fontsize=9)
    axes[1].plot(ds, [1e4*r['delta_energy'] for r in records], 'o-')
    axes[1].axhline(0, color='#777777', lw=.8)
    axes[1].set(xlabel='Bond dimension D', ylabel=r'$(E_{ours}-E_{paper})\times 10^4$', title='Negative: our reported energy is lower')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'energy_vs_paper.png', dpi=170)
    plt.close(fig)

    def draw_patterns(chosen, filename):
        fig, axes = plt.subplots(len(chosen), 2, figsize=(12.4, 5.4*len(chosen)), squeeze=False)
        for row, D in enumerate(chosen):
            q=snapshots[D]; r=q['record']
            artist=panel(axes[row,0],q['ps'],q['paper'],f'Paper D={D}, h=0.4',
                         f"Eex={r['paper_energy']:.9f}; author local-Sz values")
            panel(axes[row,1],q['ps'],q['ours'],f'Our D={D}, h=0.38; global alignment',
                  f"Eex={r['exchange_energy']:.9f}; Sz RMS={r['spatial_rms']:.4f}\n"
                  f"corr={r['correlation']:.3f}; mirror RMS={r['mirror_sz_rms']:.2g}")
        fig.subplots_adjust(left=.015,right=.91,bottom=.10/len(chosen),top=1-.12/len(chosen),hspace=.42,wspace=.01)
        fig.colorbar(artist,cax=fig.add_axes([.945,.23,.014,.5]),label=r'Local $\langle S^z\rangle$')
        fig.text(.46,.015/len(chosen),'Shared colour/size scales. Only global lattice operations; bonds show geometry, not measured strength.',
                 ha='center',fontsize=9)
        fig.savefig(OUT/filename,dpi=175,bbox_inches='tight',pad_inches=.15)
        plt.close(fig)
    draw_patterns([15], 'D15_vs_paper.png')
    draw_patterns([9,12,15], 'D9_D12_D15_vs_paper.png')

    selected=[r for r in records if r['same_D_pattern_reference']]
    fig, axes = plt.subplots(1,2,figsize=(11.6,4.6))
    axes[0].semilogy(ds,[r['mirror_sz_rms'] for r in records],'o-',label='Our chain')
    axes[0].semilogy([r['D'] for r in selected],[r['paper_mirror_sz_rms'] for r in selected],'s--',label='Paper')
    axes[0].set(xlabel='D',ylabel='Minimum reflection RMS of local Sz',title='Reflection symmetry: smaller means closer')
    axes[0].legend()
    c15=[r for r in classes if r['D']==15]
    axes[1].plot(range(1,10),[r['paper_sz'] for r in c15],'s--',label='Paper D15')
    axes[1].plot(range(1,10),[r['ours_sz'] for r in c15],'o-',label='Our D15, spatially aligned')
    axes[1].set(xlabel='Paper Fig.S3 site class',ylabel=r'$\langle S^z\rangle$',title='Three repeated sites per class')
    axes[1].set_xticks(range(1,10));axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'symmetry_and_classes.png',dpi=170);plt.close(fig)

    head=['D','本次交换能','论文数据','能量差','m/ms','SU','同D磁化RMS','相关系数','镜面RMS']
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
        r['D'],f"{r['exchange_energy']:.10f}",f"{r['paper_energy']:.10f}",f"{r['delta_energy']:+.3e}",
        f"{r['magnetization_normalized']:.9f}",r['SU_proxy'],
        f"{r['spatial_rms']:.5f}" if r['same_D_pattern_reference'] else '无同D参考',
        f"{r['correlation']:.4f}" if r['same_D_pattern_reference'] else '—',
        f"{r['mirror_sz_rms']:.3e}"))+'</tr>' for r in records)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D5来源链至D15：论文对照</title>
<style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#243443}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:7px;border-bottom:1px solid #ccd4dc;text-align:right}}.wide{{overflow:auto}}</style>
<h1>D5来源链，D=9–15：能量接近，有限D图案不同</h1>
<p>这次下载已包含D14和D15的完整观测量。每个新D继承上一个D的保存态。没有重新运行演化或CTMRG。</p>
<p>能量比较使用去掉Zeeman项的每自旋交换能；与作者Fig2b原始数据点逐D比较。我们h=0.38，作者磁化图案h=0.4；两者平均磁化均接近1/9，但此处不是严格同磁场实验。</p>
<img src="energy_vs_paper.png"><div class="wide"><table><tr>{''.join('<th>'+x+'</th>' for x in head)}</tr>{body}</table></div>
<h2>逐site空间比较</h2><p>枚举108种全局平移、旋转、反射，固定论文图案的几何坐标。没有交换单个site。D15的最佳空间RMS为0.07114，排序后的RMS仍为0.03432，说明差异也存在于磁化数值集合中，不能只用位置对调解释。</p>
<img src="D15_vs_paper.png"><img src="D9_D12_D15_vs_paper.png">
<h2>更大的D趋近镜面对称</h2><p>D15的Sz镜面RMS为1.94e-5，bond镜面最小RMS为3.64e-5；C3的Sz RMS为1.23e-13（C3是此任务施加的约束）。论文D15局部磁化镜面RMS约0.02270。我们的9类磁化出现三组成对近简并，趋近6种值。</p>
<p>论文正文也讨论了D更大时可能恢复镜面、9类减少为6类的可能性。这与我们的趋势相容，但不是同一有限D图案已被复现的证据。</p>
<img src="symmetry_and_classes.png">
<h2>核对与限制</h2><p>各D的27sites、54条唯一NN bond、18个三角形覆盖及能量/磁化求和通过一致性检查，误差小于1e-10。所有CTMRG标记收敛，但D9–15的SU代理判据均未通过；没有chi扫描。D15不同face测量间最大差8.52e-5是内部一致性指标，不是能量误差条。现有数据不能宣称能量已可靠优于论文。</p>
<p>本分支仍用旧五自旋SU演化，不是后来新增的physical-NN逐bond更新。采用论文的分块ansatz不等于完整复现作者的优化流程。作者逐bond数值未包含在本地参考数据中，故本报告不声称bond图案与论文定量一致。</p>
<p><a href="{PAPER}">论文正文</a> · <a href="{DATA}">作者数据</a> · <a href="comparison.csv">完整指标</a> · <a href="local_classes.csv">各类磁化</a> · <a href="analysis.json">检查和SHA256</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    assert all(hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    print('SOURCE FILES UNCHANGED; all geometry and observable sum checks passed.')
    print(OUT/'report.html')


if __name__=='__main__': main()
