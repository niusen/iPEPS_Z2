"""Read-only comparison of downloaded independent D7 starts, no TN calculation.

Only global space-group operations are allowed. The paper coordinate frame
is fixed; author Fig.S3 local-Sz data are compared at the SAME D=7 and h=0.4.
Uniform plotting bonds do not represent author bond-energy measurements.
"""
import hashlib
import html
import json
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analyze import ROOT, PERIOD, canonical, reference, site_position, transformed_values
from .analyze_server_results import read_csv, write_csv, rms, symmetry, permutation
from .analyze_seeded_server_results import align
from .plot_local_D5_D6_vs_paper import panel
from .geometry import C0,A1,A2,OFFSETS

INPUT = ROOT/'results/server_D7_random10_h040_C3_screen9000'
OUTPUT = INPUT/'analysis_20261105'
PAPER = 'https://arxiv.org/html/2610.34439v1'
DATA = 'https://doi.org/10.6084/m9.figshare.34731234.v3'


def load():
    names = ('summary','site_magnetization','bond_energies','triangle_observables','schmidt_spectra')
    arrays = {n:read_csv(INPUT/f'{n}.csv') for n in names}
    hashes = {n:hashlib.sha256((INPUT/f'{n}.csv').read_bytes()).hexdigest() for n in names}
    runs = {}
    for summary in arrays['summary']:
        seed = int(summary['seed'])
        assert seed not in runs
        assert int(summary['D'])==7 and float(summary['field'])==.4 and int(summary['chi'])==40
        subset = lambda name:[r for r in arrays[name] if int(r['seed'])==seed]
        sites,bonds,triangles,spectra = (subset(n) for n in names[1:])
        ps=np.array([[float(r['cartesian_x']),float(r['cartesian_y'])] for r in sites])
        sz=np.array([float(r['sz']) for r in sites])
        assert all(canonical(p)==canonical(site_position(r)) for p,r in zip(ps,sites))
        lookup={r['site_id']:i for i,r in enumerate(sites)}
        energies={tuple(sorted((lookup[r['site_id_1']],lookup[r['site_id_2']]))):float(r['energy']) for r in bonds}
        assert len(sites)==len(lookup)==27 and len(bonds)==len(energies)==54 and len(triangles)==18
        degree=np.zeros(27,int)
        for i,j in energies:
            distance=min(np.linalg.norm(ps[i]-ps[j]+x*PERIOD[:,0]+y*PERIOD[:,1])
                         for x in range(-2,3) for y in range(-2,3))
            assert abs(distance-.5)<1e-8
            degree[i]+=1;degree[j]+=1
        assert np.all(degree==4)
        triangle_pairs=[]
        for row in triangles:
            ids=[lookup[row[f'site_id_{i}']] for i in (1,2,3)]
            triangle_pairs.extend(tuple(sorted((ids[i],ids[j]))) for i,j in ((0,1),(1,2),(0,2)))
        assert len(set(triangle_pairs))==54 and set(triangle_pairs)==set(energies)
        checks=dict(seed=seed,bond_sum_error=sum(energies.values())/27-float(summary['energy_exchange']),
            triangle_sum_error=sum(float(r['exchange_energy']) for r in triangles)/27-float(summary['energy_exchange']),
            magnetization_error=2*sz.mean()-float(summary['magnetization_normalized']),
            field_energy_error=-.4*sz.mean()-float(summary['energy_field']),
            total_energy_error=float(summary['energy_exchange'])+float(summary['energy_field'])-float(summary['energy_total']),
            bond_components_error=max(abs(sum(float(r[k]) for k in ('sxx','syy','szz'))-float(r['energy'])) for r in bonds),
            schmidt_normalization_error=max(abs(sum(float(r['probability']) for r in spectra
                if int(r['a'])==a and int(r['k'])==k)-1) for a in range(3) for k in range(3)))
        assert max(abs(v) for k,v in checks.items() if k!='seed')<1e-10
        runs[seed]=dict(summary=summary,sites=sites,ps=ps,sz=sz,energies=energies,checks=checks)
    assert sorted(runs)==list(range(7101,7111))
    return arrays,hashes,runs


def source_indices(ps,dest,alignment):
    p=transformed_values(ps,np.arange(27),dest,alignment['theta'],alignment['reflection'],alignment['shift'])
    assert p is not None and len(set(p))==27
    return p.astype(int)


def reordered_bonds(run,indices):
    inverse=np.argsort(indices)
    return {tuple(sorted((int(inverse[i]),int(inverse[j])))):v for (i,j),v in run['energies'].items()}


def hexagon_values(ps,values,source):
    lookup={canonical(p):i for i,p in enumerate(ps)}
    rows=[]
    for x in range(3):
        for y in range(3):
            centre=C0+x*A1+y*A2
            perm=permutation(ps,theta=2,centre=centre)
            vals=np.array([values[lookup[canonical(centre+p)]] for p in OFFSETS['A']])
            rows.append(dict(source=source,centre_x=x,centre_y=y,
                C3_residual=rms(values-values[perm]),low_sz=float(vals.min()),high_sz=float(vals.max()),
                values=';'.join(map(str,vals))))
    return rows


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    _,hashes,runs=load()
    ps,sz_paper=reference(7)
    labels=np.array([r['paper_label'] for r in json.loads((ROOT/'references/paper_D15_reference.json').read_text())['sites']])
    paper_energy={int(1/Fraction(d)):float(e) for d,e in np.loadtxt(
        ROOT/'references/data_fig_2b_energy_vs_D.dat',dtype=str,skiprows=1)}[7]
    summaries=[];site_rows=[];hex_rows=[];details=[]
    for seed,run in sorted(runs.items()):
        s=run['summary'];a=align(run['ps'],run['sz'],ps,sz_paper)
        idx=source_indices(run['ps'],ps,a)
        run.update(aligned=a['values'],alignment=a,indices=idx,aligned_bonds=reordered_bonds(run,idx))
        sy=symmetry(run['ps'],run['sz'],run['energies'])
        entry=dict(seed=seed,energy_exchange=float(s['energy_exchange']),energy_total=float(s['energy_total']),
            paper_D7_energy=paper_energy,delta_exchange_paper=float(s['energy_exchange'])-paper_energy,
            m_normalized=float(s['magnetization_normalized']),max_abs_sx=float(s['max_abs_sx']),
            steps=int(s['steps']),imported_steps=int(s['imported_steps']),added_steps=int(s['screening_added_steps']),
            SU_proxy=s['su_proxy_converged'],CTM_converged=s['ctm_converged'],CTM_residual=float(s['ctm_residual']),
            spatial_rms=a['rms'],sorted_rms=rms(np.sort(run['sz'])-np.sort(sz_paper)),correlation=a['correlation'],
            C3_site_rms=sy['C3']['best_by_sz']['sz_rms'],mirror_site_rms=sy['mirror']['best_by_sz']['sz_rms'],
            mirror_bond_rms_at_best_site_mirror=sy['mirror']['best_by_sz']['bond_rms'],
            mirror_bond_rms_best=sy['mirror']['minimum_bond_rms'])
        summaries.append(entry)
        for label in range(1,10):
            v=run['aligned'][labels==label];ref=sz_paper[labels==label]
            site_rows.append(dict(seed=seed,paper_label=label,our_sz=float(v.mean()),paper_sz=float(ref.mean()),
                                  difference=float((v-ref).mean()),our_class_spread=float(np.ptp(v))))
        hex_rows.extend(hexagon_values(ps,run['aligned'],f'seed{seed}'))
        details.append(dict(seed=seed,alignment={k:v for k,v in a.items() if k!='values'},symmetry=sy))
    hex_rows.extend(hexagon_values(ps,sz_paper,'paper_D7'))
    pairs=[]
    seeds=sorted(runs)
    for i,seed in enumerate(seeds):
        for other in seeds[i+1:]:
            a,b=runs[seed],runs[other]
            alignment=align(a['ps'],a['sz'],b['ps'],b['sz'])
            indices=source_indices(a['ps'],b['ps'],alignment)
            bonds=reordered_bonds(a,indices)
            bd=rms([v-b['energies'][k] for k,v in bonds.items()])
            pairs.append(dict(seed1=seed,seed2=other,spatial_rms_sz=alignment['rms'],
                sorted_rms_sz=rms(np.sort(a['sz'])-np.sort(b['sz'])),bond_rms_at_best_sz_alignment=bd,
                correlation=alignment['correlation'],rotation_60deg=alignment['theta'],
                reflection=alignment['reflection'],translation_x=alignment['shift'][0],translation_y=alignment['shift'][1]))
    result=dict(paper=PAPER,author_data=DATA,paper_D=7,paper_h=.4,paper_exchange_energy=paper_energy,
        source_sha256=hashes,summaries=summaries,pairwise=pairs,alignments=details,
        checks=[r['checks'] for r in runs.values()],
        method='All 108 global operations only; paper fixed; sorted RMS is unrestricted-permutation lower bound.',
        caveat='Only CSVs downloaded, no states/logs. 7101/7102 imported long runs; others 9000-step screening. SU proxy is not a variational optimality certificate.')
    exchanged=sz_paper.copy()
    exchanged[labels==8]=sz_paper[labels==9].mean()
    exchanged[labels==9]=sz_paper[labels==8].mean()
    result['diagnostic_only_not_allowed_alignment']={str(seed):dict(
        rms_after_artificial_paper_8_9_exchange=rms(runs[seed]['aligned']-exchanged),
        original_rms=runs[seed]['alignment']['rms']) for seed in (7104,7109)}
    write_csv(OUTPUT/'comparison.csv',summaries)
    write_csv(OUTPUT/'site_class_comparison.csv',site_rows)
    write_csv(OUTPUT/'pairwise_comparison.csv',pairs)
    write_csv(OUTPUT/'hexagon_magnetizations.csv',hex_rows)
    (OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PAPER Eex',paper_energy,'Sz',','.join(f'{sz_paper[labels==i].mean():.6f}' for i in range(1,10)))
    for r in sorted(summaries,key=lambda r:r['spatial_rms']):
        vals=[x['our_sz'] for x in site_rows if x['seed']==r['seed']]
        print(f"{r['seed']}: E={r['energy_exchange']:.9f} m={r['m_normalized']:.7f} "
              f"spatial={r['spatial_rms']:.6f} sorted={r['sorted_rms']:.6f} corr={r['correlation']:.5f} "
              f"SU={r['SU_proxy']} CTM={r['CTM_converged']} Sz="+','.join(f'{v:.5f}' for v in vals))
    print('CLOSEST PAIRS',json.dumps(sorted(pairs,key=lambda r:r['spatial_rms_sz'])[:6]))
    # Plot/report generation is isolated below; source tables are read only.
    make_figures_and_report(runs,summaries,pairs,site_rows,ps,sz_paper,labels,paper_energy)
    assert hashes=={n:hashlib.sha256((INPUT/f'{n}.csv').read_bytes()).hexdigest() for n in hashes}


def make_figures_and_report(runs,summaries,pairs,site_rows,ps,sz_paper,labels,paper_energy):
    by_seed={r['seed']:r for r in summaries}
    def draw(ax,seed=None,compact=False):
        if seed is None:
            vals=sz_paper;title='Paper D=7, h=0.4 (fixed frame)'
            sub=f'Eex = {paper_energy:.9f}'
        else:
            r=by_seed[seed];vals=runs[seed]['aligned']
            title=f"Seed {seed}: SU {r['SU_proxy']}, CTM {r['CTM_converged']}"
            sub=f"Eex={r['energy_exchange']:.9f}; m/ms={r['m_normalized']:.7f}"
        artist=panel(ax,ps,vals,title,sub)
        coeff=np.linalg.solve(PERIOD,ps.T).T%1
        coeff[np.isclose(coeff,1,atol=1e-9)]=0
        shown=coeff@PERIOD.T
        for p,label in zip(shown,labels):
            if label in (8,9):
                ax.annotate(str(label),p,xytext=(9,-10),textcoords='offset points',
                    fontsize=8 if compact else 10,fontweight='bold',color='#713c89',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=.4),zorder=6)
        return artist

    fig,axs=plt.subplots(2,2,figsize=(12.6,10.8))
    for ax,seed in zip(axs.flat,(None,7104,7109,7102)):
        artist=draw(ax,seed)
    fig.subplots_adjust(left=.015,right=.92,top=.91,bottom=.085,wspace=.025,hspace=.24)
    fig.colorbar(artist,cax=fig.add_axes([.945,.24,.013,.49]),label=r'Local $\langle S^z\rangle$')
    fig.suptitle('Independent D7 starts: similar values can occupy different sites',fontsize=15,y=.98)
    fig.text(.46,.025,'Paper frame fixed; only measured states are globally translated / rotated / reflected.\n'
        'Purple 8/9 mark paper site classes. Bonds show geometry only. SU=False means screening is incomplete.',
        ha='center',fontsize=9.5,linespacing=1.5)
    fig.savefig(OUTPUT/'key_patterns_vs_paper_D7.png',dpi=175,bbox_inches='tight')
    fig.savefig(OUTPUT/'key_patterns_vs_paper_D7.svg',bbox_inches='tight')
    plt.close(fig)

    fig,axs=plt.subplots(4,3,figsize=(18,19.4))
    for ax,seed in zip(axs.flat,(None,*sorted(runs))):
        artist=draw(ax,seed,True)
    axs.flat[-1].axis('off')
    axs.flat[-1].text(.02,.75,'D=7, h=0.4, chi=40\n\n7101/7102: imported long runs\n7103-7110: 9000-step screening\n\nCTM failed: 7105, 7107\nSU proxy passed: 7102 only\n\nAll panels use the same paper frame\nand the same marker/colour scale.',
        transform=axs.flat[-1].transAxes,fontsize=12,va='top',linespacing=1.6)
    fig.subplots_adjust(left=.015,right=.94,top=.925,bottom=.045,wspace=.02,hspace=.26)
    fig.colorbar(artist,cax=fig.add_axes([.96,.3,.009,.4]),label=r'Local $\langle S^z\rangle$')
    fig.suptitle('All 10 independent D7 starts versus the SAME-D paper reference',fontsize=18,y=.99)
    fig.text(.47,.012,'Every state is aligned using one of 108 GLOBAL space-group operations only. No arbitrary site exchange is allowed.',
             ha='center',fontsize=12)
    fig.savefig(OUTPUT/'all_10_patterns_vs_paper_D7.png',dpi=155,bbox_inches='tight')
    plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(16.8,4.8))
    ranks=np.arange(1,28)
    for seed in (7104,7109,7102):
        r=by_seed[seed]
        vals=[v['our_sz'] for v in site_rows if v['seed']==seed]
        axs[0].plot(range(1,10),vals,'o-',label=f'Seed {seed}')
        axs[1].plot(ranks,np.sort(runs[seed]['sz'])[::-1],'o-',ms=3,label=f'Seed {seed}')
    paper_class=[float(sz_paper[labels==i].mean()) for i in range(1,10)]
    axs[0].plot(range(1,10),paper_class,'ks--',label='Paper D7')
    axs[1].plot(ranks,np.sort(sz_paper)[::-1],'ks--',ms=3,label='Paper D7')
    for a in axs[:2]:
        a.axhline(0,color='#aaaaaa',lw=.6);a.legend(fontsize=9);a.grid(alpha=.2)
        a.set_ylabel(r'$\langle S^z\rangle$')
    axs[0].axvspan(7.65,9.3,color='#e4d7ed',alpha=.5)
    axs[0].set(xlabel='Paper site class (fixed spatial positions)',xticks=range(1,10),title='Spatial assignment: classes 8 / 9 differ')
    axs[1].set(xlabel='Site rank, largest to smallest (27 sites)',title='Sorting removes ALL position information')
    ss=sorted(by_seed)
    matrix=np.zeros((10,10))
    for p in pairs:
        i,j=ss.index(p['seed1']),ss.index(p['seed2'])
        matrix[i,j]=matrix[j,i]=p['spatial_rms_sz']
    im=axs[2].imshow(matrix,cmap='magma_r',vmin=0,vmax=.16)
    axs[2].set(xticks=range(10),yticks=range(10),xticklabels=[str(s)[-2:] for s in ss],
        yticklabels=[str(s)[-2:] for s in ss],xlabel='Seed suffix (71xx)',ylabel='Seed suffix (71xx)',
        title='Pairwise best spatial RMS after alignment')
    fig.colorbar(im,ax=axs[2],shrink=.8)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUTPUT/'spatial_sorted_and_pairwise.png',dpi=175,bbox_inches='tight')
    plt.close(fig)

    # Full read-only report, with links to raw records and exact metrics.
    rows=[]
    for r in sorted(summaries,key=lambda r:r['energy_total']):
        rows.append(f"<tr><td>{r['seed']}</td><td>{r['energy_exchange']:.9f}</td><td>{r['energy_total']:.9f}</td>"
            f"<td>{r['m_normalized']:.7f}</td><td>{r['spatial_rms']:.6f}</td><td>{r['sorted_rms']:.6f}</td>"
            f"<td>{r['SU_proxy']}</td><td>{r['CTM_converged']}</td><td>{r['steps']:,}</td></tr>")
    localrows=[]
    for label in range(1,10):
        vals={s:next(v['our_sz'] for v in site_rows if v['seed']==s and v['paper_label']==label) for s in (7104,7109,7102)}
        localrows.append(f"<tr><td>{label}</td><td>{paper_class[label-1]:+.6f}</td>"+
            ''.join(f'<td>{vals[s]:+.6f}</td>' for s in (7104,7109,7102))+'</tr>')
    negative_7104={label:next(v['our_sz'] for v in site_rows if v['seed']==7104 and v['paper_label']==label)
                   for label in (8,9)}
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D7 随机初态图案比较</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1160px;margin:36px auto;padding:0 24px;color:#223044;background:#fafbfc}}
h1,h2{{line-height:1.35}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px;background:white}}th,td{{padding:8px;border-bottom:1px solid #dbe1e8;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;background:white}}a{{color:#205d96}}.note{{padding:16px 20px;background:#edf3fa;border-left:4px solid #527eab}}.scroll{{overflow:auto}}</style>
<h1>D=7 的十次随机初始化：当前图案没有统一，也尚未完整对上论文</h1>
<p>数据：h=0.4，χ=40，27-site C3 六边形＋三角形 ansatz。与 Ghosh–Mila 论文同 D=7、同 h=0.4 的作者公开数值比较；论文空间标签由 Fig. S3 重建。分析日期：2026-11-05。</p>
<div class="note">7104 是当前最低能量结果；7109 的磁化图案空间 RMS 最小。两者的负磁化 8、9 类与论文近似互换，不能用整体平移、旋转、反射消除。7102 虽通过 SU 收敛代理判据并得到 1/9 磁化，图案却明显不同。</div>
<h2>关键图案，论文方向始终固定</h2><img src="key_patterns_vs_paper_D7.png">
<p>统一颜色、大小及数值均表示局部 ⟨Sz⟩，不是归一化总磁化。灰色 bond 仅表示几何；作者没有提供逐 bond 数值，因此不能声称 bond 观测量也逐项复现。</p>
<h2>磁化数值接近，但空间位置不同</h2><img src="spatial_sorted_and_pairwise.png">
<p>7104 对论文的空间 RMS 为 0.037714；若完全丢弃空间信息并排序，RMS 仅为 0.002185。九类局部磁化如下，每类对应三个 site：</p>
<table><tr><th>论文标签</th><th>论文 D7</th><th>7104</th><th>7109</th><th>7102</th></tr>{''.join(localrows)}</table>
<p>7104 的第 8、9 类分别为 {negative_7104[8]:+.6f}、{negative_7104[9]:+.6f}，而论文为 {paper_class[7]:+.6f}、{paper_class[8]:+.6f}，强弱近似互换。只作诊断地人为交换论文 8/9 后残差下降，但这不是允许的空间对称操作，也没有生成任何“修正后的态”。</p>
<h2>各次运行的完整比较</h2><div class="scroll"><table><tr><th>seed</th><th>Eex/site</th><th>Etotal/site</th><th>m/ms</th><th>空间 RMS</th><th>排序 RMS</th><th>SU 代理收敛</th><th>CTMRG 收敛</th><th>SU 步数</th></tr>{''.join(rows)}</table></div>
<p>作者 D7 交换能为 {paper_energy:.12f}。同场比较态的能量应优先看含场的 Etotal；7104 的 Etotal 比用作者局部磁化重建的含场能高约 1.88×10⁻⁵/site，Eex 则高约 1.05×10⁻⁴/site。m 尚有偏差，不能仅用 Eex 的差异评判总能量。</p>
<h2>不同随机初态之间</h2><p>7104 与 7109 经整体反射等对齐后，Sz RMS=0.008517，与完全排序后的 RMS 相同；两者属于相近的 site 强弱排列，但磁化数值仍有差别，同一对齐下 bond RMS=0.012302。不能把它们当成已经数值相同的态。</p>
<p>7102 的近似镜面对称在 site 与 bond 上均成立（残差分别约 4.65×10⁻⁶ 与 6.28×10⁻⁶）；与论文的空间磁化 RMS=0.106429，明显不同。其余 9000 步结果也没有形成统一图案，但大多尚未通过 SU 收敛判据，不能据此确定最终稳定态数目。</p>
<h2>收敛与数据范围</h2><p>10 个 seed 均有完整的 27 site、54 NN bond、18 triangle 观测量。7101、7102 导入旧任务的 329600、374001 步状态，本轮未追加 SU；其余八个 seed 是 9000 步筛选。只有 7102 的 SU 代理收敛标记为真；7105、7107 的 CTMRG 在 50 轮内未收敛。当前只有 CSV，没有下载 wavefunction/checkpoint 或完整日志，因此这里只分析已有数据，不运行额外 SU 或 CTMRG。</p>
<p>7104/7109 最大 |Sx| 仍约 0.0210/0.0237，进一步说明它们尚不能等同于充分收敛的平台态。优先保留这两份服务器 checkpoint 作为后续候选；若要验证能否到达论文图案，需继续演化并再次完整测量。</p>
<h2>全部十次图案</h2><a href="all_10_patterns_vs_paper_D7.png"><img src="all_10_patterns_vs_paper_D7.png"></a>
<h2>数值与来源</h2><p><a href="comparison.csv">每次比较表</a> · <a href="site_class_comparison.csv">九类 site 数值</a> · <a href="pairwise_comparison.csv">45 对初态间比较</a> · <a href="analysis.json">方法、对齐操作、校验和完整指标</a></p>
<p>原始数据：<a href="../summary.csv">summary.csv</a>、<a href="../site_magnetization.csv">site_magnetization.csv</a>、<a href="../bond_energies.csv">bond_energies.csv</a>、<a href="../triangle_observables.csv">triangle_observables.csv</a>、<a href="../schmidt_spectra.csv">schmidt_spectra.csv</a>。全部只读；已核对 bond 求和、三角形求和、Zeeman 计数、site 磁化、Schmidt 归一化及原始文件 SHA256 未变化。</p>
<p>论文：<a href="{PAPER}">arXiv:2610.34439v1</a>；作者数据：<a href="{DATA}">Figshare v3</a>。匹配穷举 6 种旋转×2 种反射×9 种平移；参考方向固定，只变换我们的数据，不允许任意 site 交换。高相关系数不等于空间图案相同。</p></html>'''
    (OUTPUT/'report.html').write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
