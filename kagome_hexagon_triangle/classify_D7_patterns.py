"""Provisional morphology families, not a count of converged thermodynamic phases."""
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,fcluster,dendrogram
from scipy.spatial.distance import squareform
from .analyze import reference,ROOT
from .analyze_seeded_server_results import align
from .analyze_random_D7_results import load,INPUT
from .analyze_server_results import write_csv
from .plot_local_D5_D6_vs_paper import panel

OUT=INPUT/'classification_20261106'


def main():
    OUT.mkdir(exist_ok=True)
    _,hashes,runs=load();seeds=sorted(runs);ps,reference_sz=reference(7)
    matrix=np.zeros((len(seeds),len(seeds)))
    for i,s in enumerate(seeds):
        a=align(runs[s]['ps'],runs[s]['sz'],ps,reference_sz);runs[s]['aligned']=a['values']
        for j in range(i):
            other=seeds[j]
            matrix[i,j]=matrix[j,i]=align(runs[s]['ps'],runs[s]['sz'],runs[other]['ps'],runs[other]['sz'])['rms']
    complete=linkage(squareform(matrix),method='complete')
    average=linkage(squareform(matrix),method='average')
    rows=[]
    groups={'A':[7104,7108,7109],'B':[7102,7105,7106,7107,7110],'C':[7101],'borderline':[7103]}
    for group,members in groups.items():
        for seed in members:
            s=runs[seed]['summary']
            rows.append(dict(seed=seed,provisional_family=group,energy_exchange=float(s['energy_exchange']),
                m_normalized=float(s['magnetization_normalized']),SU_proxy=s['su_proxy_converged'],
                CTM_converged=s['ctm_converged'],note='Borderline: joins A with average linkage; separate or C at coarser complete-linkage cut' if group=='borderline' else 'Morphology only, not an independent phase'))
    sweeps=[]
    for method,z in (('complete',complete),('average',average)):
        for cutoff in (.035,.05,.06,.075):
            labels=fcluster(z,cutoff,criterion='distance')
            parts=[sorted(s for s,l in zip(seeds,labels) if l==label) for label in sorted(set(labels))]
            sweeps.append(dict(method=method,cutoff_sz_rms=cutoff,n_groups=len(parts),members=parts))
    fig,axs=plt.subplots(2,2,figsize=(12.5,10.6))
    for ax,(seed,title) in zip(axs.flat,((7104,'Family A: 7104 / 7108 / 7109'),
            (7102,'Family B: 7102 / 7105 / 7106 / 7107 / 7110'),(7101,'Distinct high-energy candidate: 7101'),
            (7103,'Borderline candidate: 7103'))):
        s=runs[seed]['summary']
        artist=panel(ax,ps,runs[seed]['aligned'],title,
            f"Shown seed {seed}; Eex={float(s['energy_exchange']):.9f}; SU={s['su_proxy_converged']}")
    fig.subplots_adjust(left=.015,right=.92,top=.91,bottom=.08,hspace=.25,wspace=.03)
    fig.colorbar(artist,cax=fig.add_axes([.945,.23,.012,.5]),label=r'Local $\langle S^z\rangle$')
    fig.suptitle('D7 screening: two clear families plus distinct / borderline candidates',fontsize=15,y=.98)
    fig.text(.46,.02,'All patterns globally aligned to the same paper frame. Similarity is provisional: only 7102 passed the SU proxy.\n'
             'Uniform bonds show geometry; no thermodynamic phase count is inferred.',ha='center',fontsize=10)
    fig.savefig(OUT/'family_representatives.png',dpi=165,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5.6))
    order=[7104,7108,7109,7102,7105,7106,7107,7110,7101,7103]
    ix=[seeds.index(s) for s in order]
    im=axs[0].imshow(matrix[np.ix_(ix,ix)],cmap='magma_r',vmin=0,vmax=.2)
    axs[0].set(xticks=range(10),yticks=range(10),xticklabels=[str(s)[-2:] for s in order],
        yticklabels=[str(s)[-2:] for s in order],title='Minimum Sz RMS over 108 global operations',xlabel='Seed 71xx',ylabel='Seed 71xx')
    for p in (2.5,7.5,8.5):axs[0].axhline(p,color='white',lw=1.2);axs[0].axvline(p,color='white',lw=1.2)
    fig.colorbar(im,ax=axs[0],shrink=.8)
    dendrogram(complete,labels=[str(s) for s in seeds],ax=axs[1],color_threshold=.06)
    axs[1].set(title='Complete linkage: count depends on tolerance',ylabel='Largest pairwise Sz RMS in a merged group')
    axs[1].axhline(.06,color='gray',ls='--',lw=.8);axs[1].tick_params(axis='x',labelsize=9)
    fig.tight_layout();fig.savefig(OUT/'family_distances.png',dpi=165,bbox_inches='tight');plt.close(fig)
    write_csv(OUT/'family_assignments.csv',rows)
    write_csv(OUT/'pairwise_sz_rms.csv',[dict(seed=s,**{str(t):matrix[i,j] for j,t in enumerate(seeds)}) for i,s in enumerate(seeds)])
    result=dict(groups=groups,interpretation='About 3-4 coarse morphology types; two robust multi-seed families. Not a phase count.',
        borderline_seed=7103,threshold_sensitivity=sweeps,source_sha256=hashes,
        method='27 sites; same D7 h0.4 chi40; global 6 rotations x 2 reflections x 9 translations; complete and average linkage')
    (OUT/'classification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    table=''.join(f"<tr><td>{r['seed']}</td><td>{r['provisional_family']}</td><td>{r['energy_exchange']:.9f}</td>"
        f"<td>{r['m_normalized']:.7f}</td><td>{r['SU_proxy']}</td><td>{r['CTM_converged']}</td></tr>" for r in sorted(rows,key=lambda r:r['seed']))
    page=f'''<!doctype html><meta charset="utf-8"><title>D7 图案暂定分类</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#26364a}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}a{{color:#205e98}}</style>
<h1>D=7 随机初态图案：约 3–4 类，不能计作 3–4 种稳定相</h1>
<p>最清楚的两族是 A：7104、7108、7109；B：7102、7105、7106、7107、7110。7101 是明显不同的高能候选，7103 在 A 与其它候选之间，暂列边缘态。</p>
<p>A 中 7104/7109 对齐后的 Sz RMS=0.00852；7108 与 7104 为0.02413。B 内最大两两 RMS 约0.05275，而 A/B 之间最小约0.0819，呈现较清楚的两大簇。7103 与 A 中7109的距离仅0.044，故其独立分类不稳健。</p>
<img src="family_representatives.png"><img src="family_distances.png">
<p>完整链接在 RMS=0.06 时给出4组，在0.075时给出3组；平均链接在0.06时给出3组并把7103归入A。因此“3–4类”是粗略形态分组，不是从已收敛态确定的相数。</p>
<table><tr><th>seed</th><th>暂定族</th><th>Eex/site</th><th>m/ms</th><th>SU判据</th><th>CTM收敛</th></tr>{table}</table>
<p>只有7102通过SU代理判据；7105、7107的CTMRG未收敛。其余大多只有9000步，类型可能在继续优化后合并或改变。A 的两个负磁化位置8/9与论文近似互换，不能由整体空间操作消除。</p>
<p><a href="classification.json">分组方法与阈值敏感性</a> · <a href="family_assignments.csv">分类及收敛标记</a> · <a href="../analysis_20261105/report.html">与论文 D7 的逐位置比较</a></p>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    assert hashes=={n:hashlib.sha256((INPUT/f'{n}.csv').read_bytes()).hexdigest() for n in hashes}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
