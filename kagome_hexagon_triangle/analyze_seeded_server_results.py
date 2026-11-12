"""Read the downloaded D5-seeded chain; compare in a FIXED paper frame.

No evolution, state contraction or source-table edits. Paper coordinates
come from the labelled Fig. S3 reconstruction; all 108 global space-group
operations are tested. An 8/9 exchange is a diagnostic, never an allowed
alignment operation or a claimed measured state.
"""
import hashlib
import html
import json
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analyze import ROOT,PERIOD,reference,transformed_values,canonical,site_position
from .analyze_server_results import read_csv,write_csv,rms,symmetry,permutation
from .geometry import C0,A1,A2,OFFSETS
from .plot_local_D5_D6_vs_paper import panel

INPUT=ROOT/'results/server_from_local_D5_D5_D10_h038_C3_chain'
OLD=ROOT/'results/server_D4_D10_h038_C3_chain'
OUTPUT=INPUT/'analysis_20261101'
PAPER='https://arxiv.org/html/2610.34439v1'
DATA='https://doi.org/10.6084/m9.figshare.34731234.v3'


def align(ps,values,dest,target):
    best=None
    for theta in range(6):
        for reflection in (False,True):
            for x in range(3):
                for y in range(3):
                    aligned=transformed_values(ps,values,dest,theta,reflection,(x,y))
                    if aligned is None:
                        continue
                    residual=rms(aligned-target)
                    if best is None or residual<best['rms']:
                        best=dict(rms=residual,theta=theta,reflection=reflection,shift=[x,y],
                                  correlation=float(np.corrcoef(aligned,target)[0,1]),values=aligned)
    assert best is not None
    return best


def load_runs(folder):
    arrays={name:read_csv(folder/f'{name}.csv') for name in
            ('summary','site_magnetization','bond_energies','triangle_observables','schmidt_spectra')}
    runs={}
    for summary in arrays['summary']:
        D=int(summary['D'])
        sites=[dict(r,sz=float(r['sz'])) for r in arrays['site_magnetization'] if int(r['D'])==D]
        bonds_raw=[r for r in arrays['bond_energies'] if int(r['D'])==D]
        triangles=[r for r in arrays['triangle_observables'] if int(r['D'])==D]
        spectra=[r for r in arrays['schmidt_spectra'] if int(r['D'])==D]
        ps=np.array([[float(r['cartesian_x']),float(r['cartesian_y'])] for r in sites])
        assert all(canonical(p)==canonical(site_position(r)) for p,r in zip(ps,sites))
        lookup={r['site_id']:i for i,r in enumerate(sites)}
        energies={tuple(sorted((lookup[r['site_id_1']],lookup[r['site_id_2']]))):float(r['energy'])
                  for r in bonds_raw}
        assert len(sites)==27 and len(lookup)==27 and len(energies)==54 and len(triangles)==18
        degree=np.zeros(27,dtype=int)
        for i,j in energies:
            difference=ps[i]-ps[j]
            distance=min(np.linalg.norm(difference+x*PERIOD[:,0]+y*PERIOD[:,1])
                         for x in range(-3,4) for y in range(-3,4))
            assert abs(distance-.5)<1e-8
            degree[i]+=1
            degree[j]+=1
        assert np.all(degree==4)
        pairs=[]
        for row in triangles:
            ids=[lookup[row[f'site_id_{i}']] for i in (1,2,3)]
            pairs.extend(tuple(sorted((ids[i],ids[j]))) for i,j in ((0,1),(1,2),(0,2)))
        assert len(set(pairs))==54 and set(pairs)==set(energies)
        sz=np.array([r['sz'] for r in sites])
        E=float(summary['energy_exchange'])
        checks=dict(D=D,sites=27,unique_bonds=54,triangles=18,
                    bond_sum_error=sum(energies.values())/27-E,
                    triangle_sum_error=sum(float(r['exchange_energy']) for r in triangles)/27-E,
                    magnetization_sum_error=2*sz.mean()-float(summary['magnetization_normalized']),
                    total_energy_error=float(summary['energy_total'])-E+float(summary['field'])*sz.mean(),
                    bond_components_max_error=max(abs(float(r['sxx'])+float(r['syy'])+float(r['szz'])-
                                                      float(r['energy'])) for r in bonds_raw),
                    schmidt_probability_sum_error=max(abs(sum(float(r['probability']) for r in spectra
                        if int(r['a'])==a and int(r['k'])==k)-1) for a in range(3) for k in range(3)))
        assert max(abs(v) for k,v in checks.items() if 'error' in k)<1e-10
        runs[D]=dict(summary=summary,sites=sites,ps=ps,sz=sz,energies=energies,checks=checks)
    return arrays,runs


def c3_hexagons(ps,values,source,D):
    lookup={canonical(p):i for i,p in enumerate(ps)}
    rows=[]
    for x in range(3):
        for y in range(3):
            centre=C0+x*A1+y*A2
            perm=permutation(ps,theta=2,centre=centre)
            if rms(values-values[perm])>1e-5:
                continue
            vals=np.array([values[lookup[canonical(centre+p)]] for p in OFFSETS['A']])
            rows.append(dict(source=source,D=D,centre_x=x,centre_y=y,
                             low_sz=float(vals.min()),high_sz=float(vals.max())))
    assert len(rows)==3
    return sorted(rows,key=lambda r:r['low_sz'])


def mark_labels(ax,ps,labels):
    # Annotation labels belong to the FIXED reference geometry, not a
    # permutation of the measured values. Each class occurs three times.
    from .analyze import PERIOD
    coeff=np.linalg.solve(PERIOD,ps.T).T % 1
    coeff[np.isclose(coeff,1,atol=1e-9)]=0
    shown=coeff@PERIOD.T
    for p,label in zip(shown,labels):
        if label in (8,9):
            ax.annotate(str(label),p,xytext=(10,-10),textcoords='offset points',
                        fontsize=9,fontweight='bold',color='#743b85',
                        bbox=dict(facecolor='white',edgecolor='none',pad=.25),zorder=5)


def finish_figure(fig,artist,path,title):
    fig.suptitle(title,fontsize=15,y=.985)
    fig.subplots_adjust(left=.02,right=.91,top=.87,bottom=.13,wspace=.03)
    bar=fig.colorbar(artist,cax=fig.add_axes([.94,.23,.015,.55]))
    bar.set_label(r'Local $\langle S^z\rangle$',fontsize=11)
    fig.text(.46,.035,'Paper geometry/orientation fixed. Only OUR states are globally aligned.\n'
             'Uniform bonds show geometry only; purple 8/9 annotations identify the differing classes.',
             ha='center',fontsize=10,linespacing=1.6)
    fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.15)
    if path.name=='D7_vs_paper_D7.png':
        fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    arrays,runs=load_runs(INPUT)
    _,old=load_runs(OLD)
    ds=sorted(runs)
    assert ds==list(range(ds[0],ds[-1]+1)) and ds[0]==5
    paper_energy={int(1/Fraction(d)):float(e) for d,e in np.loadtxt(
        ROOT/'references/data_fig_2b_energy_vs_D.dat',dtype=str,skiprows=1)}
    rp,rv7=reference(7)
    _,rv15=reference(15)
    map_data=json.loads((ROOT/'references/paper_D15_reference.json').read_text())
    labels=np.array([r['paper_label'] for r in map_data['sites']])
    summary,local_rows,hex_rows,details=[],[],[],[]
    for D in ds:
        run=runs[D]
        row=run['summary']
        assert row['source_state_sha256']=='ac0e8cbc2d9141f6f2574bf830957949d310f90fcd090a98d5f750636734f6bf'
        if D>5:
            assert row['parent_sha256']==runs[D-1]['summary']['state_sha256']
        aligned7=align(run['ps'],run['sz'],rp,rv7)
        aligned15=align(run['ps'],run['sz'],rp,rv15)
        old7=align(old[D]['ps'],old[D]['sz'],rp,rv7)
        branch=align(old[D]['ps'],old[D]['sz'],run['ps'],run['sz'])
        sy=symmetry(run['ps'],run['sz'],run['energies'])
        E=float(row['energy_exchange'])
        record=dict(D=D,chi=int(row['chi']),energy_exchange=E,paper_energy=paper_energy[D],
                    delta_energy_paper=E-paper_energy[D],old_D4_branch_energy=float(old[D]['summary']['energy_exchange']),
                    delta_energy_old=E-float(old[D]['summary']['energy_exchange']),
                    m_normalized=float(row['magnetization_normalized']),
                    max_abs_sx=float(row['max_abs_sx']),SU_proxy=row['su_proxy_converged'],
                    CTM_converged=row['ctm_converged'],face_consistency=float(row['face_consistency_max']),
                    paper_D7_spatial_rms=aligned7['rms'],
                    paper_D7_sorted_rms=rms(np.sort(run['sz'])-np.sort(rv7)),
                    paper_D7_correlation=aligned7['correlation'],
                    paper_D15_spatial_rms=aligned15['rms'],paper_D15_correlation=aligned15['correlation'],
                    old_branch_paper_D7_correlation=old7['correlation'],
                    branch_to_branch_rms=branch['rms'],
                    C3_rms_sz=sy['C3']['best_by_sz']['sz_rms'],
                    mirror_rms_sz=sy['mirror']['best_by_sz']['sz_rms'],
                    mirror_rms_bond=sy['mirror']['minimum_bond_rms'])
        summary.append(record)
        run.update(aligned7=aligned7['values'],aligned15=aligned15['values'],old_aligned7=old7['values'])
        details.append(dict(D=D,symmetry=sy,alignment_to_D7={k:v for k,v in aligned7.items() if k!='values'},
                            alignment_to_D15={k:v for k,v in aligned15.items() if k!='values'}))
        for label in range(1,10):
            mask=labels==label
            ours=aligned7['values'][mask]
            local_rows.append(dict(D=D,paper_label=label,new_sz=float(ours.mean()),
                new_class_spread=float(np.ptp(ours)),paper_D7_sz=float(rv7[mask].mean()),
                difference=float((ours-rv7[mask]).mean())))
        hex_rows.extend(c3_hexagons(rp,aligned7['values'],'new_D5_chain',D))
    hex_rows.extend(c3_hexagons(rp,rv7,'paper_D7',7))
    D=ds[-1]
    if D==7:
        # Quantify the suspected exchange, but DO NOT accept it as a
        # physical alignment or generate a fake "corrected" measured state.
        exchanged=rv7.copy()
        exchanged[labels==8]=rv7[labels==9].mean()
        exchanged[labels==9]=rv7[labels==8].mean()
        diagnostic=dict(description='Artificial exchange of paper classes 8 and 9; not a space-group operation',
                        rms_after_artificial_8_9_exchange=rms(runs[7]['aligned7']-exchanged),
                        rms_on_other_seven_classes=rms((runs[7]['aligned7']-rv7)[labels<8]))
    else:
        diagnostic={}

    fig,axes=plt.subplots(1,2,figsize=(12.2,5.7))
    artist=panel(axes[0],rp,runs[D]['aligned7'],f'New D={D}, from saved D5; h=0.38',
                  f"Eex={float(runs[D]['summary']['energy_exchange']):.9f}; chi=40")
    panel(axes[1],rp,rv7,'Paper D=7, h=0.4; FIXED reference',
          f"Spatial correlation={summary[-1]['paper_D7_correlation']:.5f}")
    for ax in axes:mark_labels(ax,rp,labels)
    finish_figure(fig,artist,OUTPUT/'D7_vs_paper_D7.png','Energy matches closely; negative-magnetization classes differ in position')

    fig,axes=plt.subplots(len(ds),3,figsize=(18.3,5.2*len(ds)),squeeze=False)
    for i,D in enumerate(ds):
        r=runs[D]
        artist=panel(axes[i,0],rp,r['old_aligned7'],f'Previous D4 chain: D={D}, h=0.38',
                      f"Eex={float(old[D]['summary']['energy_exchange']):.9f}")
        panel(axes[i,1],rp,r['aligned7'],f'New saved-D5 chain: D={D}, h=0.38',
              f"Eex={float(r['summary']['energy_exchange']):.9f}")
        panel(axes[i,2],rp,rv7,'Paper D=7, h=0.4; same orientation in every row',
              f"New correlation={summary[i]['paper_D7_correlation']:.5f}")
        for ax in axes[i]:mark_labels(ax,rp,labels)
    finish_figure(fig,artist,OUTPUT/'branches_vs_paper_D7.png','Downloaded results: D5, D6, D7 only')

    fig,axes=plt.subplots(1,3,figsize=(15.5,4.5))
    axes[0].plot(ds,[r['energy_exchange'] for r in summary],'o-',label='New D5 chain')
    axes[0].plot(ds,[r['paper_energy'] for r in summary],'s--',label='Paper same D')
    axes[0].plot(ds,[r['old_D4_branch_energy'] for r in summary],'^:',label='Previous D4 chain')
    axes[0].set(xlabel='D',ylabel='Exchange energy per spin')
    axes[0].legend(fontsize=9)
    axes[1].plot(ds,[1e6*r['delta_energy_paper'] for r in summary],'o-')
    axes[1].axhline(0,color='gray',lw=.8)
    axes[1].set(xlabel='D',ylabel='(New - paper) energy x 10^6')
    axes[2].plot(range(1,10),[r['new_sz'] for r in local_rows if r['D']==7],'o-',label='New D7, spatial labels')
    axes[2].plot(range(1,10),[rv7[labels==i].mean() for i in range(1,10)],'s--',label='Paper D7')
    axes[2].set(xlabel='Fig. S3 site class (not sorted)',ylabel='Local <Sz>')
    axes[2].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT/'energy_and_label_comparison.png',dpi=175,bbox_inches='tight')
    plt.close(fig)

    write_csv(OUTPUT/'comparison.csv',summary)
    write_csv(OUTPUT/'site_class_comparison.csv',local_rows)
    write_csv(OUTPUT/'c3_hexagons.csv',hex_rows)
    write_csv(OUTPUT/'consistency_checks.csv',[r['checks'] for r in runs.values()])
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (INPUT,OLD) for p in folder.glob('*.csv')}
    result=dict(summary=summary,details=details,diagnostic=diagnostic,
                input_sha256=hashes,paper=PAPER,author_data=DATA,
                downloaded_D=ds,missing_D=[d for d in range(5,11) if d not in ds],
                note='Only summary tables downloaded; server live status and SU drift histories unavailable')
    (OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+\
            ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    energy_table=table(['D','新分支Eex/site','论文同D','新−论文','m/ms','SU proxy','CTM'],
        [[r['D'],f"{r['energy_exchange']:.9f}",f"{r['paper_energy']:.9f}",f"{r['delta_energy_paper']:+.3e}",
          f"{r['m_normalized']:.9f}",r['SU_proxy'],r['CTM_converged']] for r in summary])
    labels_table=table(['Fig.S3标签','新D7 Sz','论文D7 Sz','差值'],
        [[r['paper_label'],f"{r['new_sz']:.6f}",f"{r['paper_D7_sz']:.6f}",f"{r['difference']:+.6f}"]
         for r in local_rows if r['D']==7])
    metric_table=table(['D','新分支对论文D7相关系数','旧D4分支对应值','新空间Sz RMS','新排序Sz RMS'],
        [[r['D'],f"{r['paper_D7_correlation']:.5f}",f"{r['old_branch_paper_D7_correlation']:.5f}",
          f"{r['paper_D7_spatial_rms']:.5f}",f"{r['paper_D7_sorted_rms']:.5f}"] for r in summary])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>D5初态服务器分支：截至D7与论文比较</title>
<style>body{{font:16px/1.7 system-ui;color:#243443;max-width:1200px;margin:32px auto;padding:0 24px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:8px;border-bottom:1px solid #cbd5df;text-align:right}}th{{background:#edf2f6}}.note{{background:#fff2d8;padding:15px}}</style>
<h1>从本机D5出发的分支：能量与论文很接近，仍有明确的负磁化位置差别</h1>
<p>本次下载表包含D5、D6、D7。没有D8–D10数据，也没有chain_status、SU历史或最终张量，因此无法判断服务器是否仍在计算D8。这里仅分析已下载结果，不启动张量计算。</p>
<p>初态SHA256与原本机D5保存态一致；D6和D7父态hash链连续。原本机D5只是较像论文的初步态，本轮先继续演化D5，再逐级扩充。h=0.38，χ=40，CTMRG/观测量cuda:1。</p>
<h2>能量及收敛</h2>{energy_table}
<p>这里比较不含磁场的交换能Eex，避免将−h·mean(Sz)重复计入。D5、6比论文同D低约6–7×10⁻⁶/site，D7高约1.07×10⁻⁶/site，可称在现有收缩精度下非常接近，不能认定严格等能或超越论文。全部SU代理和CTMRG标记通过；D5、6、7的末态max|Sx|分别约4.03×10⁻⁷、1.04×10⁻⁷、6.70×10⁻⁹。</p>
<p class="note">SU代理通过不证明全局最优；χ只有40，尚未验证χ收敛。D6、D7的环境位置一致性指标约3.0×10⁻⁶和2.0×10⁻⁶，不能当成严格的能量误差界。旧D4分支SU代理未通过，本轮通过，初始化和演化充分程度同时不同，不能把差异全部归因于初始化。</p>
<h2>D7与论文同D的逐点比较</h2>
<p>论文磁化为h=0.4，本次为h=0.38。纸面编号根据补充Fig.S3映射，原始列按1–9解释；重新核对了编号图，标签8位于上边、标签9位于右斜边。图中固定论文位置与方向，仅将本次态做整体平移、旋转或反射，遍历108种操作，没有任意交换site。</p>
<img src="D7_vs_paper_D7.png">{labels_table}
<p>七类磁化相差不超过0.00583。主要差别是8、9这两类负磁化（每类3个site）近似对调：新D7在8处为−0.15247、在9处为−0.07758；论文分别为−0.07518和−0.15471。完整空间RMS=0.03653，排序RMS=0.003274。人为互换参考8/9后RMS也为约0.003274，但这不是允许的全局空间操作，不能据此声称原图案完全相同。</p>
<p>六边形的配对也体现同一差别：新D7最强正磁化约+0.27893与−0.15247交替，论文约+0.27937与−0.07518交替。另一类六边形新态为(+0.24009,−0.07758)，论文为(+0.23997,−0.15471)。因此九类数值集合几乎一致，空间配对仍有不同。</p>
<h2>与旧D4分支比较</h2>{metric_table}
<img src="branches_vs_paper_D7.png"><img src="energy_and_label_comparison.png">
<p>截至D7，新分支明显更接近论文D7的局域磁化分布，也比旧分支低约1.22×10⁻⁴/site。但全局相似度高不足以证明同一态，仍应追踪8/9的相对位置。C3是ansatz预先施加的；本轮D7镜面Sz RMS约0.05164，镜面明显破缺。</p>
<h2>数据检查和下一步</h2>
<p>27个site、54条唯一最近邻bond和18个基本三角形齐全，每个site度数4，每条bond恰好属于一个三角形。bond和triangle求和、Zeeman项、磁化及SU权重归一化均吻合到浮点精度。此检查验证表格汇总，不替代张量收缩的独立交叉验证。</p>
<p>先等D8–D10的同一分支数据，检查8/9位置差别是否持续；在保存的D7态上补χ收敛测量，并在相同h下对照。能量接近且图案不同是近简并候选的线索，但当前没有论文张量，也没有局域缺陷计算，不能由此证明局域零能操作或宏观简并。</p>
<p>作者公开数据没有逐条bond数值，无法完成论文逐bond定量匹配。能量来自Fig.2b，局域Sz来自补充Fig.S3。D5、D6没有公开的同D局域Sz，图中用最接近的公开D7作参考。</p>
<p><a href="comparison.csv">逐D比较</a> · <a href="site_class_comparison.csv">九类磁化</a> · <a href="c3_hexagons.csv">C3六边形配对</a> · <a href="consistency_checks.csv">汇总检查</a> · <a href="analysis.json">完整指标和输入hash</a> · <a href="{PAPER}">论文</a> · <a href="{DATA}">作者数据v3</a></p></html>'''
    (OUTPUT/'report.html').write_text(page,encoding='utf-8')
    for path,digest in hashes.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    print(json.dumps(dict(summary=summary,diagnostic=diagnostic,hexagons=hex_rows,report=str(OUTPUT/'report.html')),indent=2))


if __name__=='__main__':
    main()
