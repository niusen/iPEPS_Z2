"""Read-only analysis of the physical-NN/six-then-C3 random D7 CSV download.

No evolution or CTMRG. Use the author's same-D=7, h=.4 local-Sz data and only
108 global space-group operations. Raw CSVs and existing analysis are untouched.
"""
import hashlib
import html
import json
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analyze import ROOT, PERIOD, canonical, reference, site_position
from .analyze_server_results import read_csv, rms, symmetry, permutation
from .analyze_seeded_server_results import align
from .analyze_random_D7_results import source_indices, reordered_bonds
from .plot_local_D5_D6_vs_paper import panel

INPUT = ROOT/'results/server_D7_random10_h040_physicalNN_six_then_C3'
OUTPUT = INPUT/'analysis_20261108'
PAPER = 'https://arxiv.org/html/2610.34439v1'
DATA = 'https://doi.org/10.6084/m9.figshare.34731234.v3'


def load():
    names = ('summary', 'all_chi_measurements', 'site_magnetization',
             'bond_energies', 'triangle_observables', 'schmidt_spectra')
    arrays = {n: read_csv(INPUT/f'{n}.csv') for n in names}
    hashes = {n: hashlib.sha256((INPUT/f'{n}.csv').read_bytes()).hexdigest() for n in names}
    runs, checks, chi_checks = {}, [], []
    for s in arrays['summary']:
        seed, chi = int(s['seed']), int(s['chi'])
        assert seed not in runs and int(s['D']) == 7 and float(s['field']) == .4
        assert chi == 60 and int(s['steps']) == 9000
        variants = {}
        for ch in (40, 60):
            chosen = lambda name: [r for r in arrays[name] if int(r['seed'])==seed and int(r['chi'])==ch]
            sites, bonds, triangles, spectra = (chosen(n) for n in names[2:])
            obs = chosen('all_chi_measurements')
            assert len(obs)==1
            obs = obs[0]
            assert obs['measurement_state_sha256']==s['state_sha256']
            ps = np.array([[float(r['cartesian_x']), float(r['cartesian_y'])] for r in sites])
            sz = np.array([float(r['sz']) for r in sites])
            assert all(canonical(p)==canonical(site_position(r)) for p,r in zip(ps,sites))
            lookup = {r['site_id']: i for i,r in enumerate(sites)}
            energies = {tuple(sorted((lookup[r['site_id_1']],lookup[r['site_id_2']]))):float(r['energy']) for r in bonds}
            assert len(sites)==len(lookup)==27 and len(bonds)==len(energies)==54 and len(triangles)==18
            degree = np.zeros(27,int)
            for i,j in energies:
                distance = min(np.linalg.norm(ps[i]-ps[j]+x*PERIOD[:,0]+y*PERIOD[:,1])
                               for x in range(-2,3) for y in range(-2,3))
                assert abs(distance-.5)<1e-8
                degree[i]+=1; degree[j]+=1
            assert np.all(degree==4)
            triangle_pairs=[]
            for row in triangles:
                ids=[lookup[row[f'site_id_{i}']] for i in (1,2,3)]
                triangle_pairs.extend(tuple(sorted((ids[i],ids[j]))) for i,j in ((0,1),(1,2),(0,2)))
            assert len(triangle_pairs)==54 and len(set(triangle_pairs))==54 and set(triangle_pairs)==set(energies)
            audit=dict(seed=seed,chi=ch,
                bond_sum_error=sum(energies.values())/27-float(obs['energy_exchange']),
                triangle_sum_error=sum(float(r['exchange_energy']) for r in triangles)/27-float(obs['energy_exchange']),
                magnetization_error=2*sz.mean()-float(obs['magnetization_normalized']),
                field_energy_error=-.4*sz.mean()-float(obs['energy_field']),
                total_energy_error=float(obs['energy_exchange'])+float(obs['energy_field'])-float(obs['energy_total']),
                bond_components_error=max(abs(sum(float(r[k]) for k in ('sxx','syy','szz'))-float(r['energy'])) for r in bonds),
                triangle_sz_error=max(abs(float(r['sum_sz'])-sum(sz[lookup[r[f'site_id_{i}']]] for i in (1,2,3))) for r in triangles),
                schmidt_norm_error=max(abs(sum(float(r['probability']) for r in spectra
                    if int(r['a'])==a and int(r['k'])==k)-1) for a in range(3) for k in range(3)))
            assert max(abs(v) for k,v in audit.items() if k not in ('seed','chi'))<1e-10
            checks.append(audit)
            variants[ch]=dict(summary=obs,sites=sites,ps=ps,sz=sz,energies=energies,
                sz_by_id={r['site_id']:float(r['sz']) for r in sites},
                bond_by_id={tuple(sorted((r['site_id_1'],r['site_id_2']))):float(r['energy']) for r in bonds})
        a,b=variants[40],variants[60]
        chi_checks.append(dict(seed=seed,
            energy_exchange_difference=abs(float(a['summary']['energy_exchange'])-float(b['summary']['energy_exchange'])),
            energy_total_difference=abs(float(a['summary']['energy_total'])-float(b['summary']['energy_total'])),
            max_site_sz_difference=max(abs(v-b['sz_by_id'][k]) for k,v in a['sz_by_id'].items()),
            max_bond_difference=max(abs(v-b['bond_by_id'][k]) for k,v in a['bond_by_id'].items()),
            reported_pass=s['chi_check_passed']))
        runs[seed]=dict(variants[60],summary=s)
        assert abs(float(s['energy_total'])-float(b['summary']['energy_total']))<1e-14
    assert sorted(runs)==list(range(7101,7111))
    return arrays, hashes, runs, checks, chi_checks


def main():
    OUTPUT.mkdir(exist_ok=True)
    arrays,hashes,runs,checks,chi_checks=load()
    metadata=json.loads((ROOT/'references/figshare_metadata.json').read_text())
    for name in ('data_fig_S3_mag_vs_D.dat','data_fig_2b_energy_vs_D.dat'):
        md5=hashlib.md5((ROOT/'references'/name).read_bytes()).hexdigest()
        assert md5==next(r['computed_md5'] for r in metadata['files'] if r['name']==name)
    ps,paper7=reference(7)
    _,paper15=reference(15)
    labels=np.array([r['paper_label'] for r in json.loads((ROOT/'references/paper_D15_reference.json').read_text())['sites']])
    assert all(np.count_nonzero(labels==i)==3 for i in range(1,10))
    paper_energy={int(1/Fraction(d)):float(e) for d,e in np.loadtxt(
        ROOT/'references/data_fig_2b_energy_vs_D.dat',dtype=str,skiprows=1)}[7]
    paper_total=paper_energy-.4*paper7.mean()
    paper_C3_rms=min(rms(paper7-paper7[permutation(ps,theta=2,shift=(x,y))])
                     for x in range(3) for y in range(3))
    assert paper_C3_rms<1e-12
    rows, classes, details = [], [], []
    for seed,run in sorted(runs.items()):
        s=run['summary']
        a7=align(run['ps'],run['sz'],ps,paper7)
        a15=align(run['ps'],run['sz'],ps,paper15)
        indices=source_indices(run['ps'],ps,a7)
        sy=symmetry(run['ps'],run['sz'],run['energies'])
        run.update(aligned=a7['values'],indices=indices,aligned_bonds=reordered_bonds(run,indices))
        row=dict(seed=seed,chi=int(s['chi']),energy_exchange=float(s['energy_exchange']),
            energy_total=float(s['energy_total']),delta_exchange_paper=float(s['energy_exchange'])-paper_energy,
            delta_total_paper=float(s['energy_total'])-paper_total,
            m_normalized=float(s['magnetization_normalized']),max_abs_sx=float(s['max_abs_sx']),
            su_proxy=s['su_proxy_converged'],c3_reduced=s['c3_final'],chi_pass=s['chi_check_passed'],
            ctm_residual=float(s['ctm_residual']),
            rms_D7=a7['rms'],correlation_D7=a7['correlation'],sorted_rms_D7=rms(np.sort(run['sz'])-np.sort(paper7)),
            rms_D15=a15['rms'],correlation_D15=a15['correlation'],
            C3_site_rms=sy['C3']['best_by_sz']['sz_rms'],C3_bond_rms=sy['C3']['best_by_sz']['bond_rms'],
            mirror_site_rms=sy['mirror']['best_by_sz']['sz_rms'],mirror_bond_rms=sy['mirror']['best_by_sz']['bond_rms'])
        exchanged=paper7.copy()
        exchanged[labels==8]=paper7[labels==9].mean()
        exchanged[labels==9]=paper7[labels==8].mean()
        row['diagnostic_forbidden_8_9_exchange_rms']=rms(run['aligned']-exchanged)
        rows.append(row)
        for label in range(1,10):
            v=run['aligned'][labels==label]
            classes.append(dict(seed=seed,paper_label=label,paper_D7_sz=float(paper7[labels==label].mean()),
                our_mean_sz=float(v.mean()),our_min_sz=float(v.min()),our_max_sz=float(v.max()),
                our_C3_orbit_range=float(np.ptp(v)),our_values=v.tolist()))
        details.append(dict(seed=seed,alignment_D7={k:v for k,v in a7.items() if k!='values'},
            alignment_D15={k:v for k,v in a15.items() if k!='values'},symmetry=sy))
    pairwise=[]
    seeds=sorted(runs)
    for i,s in enumerate(seeds):
        for t in seeds[:i]:
            a,b=runs[s],runs[t]
            aligned=align(a['ps'],a['sz'],b['ps'],b['sz'])
            idx=source_indices(a['ps'],b['ps'],aligned)
            bonds=reordered_bonds(a,idx)
            pairwise.append(dict(seed1=s,seed2=t,rms_sz=aligned['rms'],
                bond_rms_at_best_sz=rms([v-b['energies'][k] for k,v in bonds.items()])))
    result=dict(paper=PAPER,author_data=DATA,paper_D7_exchange_energy=paper_energy,
        paper_D7_total_energy_at_h04=paper_total,paper_D7_m_normalized=float(2*paper7.mean()),
        method='108 global rotations/reflections/translations; fixed paper geometry; no individual site exchanges',
        primary_reference='Author Fig.S3 data at D7 h0.4. D15 only a secondary comparison.',
        notes='Only downloaded CSVs; no final states, SU histories or C3 logs. Real tensor ansatz fixes scalar chirality to zero; not independent evidence.',
        source_sha256=hashes,checks=checks,chi_checks=chi_checks,summary=rows,
        paper_C3_site_rms=paper_C3_rms,site_classes=classes,alignments=details,pairwise=pairwise)
    (OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PAPER D7',paper_energy,'total',paper_total,'m/ms',2*paper7.mean())
    for r in sorted(rows,key=lambda x:x['rms_D7']):
        print(f"{r['seed']}: Eex={r['energy_exchange']:.10f} m={r['m_normalized']:.7f} "
              f"RMS7={r['rms_D7']:.6f} corr7={r['correlation_D7']:.5f} sorted={r['sorted_rms_D7']:.6f} "
              f"RMS15={r['rms_D15']:.6f} C3={r['C3_site_rms']:.6f} "
              f"Sx={r['max_abs_sx']:.6f} swap89={r['diagnostic_forbidden_8_9_exchange_rms']:.6f}")
    print('CHI',json.dumps(chi_checks))
    print('CLOSEST PAIRS',json.dumps(sorted(pairwise,key=lambda r:r['rms_sz'])[:5]))
    print('PAPER C3 RMS',paper_C3_rms)
    print('SEED7103 CLASSES',json.dumps([r for r in classes if r['seed']==7103]))
    make_figures(runs,rows,classes,ps,paper7,paper15,paper_energy)
    make_report(rows,chi_checks,paper_energy,paper_total)
    assert hashes=={n:hashlib.sha256((INPUT/f'{n}.csv').read_bytes()).hexdigest() for n in hashes}
    print('RAW DATA UNCHANGED; REPORT:',OUTPUT/'report.html')


def make_figures(runs,rows,classes,ps,paper7,paper15,paper_energy):
    ranked=sorted(rows,key=lambda r:r['rms_D7'])
    best=min(rows,key=lambda r:r['energy_total'])['seed']
    shown=[r['seed'] for r in ranked[:2]]
    third=best if best not in shown else ranked[2]['seed']
    shown.append(third)
    fig,axes=plt.subplots(2,2,figsize=(12.2,12.2))
    artist=panel(axes[0,0],ps,paper7,'Paper D=7, h=0.4',f'Eex/site={paper_energy:.9f}; fixed orientation')
    for ax,seed in zip((axes[0,1],axes[1,0],axes[1,1]),shown):
        r=next(r for r in rows if r['seed']==seed)
        panel(ax,ps,runs[seed]['aligned'],f'Seed {seed}; physical-NN SU, D=7',
            f"Eex={r['energy_exchange']:.9f}; m/ms={r['m_normalized']:.7f}\n"
            f"Sz RMS={r['rms_D7']:.4f}; corr={r['correlation_D7']:.3f}; max|Sx|={r['max_abs_sx']:.3f}")
    fig.subplots_adjust(left=.015,right=.91,top=.86,bottom=.075,hspace=.45,wspace=.04)
    fig.colorbar(artist,cax=fig.add_axes([.94,.23,.015,.52]),label=r'Local $\langle S^z\rangle$')
    fig.suptitle('Physical-NN random D7 states versus author D7 data',fontsize=15,y=.985)
    fig.text(.46,.025,'Only global lattice operations; same colour/size scales. Uniform bonds show geometry.\n'
        'All ten starts exhausted the 9000-step budget without passing the SU proxy.',ha='center',fontsize=10)
    fig.savefig(OUTPUT/'key_patterns_vs_paper.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUTPUT/'key_patterns_vs_paper.svg',bbox_inches='tight');plt.close(fig)

    nearest=ranked[0]
    fig,axes=plt.subplots(1,2,figsize=(12.2,6.7))
    artist=panel(axes[0],ps,paper7,'Paper D=7, h=0.4',f'Eex/site={paper_energy:.9f}; fixed orientation')
    panel(axes[1],ps,runs[nearest['seed']]['aligned'],f"Closest: seed {nearest['seed']}, D=7, h=0.4",
        f"Eex={nearest['energy_exchange']:.9f}; m/ms={nearest['m_normalized']:.7f}\n"
        f"Sz RMS={nearest['rms_D7']:.4f}; max|Sx|={nearest['max_abs_sx']:.3f}")
    fig.subplots_adjust(left=.015,right=.91,top=.76,bottom=.12,wspace=.04)
    fig.colorbar(artist,cax=fig.add_axes([.94,.22,.015,.46]),label=r'Local $\langle S^z\rangle$')
    fig.suptitle('Same D and field; best global lattice alignment',fontsize=15,y=.985)
    fig.text(.46,.035,'Shared colour and marker-size scales. Uniform bonds show geometry only.\n'
        'The closest candidate has not passed the SU proxy and does not yet show paper-like C3 symmetry.',
        ha='center',fontsize=10)
    fig.savefig(OUTPUT/'closest_vs_paper.png',dpi=180,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(4,3,figsize=(17.4,21.6))
    artist=panel(axes.flat[0],ps,paper7,'Paper D=7, h=0.4','Reference, fixed orientation')
    for ax,r in zip(list(axes.flat)[1:11],ranked):
        panel(ax,ps,runs[r['seed']]['aligned'],f"Seed {r['seed']}; chi=60",
            f"Eex={r['energy_exchange']:.7f}; m/ms={r['m_normalized']:.5f}\n"
            f"RMS7={r['rms_D7']:.4f}; C3 RMS={r['C3_site_rms']:.4f}")
    panel(axes.flat[11],ps,paper15,'Paper D=15, h=0.4','Secondary finite-D reference; same frame')
    fig.subplots_adjust(left=.015,right=.93,bottom=.035,top=.91,hspace=.43,wspace=.05)
    fig.colorbar(artist,cax=fig.add_axes([.95,.2,.012,.58]),label=r'Local $\langle S^z\rangle$')
    fig.suptitle('All ten physical-NN starts, ranked by same-D paper-pattern RMS',fontsize=16,y=.985)
    fig.text(.47,.012,'Global alignments only. All SU proxies are false; these are candidates, not ten converged phases.',
        ha='center',fontsize=11)
    fig.savefig(OUTPUT/'all_10_patterns.png',dpi=150,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5))
    for ax,r in zip(axes,ranked[:2]):
        seed=r['seed'];x=np.arange(1,10)
        ax.plot(x,[paper7[labels].mean() for labels in
            [np.array([c['paper_label'] for c in json.loads((ROOT/'references/paper_D15_reference.json').read_text())['sites']])==i for i in x]],
            'D-',color='#303030',label='Paper D7')
        selected=[c for c in classes if c['seed']==seed]
        for i,c in enumerate(selected):
            ax.scatter(i+1+np.array([-.08,0,.08]),c['our_values'],s=25,color='#b34842',
                label=f'Seed {seed}: 3 related sites' if i==0 else None)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel='Paper site class',ylabel=r'$\langle S^z\rangle$',xticks=x,
            title=f'Seed {seed}: individual sites retained')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUTPUT/'site_classes.png',dpi=170,bbox_inches='tight');plt.close(fig)


def make_report(rows,chi_checks,paper_energy,paper_total):
    nearest=min(rows,key=lambda r:r['rms_D7'])
    minimum_c3=min(r['C3_site_rms'] for r in rows)
    columns=[('seed','seed'),('energy_exchange','Eex/site'),('m_normalized','m/ms'),('rms_D7','与论文D7的Sz RMS'),
        ('correlation_D7','相关系数'),('sorted_rms_D7','排序RMS'),('C3_site_rms','C3磁化RMS'),
        ('max_abs_sx','最大|Sx|'),('chi_pass','跨χ检查')]
    body=''
    for row in sorted(rows,key=lambda r:r['rms_D7']):
        body+='<tr>'+''.join('<td>'+html.escape(f'{row[k]:.7g}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k,_ in columns)+'</tr>'
    chi_body=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in
        ('seed','energy_total_difference','max_site_sz_difference','max_bond_difference','reported_pass'))+'</tr>' for row in chi_checks)
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D7 physical-NN 随机态图案分析</title>
<style>body{{font:16px/1.65 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#263443}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ccd2d8;text-align:right}}.table{{overflow-x:auto}}a{{color:#205e98}}</style>
<h1>D=7、h=0.4：逐物理最近邻门的十次随机初态</h1>
<p><strong>当前十个结果中没有找到可以认定与论文一致的图案。</strong>同D对照最接近的是seed {nearest['seed']}，但Sz RMS仍为{nearest['rms_D7']:.5f}，相关系数{nearest['correlation_D7']:.4f}；所有候选自身的最小C3磁化残差都不小于{minimum_c3:.5f}，因此差异并非仅来自整体朝向或图上标号。数据还不能区分演化中的过渡态与稳定的竞争态。</p>
<p>最高χ=60作为主要结果，同一态χ40/60独立测量。论文主要对照也取D=7、h=0.4，参考交换能={paper_energy:.12f}，同磁场总能={paper_total:.12f}。D15只作为补充图案对照。</p>
<p>十次均完成9000步预算，但SU代理均未通过、均保持六张量。CTMRG收敛不等于SU已收敛；C3约化未触发也不自动等于物理C3完全破缺，故另算磁化和bond的对称残差。</p>
<p>只允许108种整体平移、旋转和反射，论文坐标系固定；未单独交换任何site。排序RMS允许任意置换，仅作为数值集合差异的下界。论文site标号位置继承此前Fig.S3重建，另报告不依赖重建的排序比较。所有图采用同一Sz颜色和点大小；均匀bond只表示几何，不是作者逐bond能量。</p>
<div class="table"><table><tr>{''.join('<th>'+v+'</th>' for _,v in columns)}</tr>{body}</table></div>
<img src="closest_vs_paper.png"><img src="key_patterns_vs_paper.png"><img src="site_classes.png">
<h2>全部十次</h2><a href="all_10_patterns.png"><img src="all_10_patterns.png"></a>
<h2>同一态的χ40/60检查</h2><div class="table"><table><tr><th>seed</th><th>总能差</th><th>最大site Sz差</th><th>最大bond差</th><th>通过</th></tr>{chi_body}</table></div>
<p>已核对27个site、54条不同NN键且每site度数4、18个三角形、能量与磁化求和、Schmidt归一化、各χ测量绑定同一态SHA；误差均小于1e-10。它们核查数据内部一致性，不是独立CTMRG正确性证明。</p>
<p>本次仅CSV解读，没有读取checkpoint、继续演化或重算CTMRG。下载目录未包含各seed的SU历史和C3约化日志，不能据此断言某个候选已达到最终稳定图案。</p>
<p><a href="analysis.json">完整指标、空间对齐参数和原始文件SHA256</a> · <a href="{PAPER}">论文</a> · <a href="{DATA}">作者数据</a></p></html>'''
    (OUTPUT/'report.html').write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
