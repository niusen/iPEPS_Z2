"""Read-only analysis of the two independent dense SU jobs, updated per point."""
from pathlib import Path
import json,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[3]
PY=REPO/'kagome_hexagon_triangle/results/DMI_plateau_dense_D2_D4_20261113'
JL=Path('D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/Kagome/simple_update/multicell/results/DMI_plateau_dense_D2_D4_20261113')
CENTERS=JL.parent/'DMI_plateau_fresh_centers_20261114'
REFINED=JL.parent/'DMI_plateau_convergence_checks_20261114'
OUT=REPO/'kagome_DMI_plateau_review_20261113'


def run(quiet=False):
    OUT.mkdir(parents=True,exist_ok=True);rows=[];sites=[]
    for name,root in (('python_hexagon_triangle',PY),('julia_ipess',JL)):
        for p in sorted(root.glob('D*_Dz*_*/*/observables.json')):
            try:r=json.loads(p.read_text())
            except (OSError,json.JSONDecodeError):continue
            recheck=p.with_name('observables_recheck.json')
            chosen_path=p
            if not r['valid_measurement'] and recheck.exists():
                r=json.loads(recheck.read_text())
                chosen_path=recheck
            py=name.startswith('python');ctm=r.get('ctm_converged',False) if py else r.get('ctm',{}).get('converged',False)
            transverse=[s['transverse_moment'] for s in r.get('sites',[])] if py else [(x*x+y*y)**.5 for c in r.get('cells',[]) for x,y in zip(c['Sx'],c['Sy'])]
            row=dict(ansatz=name,D=r['D'],Dz=r['Dz'],h=r['h_physical'],branch=r['branch'],
                energy=r.get('energy_total' if py else 'energy_per_spin'),m=r.get('m_normalized'),
                measurement_valid=r['valid_measurement'],ctm_converged=ctm,
                accepted=bool(r['valid_measurement'] and ctm),chi=r.get('chi'),
                error=r.get('error',r.get('original_rejection','')),source=str(chosen_path),
                measurement_basis=r.get('recheck_acceptance','original strict measurement'),
                max_imaginary_residual=r.get('max_hermitian_imaginary_part'),
                chi_energy_difference=r.get('chi_energy_difference'),chi_m_difference=r.get('chi_m_difference'))
            row['max_transverse_moment']=max(transverse) if transverse else None
            row['distance_to_one_ninth']=None if row['m'] is None else row['m']-1/9
            rows.append(row)
            if py:
                ss=[dict(site=f"{s['kind']}_{s['colour']}_{s['site']}",Sx=s['sx'],Sy=s['sy'],Sz=s['sz']) for s in r.get('sites',[])]
            else:
                ss=[dict(site=f"{c['x']}_{c['y']}_{b+1}",Sx=c['Sx'][b],Sy=c['Sy'][b],Sz=c['Sz'][b]) for c in r.get('cells',[]) for b in range(3)]
            sites.extend([dict(ansatz=name,D=row['D'],Dz=row['Dz'],h=row['h'],branch=row['branch'],
                accepted=row['accepted'],measurement_basis=row['measurement_basis'],source=row['source'],**s) for s in ss])
    for file,data in (('all_points',rows),('site_magnetization',sites)):
        if data:
            with (OUT/f'{file}.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    groups={}
    all_groups={}
    for r in rows:
        all_groups.setdefault((r['ansatz'],r['D'],r['Dz'],r['h']),[]).append(r)
        if r['accepted']:groups.setdefault((r['ansatz'],r['D'],r['Dz'],r['h']),[]).append(r)
    best=[]
    for key,values in sorted(groups.items()):
        b=min(values,key=lambda r:r['energy']);v=dict(b)
        v['branch_m_spread']=max(r['m'] for r in values)-min(r['m'] for r in values)
        v['branch_energy_spread']=max(r['energy'] for r in values)-min(r['energy'] for r in values)
        v['branches_accepted']=len(values)
        v['branches_saved']=len(all_groups[key])
        v['comparison_complete']=len(values)==2
        v['near_one_ninth']=abs(v['m']-1/9)<5e-4
        best.append(v)
    (OUT/'best_points.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
    if best:
        with (OUT/'best_points.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(best[0]));w.writeheader();w.writerows(best)
    centers=[]
    for p in sorted(CENTERS.glob('D*_Dz*_*/*/observables.json')):
        r=json.loads(p.read_text());tt=[(x*x+y*y)**.5 for c in r.get('cells',[]) for x,y in zip(c['Sx'],c['Sy'])]
        base=[v for v in best if v['ansatz']=='julia_ipess' and v['D']==r['D'] and v['Dz']==r['Dz'] and v['h']==r['h_physical']]
        centers.append(dict(D=r['D'],Dz=r['Dz'],h=r['h_physical'],trial=r['branch'],
            energy=r.get('energy_per_spin'),m=r.get('m_normalized'),max_transverse=max(tt) if tt else None,
            chi=r['chi'],valid=r['valid_measurement'] and r.get('ctm',{}).get('converged',False),
            energy_minus_previous_best=r.get('energy_per_spin')-base[0]['energy'] if base and r.get('energy_per_spin') is not None else None))
    if centers:
        with (OUT/'fresh_center_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(centers[0]));w.writeheader();w.writerows(centers)
    refined=[]
    for p in sorted(REFINED.glob('D*_Dz*_*/*/observables.json')):
        r=json.loads(p.read_text())
        for chi in (24,28):
            higher=p.with_name(f'observables_chi{chi}.json')
            if higher.exists():
                h=json.loads(higher.read_text())
                if h.get('ctm',{}).get('converged',False):r=h
        tt=[(x*x+y*y)**.5 for c in r.get('cells',[]) for x,y in zip(c['Sx'],c['Sy'])]
        refined.append(dict(D=r['D'],Dz=r['Dz'],h=r['h_physical'],trial=r['branch'],
            energy=r.get('energy_per_spin'),m=r.get('m_normalized'),max_transverse=max(tt) if tt else None,
            chi=r['chi'],valid=r['valid_measurement'] and r.get('ctm',{}).get('converged',False)))
    if refined:
        with (OUT/'continued_candidate_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(refined[0]));w.writeheader();w.writerows(refined)
    available=[dict(ansatz=r['ansatz'],D=r['D'],Dz=r['Dz'],h=r['h'],state=r['branch'],energy=r['energy'],m=r['m'],chi=r['chi'],origin='field sweep') for r in rows if r['accepted']]
    for origin,extra in (('independent random center',centers),('continued independent center',refined)):
        available.extend(dict(ansatz='julia_ipess',D=r['D'],Dz=r['Dz'],h=r['h'],state=r['trial'],energy=r['energy'],m=r['m'],chi=r['chi'],origin=origin) for r in extra if r['valid'])
    grouped={}
    for r in available:grouped.setdefault((r['ansatz'],r['D'],r['Dz'],r['h']),[]).append(r)
    all_best=[min(v,key=lambda r:r['energy']) for _,v in sorted(grouped.items())]
    if all_best:
        with (OUT/'best_available_points.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(all_best[0]));w.writeheader();w.writerows(all_best)
    platform=[r for r in refined if r['D']==4 and r['trial']=='refined_fresh1' and r['valid']]
    competing=[r for r in refined if r['D']==4 and r['trial']=='refined_fresh2' and r['valid']]
    support=''
    if platform:
        q=platform[0]
        support=f'<p style="background:#e9f5ed;padding:16px"><b>已通过收敛检查的核心结果：</b>不施加U1对称性的独立随机起源 D=4、Dz/J=0.02、h/J=0.38 态，经追加SU和χ={q["chi"]}测量，得到 m/msat={q["m"]:.12f}、E/N={q["energy"]:.12f}。这是有限D下自发形成1/9平台候选态的数值支持。</p>'
        if competing:support+=f'<p>另一独立随机起源竞争分支的磁化为 {competing[0]["m"]:.8f}，能量比该平台候选高 {competing[0]["energy"]-q["energy"]:.6g}/站点。这只比较已测到的局部极小，不能证明全局基态。</p>'
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True,constrained_layout=True)
    for i,D in enumerate((2,3,4)):
        for j,Dz in enumerate((.02,.05,.1)):
            ax=axes[i,j];ax.axhline(1/9,c='.3',ls=':',label='1/9')
            for name,color in (('python_hexagon_triangle','#c45838'),('julia_ipess','#326e9f')):
                for branch,style in (('up','o-'),('down','s--')):
                    rr=sorted([r for r in rows if r['accepted'] and r['ansatz']==name and r['D']==D and r['Dz']==Dz and r['branch']==branch],key=lambda r:r['h'])
                    if rr:ax.plot([r['h'] for r in rr],[r['m'] for r in rr],style,c=color,label=f"{'Python' if name.startswith('python') else 'Julia'} {branch}",ms=4)
            cc=[c for c in centers if c['D']==D and c['Dz']==Dz and c['valid']]
            if cc:ax.scatter([c['h'] for c in cc],[c['m'] for c in cc],marker='*',s=75,c='#63379f',label='Julia fresh center',zorder=5)
            dd=[c for c in refined if c['D']==D and c['Dz']==Dz and c['valid']]
            if dd:ax.scatter([c['h'] for c in dd],[c['m'] for c in dd],marker='D',s=30,c='#1c8f76',label='Julia continued center',zorder=6)
            ax.set_title(f'D={D}, Dz/J={Dz}');ax.set_xlim(.27,.49);ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('h/J')
            if j==0:ax.set_ylabel('m / m_sat = 2 <Sz>')
            ax.legend(fontsize=6)
    fig.savefig(OUT/'magnetization_progress.png',dpi=160);plt.close(fig)
    recent=rows[-20:];heads=('ansatz','D','Dz','h','branch','energy','m','accepted')
    table='<tr>'+''.join(f'<th>{k}</th>' for k in heads)+'</tr>'
    for r in recent:table+='<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in heads)+'</tr>'
    body=f'''<!doctype html><meta charset="utf-8"><title>非零 DMI 的 1/9 平台探索</title><style>body{{font:16px system-ui;margin:30px;line-height:1.6;max-width:1500px}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:6px}}</style>
<h1>非零 DMI：dense SU 的 1/9 磁化平台探索</h1>
{support}
<p>J=1，Dz=0.02、0.05、0.1，D=2、3、4。两套计算均不施加 U1 或空间 C3 约束。独立随机低场/高场端点分别向上/向下扫场。物理 Hamiltonian 统一为 JΣSS+DzΣκ−hΣSz；Julia 输入参数为 −h，图中磁化统一为 +2〈Sz〉。</p>
<p>已保存 {len(rows)}/252 个点，{sum(r['accepted'] for r in rows)} 个通过所记录的观测量与 CTM 检查。图随任务进度更新。部分原先因 1e-8 虚部阈值被拒绝的点使用更严格 CTM 判据、χ=12/16 复测；只有最大虚部和 face 差小于 1e-6、χ 间能量/磁化差小于 1e-5 时才纳入。CSV 的 measurement_basis 和残差列明确区分这类复测与原始 1e-8 检查，原始拒绝记录不覆盖。无数据、未通过检查的点不画入曲线。</p>
<p>先比较同一 ansatz 两分支能量，再判断较低能分支的磁化曲线；comparison_complete=false 表示缺少一条可信分支，不能因此宣布当前分支就是更低能态。能量差小于收缩精度时不能确定两者次序。单点接近 1/9 不代表平台，正反扫场相同也不能独自排除困在同一分支。</p>
<p>新启动的独立随机中心场检查已保存 {len(centers)}/8 个点，图中用紫色星号标记；<a href="fresh_center_checks.csv">中心场检查结果</a>同时比较前次扫场的最低能量和横向磁矩。原始态另保存在 Julia 的 DMI_plateau_fresh_centers_20261114 目录。</p>
<p>另有 {len(refined)}/3 个独立随机起源候选态的后续收敛检查：增加虚时间34，使用χ20/24、CTMRG最多50轮；通过者在图中用绿色菱形标记。<a href="continued_candidate_checks.csv">后续收敛结果</a>单独保存，未收敛的中心态不用于确认平台。</p>
<p>鲜态沿用五阶段 dt=(.03,.01,.003,.001,.0005)、tau=(40,20,5,2,1)，约9000步；后续场点约2066步。CPU线程10，chi=12，CTM上限 Python100/Julia50，tol=1e-6。磁化平台候选还需要更长演化、独立随机中心场初态和更大chi复核。</p>
<p><a href="all_points.csv">全部场点</a> · <a href="best_points.csv">仅比较上/下扫的较低能分支</a> · <a href="best_available_points.csv">连同独立随机中心态一起比较的已知最低能点</a> · <a href="site_magnetization.csv">逐站点磁化（含质量标记）</a></p><p><a href="D4_Dz002_platform_evidence.png">D=4、Dz=0.02 的平台与竞争分支对比图</a></p><img src="magnetization_progress.png"><h2>部分已保存数据</h2><table>{table}</table>'''
    (OUT/'report.html').write_text(body,encoding='utf-8')
    if not quiet:print(f'{len(rows)}/252 points, {sum(r["accepted"] for r in rows)} accepted; {OUT/"report.html"}')


if __name__=='__main__':run()
