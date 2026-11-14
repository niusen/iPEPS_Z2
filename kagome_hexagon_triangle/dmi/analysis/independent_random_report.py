"""Analysis only; each plotted point comes from its own random initialization."""
from pathlib import Path
import csv,json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..src.independent_random_fields import OPTIONS

REPO=Path(__file__).resolve().parents[3]
PY=Path(OPTIONS['out'])
JL=Path('D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/Kagome/simple_update/multicell/results/DMI_independent_random3_D2_D4_20261114')
OUT=REPO/'kagome_DMI_independent_random_review_20261114'


def best_by_energy(rows):
    """Select one valid seed at each physical point; never average branches."""
    best={}
    for r in rows:
        if not r['accepted'] or r['energy'] is None or r['m'] is None:continue
        if not math.isfinite(r['energy']) or not math.isfinite(r['m']):continue
        key=(r['ansatz'],r['D'],r['Dz'],r['h'])
        if key not in best or (r['energy'],r['trial'])<(best[key]['energy'],best[key]['trial']):best[key]=r
    return sorted(best.values(),key=lambda r:(r['ansatz'],r['D'],r['Dz'],r['h']))


def run(quiet=False,python_root=None,julia_root=None,out=None):
    target=Path(out) if out is not None else OUT
    target.mkdir(parents=True,exist_ok=True);rows=[];sites=[]
    pyroot=Path(python_root) if python_root is not None else PY
    jlroot=Path(julia_root) if julia_root is not None else JL
    sources=[('python_hexagon_triangle',pyroot)]
    if julia_root is not None or python_root is None:sources.append(('julia_ipess',jlroot))
    plans=[];expected=0
    for _,root in sources:
        plan=root/'plan.json'
        if plan.exists():
            pp=json.loads(plan.read_text());plans.extend(pp);expected+=len(pp)
        else:expected+=math.prod(len(OPTIONS[k]) for k in ('dims','Dz','fields','trials'))
    for method,root in sources:
        py=method.startswith('python')
        for p in sorted(root.glob('D*_Dz*_*/*/independent_result.json')):
            r=json.loads(p.read_text());ctm=r.get('ctm_converged',False) if py else r.get('ctm',{}).get('converged',False)
            row=dict(ansatz=method,D=r['D'],Dz=r['Dz'],h=r['h_physical'],trial=r['trial'],seed=r['seed'],
                energy=r.get('energy_total' if py else 'energy_per_spin'),m=r.get('m_normalized'),
                accepted=bool(r['valid_measurement'] and ctm),ctm_converged=ctm,chi=r.get('chi'),
                warm_start=r['warm_start'],measurement_basis=r.get('measurement_basis','original strict measurement'),
                max_imaginary=r.get('max_hermitian_imaginary_part'),source=str(p))
            assert not row['warm_start'];rows.append(row)
            ss=[dict(site=f"{s['kind']}_{s['colour']}_{s['site']}",Sx=s['sx'],Sy=s['sy'],Sz=s['sz']) for s in r.get('sites',[])] if py else [dict(site=f"{c['x']}_{c['y']}_{b+1}",Sx=c['Sx'][b],Sy=c['Sy'][b],Sz=c['Sz'][b]) for c in r.get('cells',[]) for b in range(3)]
            sites.extend(dict(ansatz=method,D=row['D'],Dz=row['Dz'],h=row['h'],trial=row['trial'],seed=row['seed'],accepted=row['accepted'],**s) for s in ss)
    best=best_by_energy(rows)
    for name,data in (('all_points',rows),('best_points',best),('site_magnetization',sites)):
        if data:
            with (target/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    old=REPO/'kagome_DMI_plateau_review_20261113/best_available_points.csv'
    comparison=[]
    if old.exists():
        with old.open(encoding='utf-8-sig',newline='') as f:reference={(r['ansatz'],int(r['D']),float(r['Dz']),float(r['h'])):r for r in csv.DictReader(f)}
        for r in rows:
            prev=reference.get((r['ansatz'],r['D'],r['Dz'],r['h']))
            if r['accepted'] and prev:
                comparison.append(r|dict(previous_energy=float(prev['energy']),previous_m=float(prev['m']),delta_energy=r['energy']-float(prev['energy']),reference_use='energy comparison only; never an initialization'))
    if comparison:
        with (target/'energy_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    dims=sorted({p['D'] for p in plans}) if plans else OPTIONS['dims']
    dzs=sorted({p['Dz'] for p in plans}) if plans else OPTIONS['Dz']
    fields=sorted({p['h'] for p in plans}) if plans else OPTIONS['fields']
    fig,axes=plt.subplots(len(dims),len(dzs),figsize=(4*len(dzs),3*len(dims)),squeeze=False,sharex=True,constrained_layout=True)
    for i,D in enumerate(dims):
        for j,Dz in enumerate(dzs):
            ax=axes[i,j];ax.axhline(1/9,color='.35',ls=':',label='1/9')
            for method,color,marker in (('python_hexagon_triangle','#c45838','o'),('julia_ipess','#326e9f','s')):
                trials=[r for r in rows if r['ansatz']==method and r['D']==D and r['Dz']==Dz and r['accepted'] and r['m'] is not None]
                if trials:ax.scatter([r['h'] for r in trials],[r['m'] for r in trials],color=color,s=12,alpha=.25)
                rr=[r for r in best if r['ansatz']==method and r['D']==D and r['Dz']==Dz]
                if rr:ax.plot([r['h'] for r in rr],[r['m'] for r in rr],marker+'-',color=color,ms=4,label=method+' lowest E')
            ax.set_title(f'D={D}, Dz/J={Dz}');ax.grid(alpha=.2);ax.set_xlim(min(fields)-.01,max(fields)+.01)
            if i==len(dims)-1:ax.set_xlabel('h/J')
            if j==0:ax.set_ylabel('m / m_sat')
            ax.legend(fontsize=6)
    fig.suptitle('Independent random dense initial state at EVERY field; no U1 or spatial C3 projection')
    fig.savefig(target/'magnetization.png',dpi=160);plt.close(fig)
    text=f'''<!doctype html><meta charset="utf-8"><title>逐场点独立随机初始化</title><style>body{{font:16px system-ui;max-width:1400px;margin:30px;line-height:1.7}}img{{max-width:100%}}</style>
<h1>非零 DMI：每个磁场完全独立随机初始化</h1>
<p>J=1；D={dims}；Dz/J={dzs}；h/J={fields}。各场点的试验次数与随机seed见计划文件。每一个(D,Dz,h,trial)均从新的复高斯dense张量开始，不继承任何相邻场、相邻D或旧平台态。仅允许恢复同一场点自身的断点。</p>
<p>已保存 {len(rows)}/{expected} 个场点，{sum(r['accepted'] for r in rows)} 个通过所记录的收敛/精度检查。初始张量、实际seed、初态核对记录和逐阶段checkpoint均保存。</p>
<p>每点采用完整五阶段、约9000步SU；CPU线程10，无U1或纯空间C3投影，CTM最多Python100/Julia50轮。图中仅绘制通过检查的场点，数据空缺不能被解释为没有平台。部分Python点采用明确标注的双χ复测精度标准。</p>
<p>浅色点为各seed结果，连线为同一磁场通过检查的最低能量seed，不对不同分支的磁化取平均。尚未完成或未通过检查的seed仍可能改变筛选结果。有限个随机初态不能保证全局最低能态；若比旧平台态能量更高，仍应保留旧低能态作为竞争候选。平台边界附近应追加seed和加密磁场。</p>
<p><a href="all_points.csv">所有独立随机场点与seed</a> · <a href="best_points.csv">每点最低能量seed</a> · <a href="site_magnetization.csv">逐站点磁化</a> · <a href="energy_comparison.csv">与前一轮已知低能态比较（仅分析，未用于初始化）</a></p><img src="magnetization.png">'''
    (target/'report.html').write_text(text,encoding='utf-8')
    if not quiet:print(f'{len(rows)}/{expected} points: {target/"report.html"}')


if __name__=='__main__':run()
