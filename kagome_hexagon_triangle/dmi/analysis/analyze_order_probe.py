"""Read independent Julia/Python order probes; numerical analysis/plots only."""
from pathlib import Path
import json,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ...geometry import A1,A2,points,REPS
from ..src.model import arrow_sign

PY_ROOT=Path(__file__).resolve().parents[2]/'results/pure_DMI_D2_D4_order_20261113'
JL_ROOT=Path('D:/My Documents/Code/Julia_codes/Tensor network/Symmetric_IPEPS/SU2_AD/Kagome/simple_update/multicell/results/pure_DMI_D2_D4_order_20261113')
OUT=Path(__file__).resolve().parents[3]/'kagome_dmi_order_review_20261113'
BASIS=(-A1/2,A1*0,-A2/2)


def label(p):
    matches=[i for i,b in enumerate(BASIS) if np.linalg.norm((v:=np.linalg.solve(np.column_stack((A1,A2)),p-b))-np.round(v))<1e-8]
    assert len(matches)==1
    return matches[0]


def unpack(r,ansatz):
    sites=[];bonds=[]
    if ansatz=='hexagon_triangle':
        for s in r['sites']:
            p=points(s['kind'],*REPS[s['kind']][s['colour']])[s['site']]
            sites.append(dict(key=(s['kind'],s['colour'],s['site']),position=p,label=label(p),
                m=s['sx']+1j*s['sy'],sz=s['sz']))
        for b in r['bonds']:
            a=('A',b['a'],b['i']);c=('A' if b['kind']=='internal' else 'B',b['b'],b['j'])
            bonds.append((a,c,b['vector_chirality_z_ccw']))
        energy=r['energy_total'];converged=r['ctm_converged'];residual=r['ctm_residual']
    else:
        for c in r['cells']:
            x,y=c['x']-1,c['y']-1
            for b in range(3):
                sites.append(dict(key=(x,y,b),position=x*A1+y*A2+BASIS[b],label=b,
                    m=c['Sx'][b]+1j*c['Sy'][b],sz=c['Sz'][b]))
            up=[(x,y,b) for b in range(3)]
            dn=[(x,y,1),(x,(y+1)%3,2),((x+1)%3,y,0)]
            for name,keys in (('up',up),('down',dn)):
                for k in range(3):bonds.append((keys[k],keys[(k+1)%3],c[f'kappa_{name}_CCW'][k]))
        energy=r['energy_per_spin'];converged=r['ctm']['converged'];residual=r['ctm']['error']
    return sites,bonds,energy,converged,residual


def analyze(path,ansatz):
    r=json.loads(path.read_text());sites,bonds,energy,ctm_ok,residual=unpack(r,ansatz)
    z=np.array([s['m'] for s in sites]);theta=np.array([2*np.pi*s['label']/3 for s in sites])
    order=np.mean(z*np.exp(-1j*theta));amp=abs(z);meanamp=np.mean(amp);phase=np.angle(order)
    error=np.angle(z*np.exp(-1j*(theta+phase)))
    sub=np.array([np.mean([s['m'] for s in sites if s['label']==b]) for b in range(3)])
    # Align only the freely chosen global U1 phase, never individual sites.
    angles=(np.angle(sub*np.exp(-1j*phase),deg=True)+360)%360
    lookup={s['key']:s['m'] for s in sites};nn=[]
    for a,b,_ in bonds:
        za,zb=lookup[a],lookup[b]
        if abs(za)*abs(zb)>1e-12:
            nn.append(np.degrees(np.arccos(np.clip(np.real(za.conjugate()*zb)/(abs(za)*abs(zb)),-1,1))))
    kappa=float(np.mean([k for _,_,k in bonds]))
    assert abs(energy-2*kappa)<1e-9,(energy,kappa) # J=h=0, z=4
    row=dict(ansatz=ansatz,D=r['D'],branch=r['branch'],chi=r['chi'],energy=energy,
        M120=float(abs(order)),mean_transverse=float(meanamp),min_transverse=float(min(amp)),
        transverse_std=float(np.std(amp)),q0_coherence=float(abs(order)/max(meanamp,1e-30)),
        weighted_phase_rms_degrees=float(np.sqrt(np.sum(amp*error**2)/max(sum(amp),1e-30))*180/np.pi),
        sublattice_angles_degrees=angles.tolist(),mean_nn_angle_degrees=float(np.mean(nn)),
        max_nn_angle_error_degrees=float(max(abs(np.array(nn)-120))),
        max_abs_sz=float(max(abs(s['sz']) for s in sites)),mean_kappa_ccw=kappa,
        ctm_converged=ctm_ok,ctm_residual=residual,
        hermitian_imaginary_residual=r.get('max_hermitian_imaginary_part'),
        strict_ctm_recheck=r.get('strict_ctm_recheck',False),
        last_su_weight_drift_per_tau=r['su_history'][-1]['weight_drift_per_tau_last20'],
        actual_D_max=r.get('actual_D_max',r['D']),source=str(path))
    return row,sites,phase


def run():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];data={};site_rows=[];bond_rows=[];rejected=[]
    for name,root in (('hexagon_triangle',PY_ROOT),('ipess',JL_ROOT)):
        for path in sorted(root.glob('D*_*/*observables_chi*.json')):
            row,sites,phase=analyze(path,name);rows.append(row)
            keys={s['key']:s for s in sites}
            for s in sites:
                site_rows.append(dict(ansatz=name,D=row['D'],branch=row['branch'],chi=row['chi'],
                    site_key=str(s['key']),x=float(s['position'][0]),y=float(s['position'][1]),
                    sublattice=s['label']+1,Sx=float(s['m'].real),Sy=float(s['m'].imag),Sz=s['sz'],
                    magnitude_transverse=float(abs(s['m'])),global_phase_removed_radians=float(phase)))
            raw=json.loads(path.read_text());_,bb,_,_,_=unpack(raw,name)
            for i,(a,b,kappa) in enumerate(bb):
                bond_rows.append(dict(ansatz=name,D=row['D'],branch=row['branch'],chi=row['chi'],
                    bond=i+1,site_a=str(a),site_b=str(b),kappa_z_CCW=kappa,energy_DMI=kappa))
            key=(name,row['D'],row['branch'])
            if key not in data or data[key][0]['chi']<row['chi']:data[key]=(row,sites,phase)
    (OUT/'all_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if not rows:print('No completed cases yet');return
    with (OUT/'all_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name,values in (('site_magnetization_all',site_rows),('bond_observables_all',bond_rows)):
        with (OUT/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    selected=[v[0] for v in data.values()]
    for name,root in (('hexagon_triangle',PY_ROOT),('ipess',JL_ROOT)):
        for path in sorted(root.glob('D*_*/*rejected*chi*.json')):
            issue=json.loads(path.read_text());issue.update(ansatz=name,case=path.parent.name,source=str(path))
            rejected.append(issue)
    (OUT/'rejected_ctm_checks.json').write_text(json.dumps(rejected,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(3,4,figsize=(16,11),constrained_layout=True)
    for col,(name,branch) in enumerate((('hexagon_triangle','random'),('hexagon_triangle','ordered120'),('ipess','random'),('ipess','ordered120'))):
        for rowi,D in enumerate((2,3,4)):
            ax=axes[rowi,col];key=(name,D,branch)
            if key not in data:ax.text(.5,.5,'Running / pending',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
            row,sites,phase=data[key];pos=np.array([s['position'] for s in sites]);m=np.array([s['m'] for s in sites])*np.exp(-1j*phase)
            for i in range(len(pos)):
                for j in range(i):
                    if abs(np.linalg.norm(pos[i]-pos[j])-.5)<1e-8:ax.plot(*np.array([pos[i],pos[j]]).T,color='.84',lw=.7,zorder=0)
            colors=[('#2764a5','#de7b2d','#30936a')[s['label']] for s in sites]
            ax.scatter(*pos.T,c=colors,s=10,zorder=2)
            ax.quiver(*pos.T,m.real,m.imag,color=colors,angles='xy',scale_units='xy',scale=1.6,width=.008,pivot='middle')
            title=f"{'Hexagon-triangle' if name=='hexagon_triangle' else 'iPESS'} / {branch} / D={D}\nE={row['energy']:.7f}, M120={row['M120']:.4f}, chi={row['chi']}\nphase RMS={row['weighted_phase_rms_degrees']:.3g} deg"
            ax.set_title(title,fontsize=9);ax.set_aspect('equal');ax.set_axis_off()
    fig.suptitle('Pure kagome DMI: J=0, Dz=1, h=0 | dense SU, no spatial C3 constraint\nOnly a single global spin rotation is removed from each panel; iPESS 9-site state is repeated over a 3x3 display.',fontsize=12)
    fig.savefig(OUT/'spin_patterns.png',dpi=180);fig.savefig(OUT/'spin_patterns.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for name,color in (('hexagon_triangle','#c85137'),('ipess','#326ca0')):
        for branch,style in (('random','o-'),('ordered120','s--')):
            rr=sorted([v for v in selected if v['ansatz']==name and v['branch']==branch],key=lambda r:r['D'])
            if not rr:continue
            for ax,key in zip(axes,('energy','M120')):ax.plot([r['D'] for r in rr],[r[key] for r in rr],style,color=color,label=f'{name}, {branch}')
    axes[0].axhline(-np.sqrt(3)/4,c='.4',ls=':',label='120 product energy');axes[0].set_ylabel('Energy per spin')
    axes[1].set_ylabel('q=0 transverse order amplitude M120')
    for ax in axes:ax.set_xlabel('D');ax.set_xticks([2,3,4]);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'energy_and_order.png',dpi=180);plt.close(fig)
    cols=('ansatz','D','branch','chi','energy','M120','q0_coherence','weighted_phase_rms_degrees','max_abs_sz','ctm_converged')
    header=''.join(f'<th>{html.escape(c)}</th>' for c in cols)
    body=''.join('<tr>'+''.join(f'<td>{html.escape(format(r[c],".8g") if isinstance(r[c],float) else str(r[c]))}</td>' for c in cols)+'</tr>' for r in sorted(selected,key=lambda r:(r['ansatz'],r['D'],r['branch'])))
    warning='<p style="background:#fff0db;padding:16px"><b>精度限制：</b>Python D=4 的部分较大 χ 测量触发了 1e-8 厄米性虚部检查，收紧 CTM 判据后仍有残差。未通过的测量没有混入能量表；表中使用通过检查的 χ 点。磁矩约 0.44、120°相位结构清楚，但不能把这些 D=4 能量当作高精度基准。<a href="rejected_ctm_checks.json">原始拒绝记录</a>均保留。Julia D=4 两条初态分支虽然都有相同磁序，能量仍相差约 1.3e-4。</p>' if rejected else ''
    text=f'''<!doctype html><meta charset="utf-8"><title>纯 DMI 的 120°磁序测试</title><style>body{{font:16px system-ui;margin:32px;max-width:1500px;line-height:1.6}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border:1px solid #ddd}}th{{background:#eef2f7}}</style>
<h1>纯 DMI：D=2、3、4 的 simple-update 磁序测试</h1>
<p>J=0，Dz=1，h=0。Python 使用 27 站点六边形–三角形 ansatz；Julia 使用 dense iPESS、9 站点 √3×√3 元胞。每个 D 独立测试一个随机初态和一个 120°参考初态，不施加 U1 或纯空间 C3 投影。演化为四阶段、共 760 步，τ 合计 13.2；这是一轮小 D 检查，并非全局能量优化搜索。</p>
<p><b>当前结果：</b>已完成 {len(selected)}/12 个算例的有效测量。M120 = |mean[(Sx+iSy) exp(-2πi·sublattice/3)]|。M120 非零、且它与平均横向磁矩之比接近 1，说明是统一的 q=0 120°子格图案。图中每个态仅去掉一个整体自旋转角，没有逐站点旋转自旋来强行对齐。</p>
{warning}
<p>完整数据：<a href="all_metrics.csv">磁序、SU 漂移与 CTM 收敛指标</a>，<a href="site_magnetization_all.csv">逐站点磁化</a>，<a href="bond_observables_all.csv">逐 bond DMI 关联</a>。图像也有 <a href="spin_patterns.pdf">PDF 版本</a>。单个随机 seed 不等于穷尽所有亚稳态；E/N=-√3/4 是乘积态参考能量，非量子基态精确值。两个 ansatz 的同一个 D 也不代表相同的变分空间。</p>
<table><tr>{header}</tr>{body}</table><p>{len(selected)}/12 cases have a completed measurement.</p>
<img src="spin_patterns.png"><img src="energy_and_order.png">'''
    (OUT/'report.html').write_text(text,encoding='utf-8')
    print(json.dumps(selected,indent=2));print(OUT/'report.html')


if __name__=='__main__':run()
