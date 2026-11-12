"""Plots of THIS experiment only. No reference data or pattern matching."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from ..geometry import A1, A2, REPS, points, SHIFTS

PERIOD = np.column_stack((3*A1, 3*A2))


def key(p):
    return tuple(np.round(np.linalg.solve(PERIOD, p) % 1, 7) % 1)


def draw_pattern(obs, path):
    ps = np.array([[r['cartesian_x'],r['cartesian_y']] for r in obs['sites']])
    sz = np.array([r['sz'] for r in obs['sites']])
    coeff = np.linalg.solve(PERIOD,ps.T).T % 1
    coeff[np.isclose(coeff,1,atol=1e-9)] = 0
    pp = coeff@PERIOD.T
    lookup = {key(p):i for i,p in enumerate(ps)}
    energy = {}
    for r in obs['bonds']:
        a = points('A', *REPS['A'][r['a']])[r['i']]
        if r['kind'] == 'internal':
            b = points('A', *REPS['A'][r['a']])[r['j']]
        else:
            n,m = REPS['A'][r['a']]
            sn,sm = SHIFTS[r['k']]
            b = points('B', n+sn, m+sm)[r['j']]
        energy[tuple(sorted((lookup[key(a)],lookup[key(b)])))] = r['energy']
    assert len(energy) == 54
    fig, ax = plt.subplots(figsize=(8,6.5))
    for i,p in enumerate(pp):
        for j,q0 in enumerate(pp):
            for x in range(-1,2):
                for y in range(-1,2):
                    q = q0+x*PERIOD[:,0]+y*PERIOD[:,1]
                    if abs(np.linalg.norm(p-q)-.5) < 1e-7:
                        e = energy[tuple(sorted((i,j)))]
                        ax.plot([p[0],q[0]],[p[1],q[1]],color='#52606c',lw=.5+4*abs(e),alpha=.8,zorder=1)
    ax.scatter(pp[:,0],pp[:,1],c=sz,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-.5,vcenter=0,vmax=.5),
               s=60+950*np.abs(sz),edgecolors='#28313a',linewidths=.6,zorder=3)
    for p,v in zip(pp,sz):
        ax.annotate(f'{v:.4f}',p,xytext=(0,10),textcoords='offset points',ha='center',fontsize=7)
    corners = np.array([[0,0],PERIOD[:,0],PERIOD.sum(axis=1),PERIOD[:,1],[0,0]])
    ax.plot(corners[:,0],corners[:,1],'--',color='#8f9da7',lw=.8)
    ax.set_title(f"D={obs['D']}, h={obs['field']}, chi={obs['chi']}\n"
                 f"Eex={obs['energy_exchange']:.8f}, m/ms={obs['magnetization_normalized']:.8f}")
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def make_report(out):
    out = Path(out)
    observations = []
    for path in sorted(out.glob('D*/completed.json'),key=lambda p:int(p.parent.name[1:])):
        obs = json.loads((path.parent/'observables.json').read_text())
        if not (path.parent/'pattern.png').exists():
            draw_pattern(obs,path.parent/'pattern.png')
        observations.append(obs)
    if not observations:
        return
    ds = [r['D'] for r in observations]
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    axes[0].plot(ds,[r['energy_exchange'] for r in observations],'o-',label='Exchange energy')
    axes[0].plot(ds,[r['energy_total'] for r in observations],'s-',label='Total energy')
    axes[0].set(xlabel='D',ylabel='Energy per physical spin')
    axes[0].legend(fontsize=8)
    axes[1].plot(ds,[r['magnetization_normalized'] for r in observations],'o-',label='m/ms')
    axes[1].plot(ds,[r['max_abs_sx'] for r in observations],'s--',label='max abs Sx')
    axes[1].set(xlabel='D',ylabel='Magnetization')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'energy_magnetization.png',dpi=180)
    plt.close(fig)
    rows = ''.join(f'<tr><td>{r["D"]}</td><td>{r["energy_exchange"]:.9f}</td>'
                   f'<td>{r["energy_total"]:.9f}</td><td>{r["magnetization_normalized"]:.8f}</td>'
                   f'<td>{r["max_abs_sx"]:.4g}</td><td>{r["ctm_converged"]}</td></tr>' for r in observations)
    pictures = ''.join(f'<h2>D={D}</h2><img src="D{D}/pattern.png">' for D in ds)
    page = f'''<!doctype html><meta charset="utf-8"><title>Blocked iPEPS D{ds[0]}-D{ds[-1]}</title>
<style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#243443}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ccd5dd}}</style>
<h1>D{ds[0]}→D{ds[-1]} 六边形＋三自旋块 iPEPS</h1>
<p>每个新D严格补零继承前一个D的最终态。实数dense张量，施加C3旋转副本关系，磁化不固定。SU阶段不调用CTMRG；每个D结束后测量一次。</p>
<p>H=ΣS·S−hΣSz，m/ms=2 mean Sz；Eex不含磁场。SU proxy通过不代表严格物理收敛。圆圈表示局域Sz，键宽表示|〈Si·Sj〉|。这里只展示本次计算自身的数据。</p>
<p><a href="summary.txt">进度与收敛标记</a> · <a href="summary.csv">能量汇总</a> · <a href="site_magnetization.csv">27个site</a> · <a href="bond_energies.csv">54条bond</a> · <a href="triangle_observables.csv">18个三角形</a></p>
<table><tr><th>D</th><th>Eex/site</th><th>Etot/site</th><th>m/ms</th><th>max abs Sx</th><th>CTM</th></tr>'''+rows+'''</table><img src="energy_magnetization.png">'''+pictures
    (out/'report.html').write_text(page,encoding='utf-8')
