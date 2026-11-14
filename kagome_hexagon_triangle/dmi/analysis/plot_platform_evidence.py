"""Scientific figure of the strongest finite-D weak-DMI candidate."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plateau_report import OUT


def run():
    with (OUT/'all_points.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
    rr=[r for r in rr if int(r['D'])==4 and float(r['Dz'])==.02]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    ax=axes[0];ax.axhline(1/9,color='.25',ls=':',label='1/9')
    ax.axvspan(.36,.44,color='#e5f2e8',zorder=0)
    for name,color in (('python_hexagon_triangle','#c45838'),('julia_ipess','#326e9f')):
        for branch,style in (('up','o-'),('down','s--')):
            rows=sorted([r for r in rr if r['ansatz']==name and r['branch']==branch and r['accepted']=='True'],key=lambda r:float(r['h']))
            ax.plot([float(r['h']) for r in rows],[float(r['m']) for r in rows],style,color=color,label=f"{'Python' if name.startswith('python') else 'Julia'} {branch}",ms=4)
        diag=[r for r in rr if r['ansatz']==name and r['accepted']!='True' and r['m']]
        if diag:ax.scatter([float(r['h']) for r in diag],[float(r['m']) for r in diag],facecolors='none',edgecolors=color,s=65,label='Diagnostic only')
    ax.set(xlabel='h/J',ylabel='m / m_sat',title='D=4, Dz/J=0.02: dense SU, no U1 constraint')
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    jl={}
    for r in rr:
        if r['ansatz']=='julia_ipess' and r['accepted']=='True':jl.setdefault(float(r['h']),{})[r['branch']]=float(r['energy'])
    x=np.array(sorted(h for h,v in jl.items() if len(v)==2));y=np.array([(jl[h]['down']-jl[h]['up'])*1000 for h in x])
    ax=axes[1];ax.axhline(0,color='.3',lw=1);ax.plot(x,y,'o-',color='#326e9f')
    ax.fill_between(x,0,y,where=y>=0,color='#8fc89c',alpha=.4,interpolate=True)
    ax.fill_between(x,0,y,where=y<0,color='#e7b28b',alpha=.4,interpolate=True)
    ax.set(xlabel='h/J',ylabel='(E_down - E_up) x 1000 per spin',title='Julia: two tested continuation branches')
    ax.text(.5,.93,'Positive: near-1/9 branch has lower energy',ha='center',transform=ax.transAxes,fontsize=8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    ax.grid(alpha=.2)
    fig.savefig(OUT/'D4_Dz002_platform_evidence.png',dpi=180)
    fig.savefig(OUT/'D4_Dz002_platform_evidence.pdf');plt.close(fig)
    print(OUT/'D4_Dz002_platform_evidence.png')


if __name__=='__main__':run()
