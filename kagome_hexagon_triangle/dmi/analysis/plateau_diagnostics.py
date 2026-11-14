"""Finite-D branch diagnostics; crossing estimates are NOT phase boundaries."""
import csv,json
from collections import defaultdict
import numpy as np
from .plateau_report import OUT,run


def analyze():
    run(quiet=True)
    with (OUT/'all_points.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows:
        if r['accepted']!='True':continue
        for k in ('D','Dz','h','energy','m'):r[k]=float(r[k])
        groups[r['ansatz'],int(r['D']),r['Dz']].append(r)
    result=[]
    for (name,D,dz),rr in sorted(groups.items()):
        by_h=defaultdict(dict)
        for r in rr:by_h[r['h']][r['branch']]=r
        paired=sorted((h,v['up']['energy']-v['down']['energy']) for h,v in by_h.items() if len(v)==2)
        crossings=[]
        for (a,da),(b,db) in zip(paired,paired[1:]):
            if da*db<0:crossings.append(dict(bracket=[a,b],linear_estimate=a-da*(b-a)/(db-da)))
        lines=[]
        for branch in ('up','down'):
            near=sorted([r for r in rr if r['branch']==branch and abs(r['m']-1/9)<5e-4],key=lambda r:r['h'])
            if len(near)<3:continue
            x=np.array([r['h'] for r in near]);y=np.array([r['energy'] for r in near]);fit=np.polyfit(x,y,1)
            lines.append(dict(branch=branch,fields=x.tolist(),m_min=min(r['m'] for r in near),m_max=max(r['m'] for r in near),
                energy_slope=float(fit[0]),expected_slope=-1/18,max_linear_residual=float(max(abs(y-np.polyval(fit,x)))),
                warning='Near-1/9 continuation can be metastable; compare energies and fresh starts.'))
        result.append(dict(ansatz=name,D=D,Dz=dz,paired_points=len(paired),two_branch_crossing_estimates=crossings,near_plateau_line_fits=lines))
    (OUT/'branch_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for v in result:
        if v['Dz']==.02 and v['D']==4:print(json.dumps(v,indent=2))


if __name__=='__main__':analyze()
