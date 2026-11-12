"""Compare previous and new CSV/observable snapshots in one fixed paper frame.

This performs only NumPy analysis and plotting, not SU or tensor contraction.
"""
import hashlib
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze import ROOT, reference, site_position
from .analyze_seeded_server_results import align
from .analyze_server_results import read_csv, rms, permutation
from .plot_local_D5_D6_vs_paper import panel

OUT=ROOT/'results/cross_batch_pattern_comparison_20261109'


def main():
    OUT.mkdir(exist_ok=True)
    ps,ref7=reference(7)
    _,ref15=reference(15)
    labels=np.array([r['paper_label'] for r in json.loads((ROOT/'references/paper_D15_reference.json').read_text())['sites']])
    swaps=ref7.copy();swaps[labels==8]=ref7[labels==9].mean();swaps[labels==9]=ref7[labels==8].mean()
    records=[];raw={};hashes={}
    def add(batch,summary,sites,path):
        coords=np.array([site_position(r) for r in sites])
        sz=np.array([float(r['sz']) for r in sites])
        assert len(sites)==27
        a7=align(coords,sz,ps,ref7);a15=align(coords,sz,ps,ref15)
        D=int(summary['D']);seed=int(summary.get('seed',1701));h=float(summary['field'])
        key=f'{batch}_D{D}_seed{seed}'
        c3=min(rms(sz-sz[permutation(coords,theta=2,shift=(x,y))]) for x in range(3) for y in range(3))
        m=float(summary['magnetization_normalized'])
        assert abs(2*sz.mean()-m)<1e-10
        row=dict(key=key,batch=batch,D=D,seed=seed,h=h,chi=int(summary['chi']),
            Eex=float(summary['energy_exchange']),m_normalized=m,
            max_abs_sx=max(abs(float(r.get('sx',0.))) for r in sites),
            SU_proxy=summary.get('su_proxy_converged','not_recorded'),
            rms7=a7['rms'],correlation7=a7['correlation'],rms15=a15['rms'],correlation15=a15['correlation'],
            sorted_rms7=rms(np.sort(sz)-np.sort(ref7)),C3_site_rms=c3,
            diagnostic_forbidden_swap89_rms=rms(a7['values']-swaps),source=str(path))
        records.append(row)
        raw[key]=dict(row=row,aligned7=a7['values'],aligned15=a15['values'])
    folders=[('old_random_C3','server_D7_random10_h040_C3_screen9000'),
             ('new_physicalNN_six','server_D7_random10_h040_physicalNN_six_then_C3'),
             ('D4_chain','server_D4_D10_h038_C3_chain'),
             ('D5_seeded_chain','server_from_local_D5_D5_D10_h038_C3_chain')]
    for batch,folder in folders:
        folder=ROOT/'results'/folder
        summaries=read_csv(folder/'summary.csv');sites=read_csv(folder/'site_magnetization.csv')
        for name in ('summary.csv','site_magnetization.csv'):
            path=folder/name;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for s in summaries:
            # Earlier single-state-per-D exports omit seed/chi from site rows.
            selected=[r for r in sites if int(r['D'])==int(s['D']) and
                      ('seed' not in r or int(r['seed'])==int(s['seed'])) and
                      ('chi' not in r or int(r['chi'])==int(s['chi']))]
            add(batch,s,selected,folder/'summary.csv')
    for D,chi in ((5,24),(6,32)):
        path=ROOT/f'results/D{D}_continuation_h038/D{D}_seed1701/verified_observables_chi{chi}.json'
        data=json.loads(path.read_text());data['chi']=chi
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        add('early_local',data,data['sites'],path)
    assert len(records)==32
    result=dict(primary_reference='Author D7 h0.4',secondary_reference='Author D15 h0.4',
        method='Same 108 global lattice operations for every state; no individual site exchange',
        cautions=['Different D/h are morphology references, not equal-parameter tests.',
                  'Old runs impose C3; new physical-NN runs keep six free tensors. Not a controlled optimizer comparison.',
                  'Artificial class8/9 exchange is diagnostic only, never a valid alignment.'],
        sources_sha256=hashes,records=records)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('TOP TO PAPER D7')
    for r in sorted(records,key=lambda x:x['rms7'])[:12]:
        print(r['key'],'h',r['h'],'RMS7',r['rms7'],'corr7',r['correlation7'],
              'RMS15',r['rms15'],'corr15',r['correlation15'],'SU',r['SU_proxy'])
    print('TOP TO PAPER D15')
    for r in sorted(records,key=lambda x:x['rms15'])[:7]:
        print(r['key'],r['rms15'],r['correlation15'])
    print('BEST SAME D7 H04',min((r for r in records if r['D']==7 and r['h']==.4),key=lambda r:r['rms7']))
    print('SEEDED_D7',[r for r in records if r['batch']=='D5_seeded_chain' and r['D']==7])

    chosen=['old_random_C3_D7_seed7109','new_physicalNN_six_D7_seed7103','D5_seeded_chain_D7_seed1701']
    fig,axes=plt.subplots(2,2,figsize=(12.2,12.2))
    artist=panel(axes[0,0],ps,ref7,'Paper D=7, h=0.4','Fixed reference orientation')
    titles=['Previous C3 random batch: seed7109','Latest physical-NN batch: seed7103','D5-seeded chain: D7, seed1701']
    for ax,key,title in zip((axes[0,1],axes[1,0],axes[1,1]),chosen,titles):
        r=raw[key]['row']
        panel(ax,ps,raw[key]['aligned7'],title,
            f"h={r['h']}; Sz RMS={r['rms7']:.4f}; corr={r['correlation7']:.3f}\n"
            f"Eex={r['Eex']:.9f}; SU proxy={r['SU_proxy']}")
    fig.subplots_adjust(left=.015,right=.91,top=.86,bottom=.075,hspace=.45,wspace=.04)
    fig.colorbar(artist,cax=fig.add_axes([.94,.23,.015,.52]),label=r'Local $\langle S^z\rangle$')
    fig.suptitle('Previous versus latest patterns: same fixed paper-D7 frame',fontsize=15,y=.985)
    fig.text(.46,.025,'Only global lattice operations; shared colour and marker-size scales.\n'
        'Old runs impose C3. The new six-tensor protocol has not passed its SU proxy.',ha='center',fontsize=10)
    fig.savefig(OUT/'old_new_vs_paper_D7.png',dpi=180,bbox_inches='tight');plt.close(fig)

    columns=[('batch','批次'),('D','D'),('seed','seed'),('h','h'),('rms7','论文D7 RMS'),
        ('correlation7','D7相关系数'),('rms15','论文D15 RMS'),('correlation15','D15相关系数'),('SU_proxy','SU代理')]
    table=''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k,_ in columns)+'</tr>'
        for r in sorted(records,key=lambda r:r['rms7']))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>各批图案与论文统一对照</title>
<style>body{{font:16px/1.65 system-ui;color:#263443;max-width:1200px;margin:30px auto;padding:0 20px}}img{{max-width:100%}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ccd2d8;text-align:right}}</style>
<h1>各批结果：同一套空间对齐标准</h1>
<p>按同D=7、h=0.4及合法空间操作后的逐site Sz RMS，上一批C3随机态seed7109比新批次seed7103更接近论文。已通过SU判据的D5继承链D7也非常接近这一旧图案，但其h=0.38。二者仍有论文8/9两类位置差异。</p>
<p>较早本机D5/D6与论文D15的相关系数更高，但D和h都不同、幅值差异存在，不能混作同参数复现。旧C3受限与新六张量自由演化还改变了算法和变分约束，图案接近程度不能直接作为算法准确性的排名。</p>
<img src="old_new_vs_paper_D7.png"><div class="table"><table><tr>{''.join('<th>'+v+'</th>' for _,v in columns)}</tr>{table}</table></div>
<p>RMS衡量实际Sz数值与空间排列的综合差异；相关系数更偏重形状。不同D与场值的行只能作形态参照。未重新运行SU或CTMRG。</p>
<p><a href="comparison.json">32个态的完整指标及来源SHA256</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    assert all(hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    print(OUT/'report.html')


if __name__=='__main__':main()
