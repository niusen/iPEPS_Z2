"""Plot the earlier independent D5/D6 states against author local-Sz data.

This only reads saved observables. No tensor evolution or CTMRG is run.
All lattice bonds are uniform guides; author per-bond values are unavailable.
"""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .analyze import ROOT, PERIOD, compare

OUT = ROOT/'results'/'local_D5_D6_paper_comparison'
FILES = [ROOT/'results'/'D5_continuation_h038'/'D5_seed1701'/'verified_observables_chi24.json',
         ROOT/'results'/'D6_continuation_h038'/'D6_seed1701'/'verified_observables_chi32.json']
NORM = TwoSlopeNorm(vmin=-.35, vcenter=0., vmax=.35)


def panel(ax, ps, sz, title, subtitle):
    coeff = np.linalg.solve(PERIOD, ps.T).T % 1
    coeff[np.isclose(coeff, 1, atol=1e-9)] = 0
    shown = coeff@PERIOD.T
    # Draw every visible nearest-neighbour segment once. Bond width is
    # intentionally uniform across measured and reference panels.
    segments = set()
    for p in shown:
        for q0 in shown:
            for x in range(-1,2):
                for y in range(-1,2):
                    q = q0+x*PERIOD[:,0]+y*PERIOD[:,1]
                    if abs(np.linalg.norm(p-q)-.5) > 1e-8:
                        continue
                    key = tuple(sorted((tuple(np.round(p,8)),tuple(np.round(q,8)))))
                    if key in segments:
                        continue
                    segments.add(key)
                    ax.plot([p[0],q[0]],[p[1],q[1]],color='#9aa9b6',lw=1.05,zorder=1)
    corners = np.array([[0,0],PERIOD[:,0],PERIOD.sum(axis=1),PERIOD[:,1],[0,0]])
    ax.plot(corners[:,0],corners[:,1],'--',color='#65798d',lw=.9,zorder=1)
    artist = ax.scatter(shown[:,0],shown[:,1],c=sz,cmap='coolwarm',norm=NORM,
                        s=65+1050*np.abs(sz),edgecolors='#344452',linewidths=.7,zorder=3)
    for p,v in zip(shown,sz):
        ax.annotate(f'{v:+.3f}',p,xytext=(0,13),textcoords='offset points',
                    ha='center',va='bottom',fontsize=8,color='#243443',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=.3),zorder=4)
    ax.set_title(title+'\n'+subtitle,fontsize=11.5,pad=13,linespacing=1.6)
    ax.set_aspect('equal')
    ax.set_xlim(-.8,4.3)
    ax.set_ylim(-.65,2.98)
    ax.axis('off')
    return artist


def figure(data, include_D7=False):
    ncols = 3 if include_D7 else 2
    fig,axes = plt.subplots(2,ncols,figsize=(6.1*ncols,10.5))
    fig.subplots_adjust(left=.025,right=.92,bottom=.065,top=.91,wspace=.02,hspace=.27)
    for row,item in enumerate(data):
        obs,ps,sz,refs = item
        artist = panel(axes[row,0],ps,sz,
                       f"Earlier local D={obs['D']}, h=0.38, chi={obs['chi']}",
                       f"m/ms={obs['magnetization_normalized']:.7f}")
        reference_Ds = (7,15) if include_D7 else (15,)
        for col,D in enumerate(reference_Ds,1):
            vals,stats = refs[D]
            panel(axes[row,col],ps,vals,f'Paper D={D}, h=0.4; spatially aligned',
                  f"Correlation={stats['correlation']:.4f}; Sz RMS={stats['rms']:.4f}")
    cax = fig.add_axes([.943,.21,.014,.52])
    bar = fig.colorbar(artist,cax=cax)
    bar.set_label(r'Local $\langle S^z\rangle$ (not $m/m_s$)',fontsize=11)
    fig.suptitle('Earlier independently initialized D5/D6 states versus Ghosh-Mila paper',
                 fontsize=15,y=.985)
    fig.text(.46,.025,'Shared colour and marker-size scales. Uniform bonds show geometry only.\n'
             'Reference panels allow only a single global translation / rotation / reflection.',
             ha='center',fontsize=10,color='#475569',linespacing=1.6)
    stem = 'D5_D6_vs_paper_D7_D15' if include_D7 else 'D5_D6_vs_paper_D15'
    fig.savefig(OUT/(stem+'.png'),dpi=200,facecolor='white',bbox_inches='tight',pad_inches=.15)
    if not include_D7:
        fig.savefig(OUT/(stem+'.svg'),facecolor='white',bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,rows = [],[]
    for path in FILES:
        obs = json.loads(path.read_text())
        refs = {}
        for D in (7,15):
            ps,sz,aligned,raw = compare(obs,D)
            stats = dict(D=obs['D'],chi=obs['chi'],reference_D=D,
                         correlation=raw['paper_D15_best_spatial_correlation'],
                         rms=raw['paper_D15_best_spatial_rms_sz'],
                         sorted_rms=raw['paper_D15_sorted_rms_sz'],
                         alignment=raw['reference_alignment'],
                         source=str(path.relative_to(ROOT)),
                         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            refs[D] = (aligned,stats)
            rows.append(stats)
        data.append((obs,ps,sz,refs))
    figure(data)
    figure(data,include_D7=True)
    (OUT/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    table = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                    (r['D'],r['chi'],r['reference_D'],f"{r['correlation']:.5f}",f"{r['rms']:.5f}"))
                    +'</tr>' for r in rows)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>之前本机D5、D6与论文图案比较</title>
<style>body{font:16px/1.7 system-ui;color:#243443;max-width:1250px;margin:32px auto;padding:0 22px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ccd5df;text-align:right}</style>
<h1>之前本机独立初始化的 D=5、6：局域磁化图案</h1>
<p>这里使用之前本机D5、D6各自独立随机初始化并继续演化后的结果，不是服务器D4→D10继承链。D5取χ=24复测，D6取χ=32复测。本次仅绘图，没有重新进行张量计算。</p>
<p>左列为实际结果；右列为论文D15局域磁化，经整体平移、旋转和反射选出最佳对齐。两个右图可能采用不同的整体取向，但都来自同一份论文D15数据；没有单独交换site。颜色和圆圈大小统一表示带符号的〈Sz〉，不是归一化磁化m/ms。所有bond均为等宽几何辅助线，不表示bond能量。</p>
<img src="D5_D6_vs_paper_D15.png">
<p>空间相关系数较高，支持“图案比较像”的描述，但逐点磁化仍有约0.049的RMS差异。本机h=0.38，论文图案h=0.4；没有作者D5、D6的逐site数据，因此不属于同D对照。D5仍有max|Sx|≈0.0231，D6约0.00763；这些初步态还不能认定为完全收敛的严格平台态。</p>
<table><tr><th>本机D</th><th>χ</th><th>论文D</th><th>空间相关系数</th><th>Sz RMS</th></tr>'''+table+'''</table>
<h2>补充：最接近的公开D=7图案与D=15图案</h2>
<p>每个参考panel分别选取最佳整体空间对齐。论文没有公开D5、D6的局域磁化表。</p>
<img src="D5_D6_vs_paper_D7_D15.png">
<p><a href="D5_D6_vs_paper_D15.svg">矢量图SVG</a> · <a href="comparison.json">对齐参数和指标</a> · <a href="https://doi.org/10.6084/m9.figshare.34731234.v3">作者原始数据</a> · <a href="https://arxiv.org/html/2610.34439v1">论文</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(rows,indent=2))
    print(OUT/'report.html')


if __name__ == '__main__':
    main()
