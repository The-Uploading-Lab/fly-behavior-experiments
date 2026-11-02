"""Generate the two review-draft figures from the public numeric records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyse import analyse
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'outputs/publication/figures'
OUT.mkdir(parents=True,exist_ok=True)
result=analyse()
x=json.loads((ROOT/'data/muscle-configuration-rows.json').read_text())
by={(r['who'],r['seed']):r for r in x['results']}
seeds=range(46,76)
def standing(r): return r['upright']>.90 and r['z_end']>1.0
a=np.array([standing(by['champ',s]) for s in seeds],int)
b=np.array([standing(by['cand5',s]) for s in seeds],int)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(6.4,2.7),layout='constrained')
axs[0].bar([0,1],[a.mean(),b.mean()],color=['#6e6e6e','#28647d'],width=.55)
axs[0].set(xticks=[0,1],xticklabels=['Predecessor','Candidate'],ylim=(0,1),ylabel='Fraction standing')
for i,n in enumerate([a.sum(),b.sum()]): axs[0].text(i,n/30+.025,f'{n}/30',ha='center')
axs[0].set_title('A  Archived configuration comparison',loc='left',fontsize=9)
rows=json.loads((ROOT/'data/delay-replay-rows.json').read_text())['rows']
arms=['base','flat6','perm']
for seed in range(201303,201309):
    vals=[next(r['record']['speed'] for r in rows if r['arm']==arm and r['seed']==seed) for arm in arms]
    axs[1].plot(range(3),vals,'o-',lw=.7,ms=3,alpha=.65)
axs[1].set(xticks=range(3),xticklabels=['Original','Uniform','Shuffled'],ylabel='Simulated speed (mm/s)',ylim=(0,None))
axs[1].set_title('B  Replayed delay comparison',loc='left',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'fig1-configuration-comparisons.{ext}',dpi=240)
plt.close(fig)
cal=json.loads((ROOT/'data/calcium-and-walking-summaries.json').read_text())['calcium']['retention']
fig,axs=plt.subplots(1,2,figsize=(6.4,2.6),layout='constrained',sharey=True)
for ax,cell in zip(axs,['L1','L2']):
    rr=[r for r in cal if r['cell']==cell]
    ax.plot([r['level'] for r in rr],[r['observed'] for r in rr],'o-',color='#222222',label='Animal median')
    ax.plot([r['level'] for r in rr],[r['simulated'] for r in rr],'s--',color='#28647d',label='H4 model')
    ax.set(title=cell,xlabel='Relative light intensity',ylim=(0,1),xticks=[.25,.5,.75,1])
axs[0].set_ylabel('10 s response / transition peak')
axs[1].legend(frameon=False,fontsize=8,loc='lower right')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'fig2-calcium-comparison.{ext}',dpi=240)
plt.close(fig)
