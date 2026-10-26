"""Figure for RESULTS: median excess loss vs T by automaton size, Exp 19 (GPU) over Exp 11 (CPU). Run from expt_v1/: python plot_results.py ../dfa_transformer_gpu_results"""
import csv,statistics as S,collections as C,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=sys.argv[1]
def load(p):
    d=C.defaultdict(list)
    for r in csv.DictReader(open(p)): d[(int(r['K']),float(r['alpha']),int(r['T']))].append(float(r['excess_bits']))
    return d
e19=load('results/metrics.csv')
e11=load('../../11_dfa_exact_complexity_vs_alignment/expt_v1/results/metrics.csv')
col={4:'#86b6ef',8:'#3987e5',16:'#1c5cab',32:'#0d366b'}
ink,muted='#1f1f1d','#6b6a66'
fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharey=True,facecolor='#fcfcfb')
for ax,a in zip(axs,(1.0,0.15)):
    ax.set_facecolor('#fcfcfb')
    for n in (4,8,16,32):
        T19=sorted(T for (k,al,T) in e19 if k==n and al==a); T11=sorted(T for (k,al,T) in e11 if k==n and al==a)
        ax.plot(T11,[S.median(e11[(n,a,T)]) for T in T11],ls=(0,(1.5,2)),lw=2,color=col[n],alpha=.9)
        y=[S.median(e19[(n,a,T)]) for T in T19]
        ax.plot(T19,y,lw=2,color=col[n],marker='o',ms=4,label=f'n = {n} states',mec='#fcfcfb',mew=1)
        ax.annotate(f'n={n}',(T19[-1],y[-1]),xytext=(6,0),textcoords='offset points',va='center',fontsize=9,color=ink)
    ax.axvline(2**16,color=muted,lw=1,ls='--'); ax.text(2**16*1.1,1.17,'Exp 11 CPU cap (2^16)',fontsize=8,color=muted)
    ax.axhline(0.5,color=muted,lw=.8,alpha=.5)
    ax.set_xscale('log',base=2); ax.set_xlim(2**9/1.3,2**20*2.2); ax.set_ylim(0,1.25)
    ax.set_title(f'target share α = {a:g}',fontsize=11,color=ink,loc='left')
    ax.set_xlabel('training examples T',color=ink); ax.grid(alpha=.25,lw=.6)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
axs[0].set_ylabel('median excess loss (bits; ~1 = chance)',color=ink)
axs[0].legend(frameon=False,fontsize=9,loc='lower left')
fig.suptitle('Exp 19: Exp 11 transformer DFA grid trained 16x longer on GPU (solid) vs Exp 11 on CPU (dotted); median of 8 seeds × 4 targets',fontsize=10.5,color=ink,x=0.01,ha='left')
fig.tight_layout(); fig.savefig(out+'.png',dpi=150,facecolor='#fcfcfb'); fig.savefig(out+'.pdf',facecolor='#fcfcfb')
