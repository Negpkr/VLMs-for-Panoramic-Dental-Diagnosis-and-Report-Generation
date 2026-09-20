import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT="Dental last submissions/figures"
MODELS=["LLaVA-1.5","LLaVA-Med","HuatuoGPT-V","DentVLM","MedGemma","Qwen2.5-VL","LLaVA-OneV","LLaVA-Rad"]
FAM=["general","medical","medical","dental","medical","general","general","radiology"]
FC={"general":"#0072B2","medical":"#009E73","dental":"#D55E00","radiology":"#E69F00"}
# Tufts Macro-F1 (zero,few,cot)
TUF=[(.492,.515,.491),(.470,.488,.451),(.497,.527,.505),(.442,.441,.441),
     (.390,.317,.515),(.486,.486,.458),(.491,.433,.505),(.418,.666,.463)]
# DENTEX Table-A Macro-F1 (zero,few,cot)
DEN=[(.497,.535,.465),(.459,.485,.460),(.519,.530,.540),(.460,.459,.459),
     (.268,.558,.354),(.473,.492,.531),(.517,.502,.495),(.462,.371,.472)]
MK=["o","s","D","^","v","P","X","*"]

# ---------- Figure: strategy-sensitivity slope (2 panels) ----------
fig,axes=plt.subplots(1,2,figsize=(10.4,4.8),sharey=False)
for ax,DATA,title in [(axes[0],TUF,"Tufts (missing-teeth)"),(axes[1],DEN,"DENTEX (abnormal-tooth)")]:
    for i,m in enumerate(MODELS):
        ys=DATA[i]; xs=[0,1,2]
        ax.plot(xs,ys,marker=MK[i],color=FC[FAM[i]],lw=1.6,ms=7,mec="black",mew=0.4,label=m,alpha=0.9)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Zero-shot","Few-shot","CoT"])
    ax.set_title(title,fontsize=10); ax.grid(alpha=0.3); ax.set_ylabel("Macro F1")
    ax.set_ylim(0.24,0.70)
axes[1].legend(fontsize=7.5,ncol=2,loc="lower right",framealpha=0.92)
fig.suptitle("Prompting-strategy sensitivity per model (which strategy helps which model)",fontsize=11,y=1.00)
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_strategy_sensitivity.{e}",dpi=200,bbox_inches="tight")
plt.close(fig); print("WROTE fig_strategy_sensitivity")

# ---------- Figure: recency vs best performance ----------
# first-public dates (arXiv YYMM; DentVLM designated 2025)
DATE=[2023+9.5/12, 2023+5.5/12, 2024+5.5/12, 2025+5.0/12, 2025+6.5/12, 2025+1.5/12, 2024+7.5/12, 2024+2.5/12]
BEST=[max(t) for t in TUF]  # best Tufts Macro-F1
fig,ax=plt.subplots(figsize=(8.6,5.0))
for i,m in enumerate(MODELS):
    ax.scatter(DATE[i],BEST[i],s=150,marker=MK[i],color=FC[FAM[i]],edgecolor="black",linewidth=0.7,zorder=3)
    dx,dy=(6,5)
    if m=="MedGemma": dx,dy=(6,-12)
    if m=="DentVLM": dx,dy=(6,-12)
    if m=="LLaVA-OneV": dx,dy=(-58,4)
    ax.annotate(m,(DATE[i],BEST[i]),textcoords="offset points",xytext=(dx,dy),fontsize=8)
# trend line
z=np.polyfit(DATE,BEST,1); xs=np.linspace(min(DATE)-0.1,max(DATE)+0.1,50)
ax.plot(xs,np.polyval(z,xs),"--",color="grey",lw=1.2,zorder=1,label=f"linear trend (slope {z[0]:+.03f}/yr)")
ax.set_xlabel("Model first-public date"); ax.set_ylabel("Best Macro F1 on Tufts (any strategy)")
ax.set_title("Peak Macro F1 clusters in a narrow band across families and release dates\nDental branding gives no edge (DentVLM lowest); the one outlier (LLaVA-Rad) is task-format matched",fontsize=9.3)
import matplotlib.patches as mp
fam_h=[mp.Patch(color=FC[c],label=c) for c in ["general","medical","dental","radiology"]]
leg1=ax.legend(handles=fam_h,fontsize=8,title="family",loc="upper left")
ax.add_artist(leg1); ax.legend(fontsize=8,loc="lower right")
ax.grid(alpha=0.3); ax.set_ylim(0.40,0.70)
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_recency_performance.{e}",dpi=200,bbox_inches="tight")
plt.close(fig); print("WROTE fig_recency_performance")
