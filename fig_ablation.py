import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT="Dental last submissions/figures"
# tab:reinforce  (plain, reinforced)  Tufts pooled-slot Macro F1
DATA={
 "Zero-shot":{"MedGemma":(0.390,0.385),"Qwen2.5-VL":(0.486,0.505),"LLaVA-OneVision":(0.491,0.490)},
 "Few-shot": {"MedGemma":(0.317,0.257),"Qwen2.5-VL":(0.486,0.486),"LLaVA-OneVision":(0.433,0.429)},
 "CoT":      {"MedGemma":(0.515,0.456),"Qwen2.5-VL":(0.458,0.473),"LLaVA-OneVision":(0.505,0.475)}}
MODELS=["MedGemma","Qwen2.5-VL","LLaVA-OneVision"]; STR=["Zero-shot","Few-shot","CoT"]
OKm={"MedGemma":"#009E73","Qwen2.5-VL":"#0072B2","LLaVA-OneVision":"#E69F00"}
rows=[]
for s in STR:
    for m in MODELS:
        p,r=DATA[s][m]; rows.append((f"{m}\n({s})",m,r-p))
labels=[x[0] for x in rows]; deltas=[x[2] for x in rows]; cols=[OKm[x[1]] for x in rows]
y=np.arange(len(rows))[::-1]
fig,ax=plt.subplots(figsize=(7.4,5.2))
ax.barh(y,deltas,color=cols,edgecolor="black",linewidth=0.4,height=0.62)
ax.axvline(0,color="black",linewidth=0.8)
for yi,dv in zip(y,deltas):
    ax.text(dv+(0.001 if dv>=0 else -0.001),yi,f"{dv:+.3f}",va="center",
            ha="left" if dv>=0 else "right",fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8)
ax.set_xlim(-0.075,0.045)
ax.set_xlabel("Δ Macro F1  (reinforced − plain prompt)")
ax.set_title("Prompt-format reinforcement rarely helps and can hurt (Tufts)\nnegative = schema-reinforcement lowered Macro F1",fontsize=10)
ax.axvspan(-0.075,0,color="#FDECEA",zorder=0); ax.axvspan(0,0.045,color="#EAF6EF",zorder=0)
ax.text(-0.037,len(rows)-0.3,"hurts",color="#b03a2e",fontsize=9,ha="center",style="italic")
ax.text(0.022,len(rows)-0.3,"helps",color="#1e7d4f",fontsize=9,ha="center",style="italic")
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=OKm[m],label=m) for m in MODELS],fontsize=8,loc="lower left",framealpha=0.9)
ax.grid(axis="x",alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_reinforce_delta.{e}",dpi=200,bbox_inches="tight")
print("WROTE fig_reinforce_delta")
