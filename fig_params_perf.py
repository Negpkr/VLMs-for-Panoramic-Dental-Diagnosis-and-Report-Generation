import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT="Dental last submissions/figures"
M=["LLaVA-1.5","LLaVA-Med","HuatuoGPT-V","DentVLM","MedGemma","Qwen2.5-VL","LLaVA-OneV","LLaVA-Rad"]
PAR=[7,7,7,8,4,7,7,7]
BESTTUF=[.515,.488,.527,.442,.515,.486,.505,.666]   # best Tufts Macro-F1
FAM=["general","medical","medical","dental","medical","general","general","radiology"]
FC={"general":"#0072B2","medical":"#009E73","dental":"#D55E00","radiology":"#E69F00"}
MK=["o","s","D","^","v","P","X","*"]
# deterministic horizontal jitter for the crowded 7B column
jit={0:-0.34,1:-0.14,2:0.06,5:0.26,6:0.46,7:0.66}
fig,ax=plt.subplots(figsize=(8.6,5.0))
ax.axvspan(6.55,7.75,color="#f2f2f2",zorder=0)
ax.text(7.15,0.415,"7B cluster",ha="center",fontsize=8,color="#888",style="italic")
for i,m in enumerate(M):
    x=PAR[i]+jit.get(i,0.0)
    ax.scatter(x,BESTTUF[i],s=150,marker=MK[i],color=FC[FAM[i]],edgecolor="black",linewidth=0.7,zorder=3)
    off={"LLaVA-1.5":(-6,-14,"right"),"LLaVA-Med":(-6,-13,"right"),"HuatuoGPT-V":(0,9,"center"),
         "Qwen2.5-VL":(8,-2,"left"),"LLaVA-OneV":(9,6,"left"),"LLaVA-Rad":(-8,6,"right"),
         "MedGemma":(9,5,"left"),"DentVLM":(9,5,"left")}
    dx,dy,ha=off.get(m,(8,5,"left"))
    ax.annotate(m,(x,BESTTUF[i]),textcoords="offset points",xytext=(dx,dy),fontsize=8,ha=ha)
ax.set_xticks([4,7,8]); ax.set_xlim(3.3,8.9); ax.set_ylim(0.40,0.70)
ax.set_xlabel("Model size (billion parameters)")
ax.set_ylabel("Best Tufts Macro F1 (any strategy)")
ax.set_title("Parameter count does not track performance\n"
             "the 4B MedGemma matches the 7B field; the 8B dental model is weakest; the best is a 7B radiology model",fontsize=9.3)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=FC[c],label=c) for c in ["general","medical","dental","radiology"]],
          fontsize=8,title="family",loc="lower left")
ax.grid(alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
for e in("pdf","png"): fig.savefig(f"{OUT}/fig_params_performance.{e}",dpi=200,bbox_inches="tight")
print("WROTE fig_params_performance")
