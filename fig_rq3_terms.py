import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT="Dental last submissions/figures"
d=json.load(open("/tmp/claude-8063/-home-s222393187-Dental--claude-worktrees-dental-folder-access-1f4b32/410b55cb-df29-4b8b-966f-8f9691ffbedc/scratchpad/rq3_dist.json"))
top=d["top"]; EXP=d["expert"]; MC=d["models"]
def rel(C,w): return 100*C.get(w,0)/max(1,sum(C.values()))
SER=[("Expert","#000000"),("HuatuoGPT-V","#009E73"),("Qwen2.5-VL","#0072B2"),("MedGemma","#E69F00")]
y=np.arange(len(top))[::-1]; h=0.2
fig,ax=plt.subplots(figsize=(8.8,6.2))
for j,(name,col) in enumerate(SER):
    C=EXP if name=="Expert" else MC[name]
    vals=[rel(C,w) for w in top]
    ax.barh(y+(1.5-j)*h,vals,h,label=name,color=col,edgecolor="black",linewidth=0.3)
ax.set_yticks(y); ax.set_yticklabels(top,fontsize=9)
ax.set_xlabel("Term frequency (% of all tokens in the report corpus)")
ax.set_title("RQ3 clinical-term usage: expert vs generated reports (top expert terms)\n"
             "MedGemma barely uses clinical vocabulary (terse/degenerate reports); others track the expert",fontsize=9.5)
ax.legend(fontsize=8,loc="lower right"); ax.grid(axis="x",alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_rq3_term_frequency.{e}",dpi=200,bbox_inches="tight")
print("WROTE fig_rq3_term_frequency")
