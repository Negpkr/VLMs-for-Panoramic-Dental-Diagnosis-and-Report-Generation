import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
OUT="Dental last submissions/figures"
d=json.load(open("/tmp/claude-8063/-home-s222393187-Dental--claude-worktrees-dental-folder-access-1f4b32/410b55cb-df29-4b8b-966f-8f9691ffbedc/scratchpad/abstain.json"))
MODELS=["LLaVA-1.5","LLaVA-Med","HuatuoGPT-V","DentVLM","MedGemma","Qwen2.5-VL","LLaVA-OneV","LLaVA-Rad"]
STRAT=[("zero_shot","Zero-shot"),("few_shot","Few-shot"),("cot","CoT")]
OK=["#0072B2","#E69F00","#009E73"]
x=np.arange(len(MODELS)); w=0.26
fig,ax=plt.subplots(figsize=(9.2,4.6))
for j,(k,lab) in enumerate(STRAT):
    vals=[d[m][k] for m in MODELS]
    b=ax.bar(x+(j-1)*w,vals,w,label=lab,color=OK[j],edgecolor="black",linewidth=0.4)
    for xi,v in zip(x+(j-1)*w,vals):
        if v>=1: ax.text(xi,v+1.2,f"{v:.0f}",ha="center",va="bottom",fontsize=6.5)
ax.axvspan(3.5,7.5,color="#FFF6E5",zorder=0)
ax.text(5.5,104,"newer models",ha="center",fontsize=8,style="italic",color="#8a6d00")
ax.set_xticks(x); ax.set_xticklabels(MODELS,rotation=20,ha="right")
ax.set_ylabel("Empty / abstained outputs (% of DENTEX cases)")
ax.set_ylim(0,110); ax.set_yticks(range(0,101,20))
ax.set_title("Structured-output compliance on DENTEX pathology (every case has ≥1 ground-truth finding)\nHigher bar = the model returned no findings at all (abstention / format failure)",fontsize=9.5)
ax.legend(fontsize=8,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.14),frameon=False)
ax.grid(axis="y",alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_abstention_dentex.{e}",dpi=200,bbox_inches="tight")
print("WROTE fig_abstention_dentex")
