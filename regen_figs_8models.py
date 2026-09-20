import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT="/home/s222393187/Dental/Dental last submissions/figures"
MODELS=["LLaVA-1.5","LLaVA-Med","HuatuoGPT-V","DentVLM","MedGemma","Qwen2.5-VL","LLaVA-OneV","LLaVA-Rad"]
STRAT=["Zero-shot","Few-shot","CoT"]
OK=["#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#56B4E9","#8C564B","#000000"]  # Okabe-Ito-ish (colourblind-safe)
def newer_shade(ax): ax.axvspan(3.5,7.5,color="#f2c94c",alpha=0.13,zorder=0)
# ---- Macro-F1 (Figure3) ----
TUFTS={"Zero-shot":[.492,.470,.497,.442,.390,.486,.491,.418],"Few-shot":[.515,.488,.527,.441,.317,.486,.433,.666],"CoT":[.491,.451,.505,.441,.515,.458,.505,.463]}
DENTEXA={"Zero-shot":[.497,.459,.519,.460,.268,.473,.517,.462],"Few-shot":[.535,.485,.530,.459,.558,.492,.502,.371],"CoT":[.465,.460,.540,.459,.354,.531,.495,.472]}
def grouped(ax,data,title,baseline=None):
    x=np.arange(len(MODELS)); w=0.26; sc=[OK[0],OK[1],OK[2]]
    for i,s in enumerate(STRAT):
        ax.bar(x+(i-1)*w,data[s],w,label=s,color=sc[i],edgecolor="black",linewidth=0.4)
    if baseline is not None:
        ax.axhline(baseline,ls="--",lw=1,color="#b3261e")
        ax.text(-0.35,0.715,f"-- no-finding baseline ({baseline:.2f})",ha="left",va="top",fontsize=7,color="#b3261e")
    newer_shade(ax)
    ax.set_xticks(x); ax.set_xticklabels(MODELS,rotation=35,ha="right",fontsize=8)
    ax.set_ylabel("Macro F1"); ax.set_title(title,fontsize=10); ax.set_ylim(0,0.75); ax.legend(fontsize=8,ncol=3,loc="upper right",framealpha=0.9)
fig,axes=plt.subplots(2,1,figsize=(7.0,7.4))
grouped(axes[0],TUFTS,"Tufts missing-teeth detection (Macro F1)")
grouped(axes[1],DENTEXA,"DENTEX abnormal-tooth detection (Macro F1)",baseline=0.459)
axes[0].text(5.5,0.69,"newer models",fontsize=8,style="italic",ha="center",color="#a07800")
plt.tight_layout()
for e in("pdf","png"): fig.savefig(f"{OUT}/Figure3_model_strategy_macroF1.{e}",dpi=200,bbox_inches="tight")
plt.close(fig)
# ---- RQ3 similarity ----
RQ=dict(BLEU=[.030,.107,.122,.008,.114,.178,.031,.038],R1=[.258,.411,.435,.159,.329,.413,.258,.255],
 R2=[.097,.192,.199,.038,.150,.236,.118,.115],RL=[.216,.330,.301,.127,.229,.312,.235,.230],
 BERT=[.620,.668,.653,.488,.600,.602,.579,.536],KL=[7.34,5.71,4.57,8.48,6.06,4.73,7.26,7.25],JSD=[.656,.492,.444,.817,.591,.497,.662,.709])
fig,ax=plt.subplots(figsize=(7.2,3.9)); x=np.arange(len(MODELS)); w=0.16
for i,(k,lab) in enumerate([("BLEU","BLEU-4"),("R1","ROUGE-1"),("R2","ROUGE-2"),("RL","ROUGE-L"),("BERT","BERTScore")]):
    ax.bar(x+(i-2)*w,RQ[k],w,label=lab,color=OK[i],edgecolor="black",linewidth=0.4)
newer_shade(ax); ax.set_xticks(x); ax.set_xticklabels(MODELS,rotation=35,ha="right",fontsize=8)
ax.set_ylabel("score (higher better)"); ax.set_title("RQ3 report similarity to expert reports",fontsize=10); ax.set_ylim(0,0.8); ax.legend(fontsize=8,ncol=5,loc="upper center")
plt.tight_layout()
for e in("pdf","png"): fig.savefig(f"{OUT}/fig_rq3_similarity.{e}",dpi=200,bbox_inches="tight")
plt.close(fig)
# ---- RQ3 divergence ----
fig,ax=plt.subplots(figsize=(7.2,3.7)); w=0.38
ax.bar(x-w/2,RQ["KL"],w,label="KL (bits)",color=OK[0],edgecolor="black")
ax2=ax.twinx(); ax2.bar(x+w/2,RQ["JSD"],w,label="JSD (bits)",color=OK[1],edgecolor="black")
newer_shade(ax); ax.set_xticks(x); ax.set_xticklabels(MODELS,rotation=35,ha="right",fontsize=8)
ax.set_ylabel("KL divergence (bits)"); ax2.set_ylabel("JSD (bits)"); ax2.set_ylim(0,1)
ax.set_title("RQ3 divergence from expert token distribution (lower better)",fontsize=10)
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,fontsize=8,loc="upper left")
plt.tight_layout()
for e in("pdf","png"): fig.savefig(f"{OUT}/fig_rq3_divergence.{e}",dpi=200,bbox_inches="tight")
plt.close(fig)
# ---- Figure4 heatmaps (viridis) ----
TM=np.array([[.221,.220,.141],[.077,.117,.024],[.243,.312,.263],[.002,.001,.001],[.323,.333,.295],[.156,.187,.053],[.163,.259,.174],[.322,.440,.270]])
DA=np.array([[.109,.197,.013],[.000,.160,.002],[.149,.229,.197],[.001,.000,.000],[.277,.334,.213],[.032,.169,.174],[.147,.268,.089],[.097,.215,.040]])
fig,axes=plt.subplots(1,2,figsize=(7.8,4.8),gridspec_kw={"wspace":0.10})
for ax,mat,title in [(axes[0],TM,"Tufts: missing-tooth F1"),(axes[1],DA,"DENTEX: abnormal-tooth F1")]:
    im=ax.imshow(mat,cmap="viridis",vmin=0,vmax=0.55,aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(STRAT,fontsize=8); ax.set_yticks(range(8))
    ax.set_yticklabels(MODELS if ax is axes[0] else [""]*8,fontsize=8); ax.set_title(title,fontsize=10)
    for i in range(8):
        for j in range(3):
            v=mat[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7,color=("white" if v<0.33 else "black"))
    ax.axhline(3.5,color="#d62728",lw=1.4)
fig.colorbar(im,ax=axes,fraction=0.03,pad=0.02,label="F1")
fig.suptitle("Per-class positive F1 by model $\\times$ strategy (red line: original vs newer)",fontsize=10)
for e in("pdf","png"): fig.savefig(f"{OUT}/Figure4_per_class_F1_heatmaps.{e}",dpi=200,bbox_inches="tight")
plt.close(fig)
# ---- Hallucination vs recall (8 models) ----
CHR=[.941,.938,.914,1.000,.861,.967,.908,.980]; REC=[.050,.088,.116,.000,.312,.046,.232,.087]
CAT=["general","medical","medical","dental","medical","general","general","radiology"]
cmap={"general":OK[0],"medical":OK[2],"dental":OK[3],"radiology":OK[1]}
fig,ax=plt.subplots(figsize=(7.2,5.0))
for i,m in enumerate(MODELS):
    ax.scatter(REC[i],CHR[i],s=130,color=cmap[CAT[i]],edgecolor="black",linewidth=0.7,zorder=3)
    off=(-58,4) if m=="MedGemma" else (7,4)
    ax.annotate(m,(REC[i],CHR[i]),textcoords="offset points",xytext=off,fontsize=8)
import matplotlib.patches as mp
handles=[mp.Patch(color=cmap[c],label=c) for c in ["general","medical","dental","radiology"]]
ax.legend(handles=handles,fontsize=8,title="model type",loc="upper right")
ax.set_xlabel("Recall / sensitivity (finding level)"); ax.set_ylabel("Clinical Hallucination Rate  (CHR = 1 - precision)")
ax.set_title("Hallucination vs recall on DENTEX (few-shot)\nideal = high recall (right), low CHR (bottom)",fontsize=10)
ax.set_xlim(-0.02,0.35); ax.set_ylim(0.82,1.01); ax.grid(alpha=0.3)
plt.tight_layout()
for e in("pdf","png"): fig.savefig(f"{OUT}/fig_hallucination_vs_recall.{e}",dpi=200,bbox_inches="tight")
plt.close(fig)
print("WROTE Figure3, Figure4, fig_rq3_similarity, fig_rq3_divergence, fig_hallucination_vs_recall (COLOR, 8 models)")
