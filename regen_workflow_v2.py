import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT="/home/s222393187/Dental/Dental last submissions/figures"
NAVY="#1f3a5f"; TEAL="#0f7b6c"; BLUE="#3a5a8c"; GREEN="#2e7d4f"; CORAL="#c0562f"
AMBER="#b8860b"; PURPLE="#5b4a8a"; RED="#a33"; GREY="#555"
fig,ax=plt.subplots(figsize=(12.4,13.0)); ax.set_xlim(0,100); ax.set_ylim(0,128); ax.axis("off")
def rbox(x,y,w,h,fc,ec,lw=1.4,r=1.1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.15,rounding_size={r}",lw=lw,edgecolor=ec,facecolor=fc))
def titled(x,y,w,h,title,sub,ec,fc,tc=None,tsize=11,ssize=8.5):
    rbox(x,y,w,h,fc,ec); tc=tc or ec
    ax.text(x+1.8,y+h-2.0,title,ha="left",va="center",fontsize=tsize,fontweight="bold",color=tc)
    if sub: ax.text(x+1.8,y+h-4.3,sub,ha="left",va="top",fontsize=ssize,color="#333",linespacing=1.2)
def seclabel(x,y,txt,c): ax.text(x,y,txt,ha="left",va="center",fontsize=9.5,fontweight="bold",color=c)
def arrow(y1,y2,x=50): ax.add_patch(FancyArrowPatch((x,y1),(x,y2),arrowstyle="-|>",mutation_scale=15,lw=1.6,color="#8a97a8"))

ax.text(50,125,"Vision–Language Models for Panoramic Dental Radiographs",ha="center",fontsize=16,fontweight="bold",color=NAVY)
ax.text(50,121.4,"Evaluation workflow  ·  8 models × 3 prompting strategies × 3 datasets  ·  3 research questions",ha="center",fontsize=10.5,color=GREY)
# 1 Data
seclabel(4,117,"1 · DATA SOURCES",TEAL)
for x,t,s in [(4,"Tufts Dental Database","1,000 panoramic · Universal numbering"),
              (35.5,"DENTEX (QED)","752 panoramic · FDI · 4 disease classes"),
              (67,"PR-Reports","121 cases · expert radiologist reports")]:
    titled(x,109.5,29,6.0,t,s,TEAL,"#e3f1ee",tsize=10,ssize=8)
# 2 GT
seclabel(4,105.5,"2 · GROUND-TRUTH PREPARATION & VERIFICATION",BLUE)
rbox(4,98.5,92,5.6,"#e7edf7",BLUE); ax.text(50,101.3,"32-slot missing-teeth masks (Tufts) · abnormal-tooth + disease labels (DENTEX) · paired expert reports · re-extraction & readability checks",ha="center",va="center",fontsize=8.5,color="#333")
# 3 Model zoo
rbox(4,68,92,25.5,"#f3f5f9","#c9d2df",lw=1.2,r=1.2)
rbox(4,89.6,92,3.6,NAVY,NAVY,lw=0,r=1.2)
ax.text(5.8,91.4,"3 · Model zoo  ×  prompting strategies  —  common inference interface",ha="left",va="center",fontsize=10.5,fontweight="bold",color="white")
models=[("LLaVA-1.5-7B","general-purpose",BLUE,"#e8edf7"),("LLaVA-Med-v1.5-7B","medical",GREEN,"#e6f2e9"),
        ("HuatuoGPT-Vision-7B","medical",GREEN,"#e6f2e9"),("DentVLM","dental-specialised",CORAL,"#fbeee7"),
        ("Qwen2.5-VL-7B ★","general-purpose",BLUE,"#e8edf7"),("LLaVA-OneVision-7B ★","general-purpose",BLUE,"#e8edf7"),
        ("MedGemma-4B ★","medical (Gemma-3)",GREEN,"#e6f2e9"),("LLaVA-Rad ★","radiology-adapted",CORAL,"#fdf0e6")]
bw=20.9; gap=1.8; x0=5.6
for i,(nm,ty,ec,fc) in enumerate(models):
    row=i//4; col=i%4; x=x0+col*(bw+gap); y=83.0-row*6.4
    titled(x,y,bw,5.4,nm,ty,ec,fc,tsize=8.2,ssize=7.2)
ax.text(94.2,72.3,"★ newer\n   models",ha="right",va="center",fontsize=6.8,color=GREY)
ax.text(6,70.0,"Prompting:",ha="left",va="center",fontsize=9,fontweight="bold",color=NAVY)
px=17
for lab in ["Zero-shot","Few-shot","Chain-of-thought"]:
    w=15 if "Chain" in lab else 12
    rbox(px,68.6,w,2.8,"white",NAVY,lw=1.3,r=1.4); ax.text(px+w/2,70.0,lab,ha="center",va="center",fontsize=8.3,fontweight="bold",color=NAVY); px+=w+2.5
ax.text(px+1,70.0,"effect analysed under RQ2",ha="left",va="center",fontsize=7.2,style="italic",color=GREY)
# 4 inference / 5 parser
titled(4,61,92,5.6,"4 · Inference harness","Slurm on Deakin GPU host · L40S / RTX 4500 Ada / A4000 / A100 · fp16 (bf16 for MedGemma) · per-model checkpointing → raw generations",BLUE,"#eef2f8",tsize=10,ssize=8)
titled(4,53.5,92,5.6,"5 · Schema-constrained parser","Strict output schema · explicit numbering (Universal ↔ FDI) · empty / off-schema output kept as all-negative → per-tooth predictions",BLUE,"#eef2f8",tsize=10,ssize=8)
# 6 scoring
seclabel(4,50,"6 · SCORING & METRICS",PURPLE)
for x,t,s in [(4,"Patient-level classification","Macro-F1 · accuracy · specificity\nbalanced acc. · 95% patient bootstrap"),
              (35.5,"Multi-label pathology","Hamming · example / label F1 · Jaccard\nMacro-4 F1 vs no-finding baseline"),
              (67,"Report quality (RQ3)","BLEU-4 · ROUGE-1/2/L · BERTScore-F1\nKL & JS divergence")]:
    titled(x,41,29,7.2,t,s,PURPLE,"#efeaf7",tsize=9,ssize=7.4)
titled(4,32.6,92,6.8,"Hallucination & error-type audit","CHR = 1 − precision · hallucinations per image · correct-condition / wrong-tooth vs. unsupported localisation\nplain-prompt vs. schema-reinforced prompt ablation",RED,"#f7e9ea",tsize=9,ssize=7.5)
# 7 RQs
seclabel(4,30.5,"7 · RESEARCH QUESTIONS",AMBER)
for x,t,s in [(4,"RQ1 · Diagnostic performance","how do general / medical / dental VLMs compare?"),
              (35.5,"RQ2 · Prompting strategies","does zero / few / CoT change reliability?"),
              (67,"RQ3 · Report quality","how close are reports to expert references?")]:
    titled(x,23,29,6.0,t,s,AMBER,"#fbf3d9",tsize=8.6,ssize=7.2)
# 8 outputs
rbox(4,12,92,7.6,"#e7f3ea",GREEN)
ax.text(5.8,18.0,"8 · Outputs & synthesis",ha="left",va="center",fontsize=10,fontweight="bold",color=GREEN)
ax.text(5.8,15.3,"Summary tables · per-class & strategy figures · CHR & divergence plots · per-case JSON → research analysis & manuscript",ha="left",va="center",fontsize=8,color="#333")
ax.text(5.8,13.3,"Headline: medical instruction tuning & model recency > dental branding · newer general models match specialists · disease-level F1 near floor · CHR ≈ 0.9 on DENTEX",ha="left",va="center",fontsize=7.5,style="italic",color=GREEN)
rbox(30,3.5,40,4.6,NAVY,NAVY,lw=0,r=2.3); ax.text(50,5.8,"Analysis · Discussion · Publication",ha="center",va="center",fontsize=12,fontweight="bold",color="white")
for y1,y2 in [(109.4,104.3),(98.4,93.7),(67.9,66.8),(60.9,59.2),(53.4,48.4),(40.9,39.5),(32.4,29.4),(22.9,19.8),(11.9,8.2)]:
    arrow(y1,y2)
plt.subplots_adjust(left=0.01,right=0.99,top=0.995,bottom=0.005)
for ext in ("pdf","png"): fig.savefig(f"{OUT}/workflow_diagram.{ext}",dpi=200,bbox_inches="tight")
print("WROTE workflow_diagram v2b")
