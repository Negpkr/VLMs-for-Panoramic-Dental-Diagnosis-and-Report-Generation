import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT="/home/s222393187/Dental/Dental last submissions/figures"
fig,ax=plt.subplots(figsize=(11.5,7.6)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
def box(cx,cy,w,h,title,sub=None,fc="#eef2f7",ec="black",tsize=10,ssize=8,bold=True):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.2,rounding_size=1.2",
        linewidth=1.0,edgecolor=ec,facecolor=fc))
    if sub:
        ax.text(cx,cy+h*0.16,title,ha="center",va="center",fontsize=tsize,fontweight="bold" if bold else "normal")
        ax.text(cx,cy-h*0.22,sub,ha="center",va="center",fontsize=ssize,color="#333")
    else:
        ax.text(cx,cy,title,ha="center",va="center",fontsize=tsize,fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.3,color="black"))
def band(y,label):
    ax.text(1.5,y,label,ha="left",va="center",fontsize=8.5,fontweight="bold",color="#555",rotation=90)

# Title
ax.text(50,97,"Vision–Language Models for Panoramic Dental Radiographs",ha="center",fontsize=14,fontweight="bold")
ax.text(50,93,"Evaluation workflow  ·  8 models  ·  3 prompting strategies  ·  3 datasets  ·  3 research questions",
        ha="center",fontsize=9.5,color="#444")

# Row A: datasets
band(85,"1  Data")
for cx,name,sub in [(22,"Tufts","1,000 panoramics\nUniversal 1–32"),
                    (50,"DENTEX (QED)","752 images\nFDI 11–48, 4 diseases"),
                    (78,"PR-Reports","121 panoramics\nexpert reports")]:
    box(cx,85,24,9,name,sub,fc="#e7eef7")
# Row B: model panel (8 models grouped) + prompting
band(69,"2  Models")
ax.add_patch(FancyBboxPatch((8,62),84,11,boxstyle="round,pad=0.2,rounding_size=1.2",lw=1.0,edgecolor="black",facecolor="#f4f1e8"))
ax.text(50,71.2,"Eight open-source VLMs under one common interface & strict parser",ha="center",fontsize=9.5,fontweight="bold")
groups=[("General","#e7eef7",["LLaVA-1.5","Qwen2.5-VL*","LLaVA-OneV.*"]),
        ("Medical","#e9f3ea",["LLaVA-Med","HuatuoGPT-V","MedGemma*"]),
        ("Radiology","#fbeee0",["LLaVA-Rad*"]),
        ("Dental","#f7e9ef",["DentVLM"])]
xx=11
for gname,gc,ms in groups:
    gw=len(ms)*9.5+2
    ax.text(xx+gw/2-1,66.7,gname,ha="center",fontsize=7.5,style="italic",color="#555")
    for m in ms:
        box(xx+4.0,64.3,8.6,3.4,m,fc=gc,tsize=6.8,bold=False)
        xx+=9.2
    xx+=2.2
ax.text(91,63.5,"* newer\n  models",ha="right",fontsize=6.5,color="#888")
# prompting chip
box(50,58.5,60,3.2,"Prompting:  zero-shot   ·   few-shot   ·   chain-of-thought (CoT)",fc="#eeeeee",tsize=8.2,bold=False)

# Row C: inference
band(52,"3  Run")
box(50,52,52,4.2,"Inference harness  →  free-text generation",fc="#eef2f7",tsize=9)
# Row D: parser
band(45,"4  Parse")
box(50,45,66,5.0,"Strict numbering-aware parser","reads answer line only; Universal/FDI; empty or off-schema → all-negative",fc="#f4f1e8",tsize=9)
# Row E: scoring
band(33,"5  Score")
for cx,t,s in [(20,"Detection","Tufts miss/present;\nDENTEX abnormal (Macro-F1)"),
               (43,"Pathology","4 diseases one-vs-rest\n(Macro-4 F1)"),
               (66,"Reporting","BLEU/ROUGE/BERTScore\nKL, JSD"),
               (88,"Hallucination","CHR = 1−precision;\nerror-type audit")]:
    box(cx,33,20,8.5,t,s,fc="#eef2f7",tsize=8.5,ssize=7)
# Row F: stats
band(22,"6  Stats")
box(50,22,66,4.6,"Bootstrap 95% CIs  ·  accuracy & Hamming reported, not headline  ·  plain vs. reinforced-prompt ablation",fc="#eeeeee",tsize=8,bold=False)
# Row G: RQs
band(12,"7  RQs")
for cx,t in [(25,"RQ1  detection"),(50,"RQ2  prompting"),(75,"RQ3  reporting")]:
    box(cx,12,26,5.2,t,fc="#e9f3ea",tsize=9)
box(50,4.5,60,4.2,"Analysis  ·  Discussion  ·  Publication",fc="#f7f7f7",tsize=9,bold=False)

# arrows between bands
for y1,y2 in [(80.5,73.5),(57,54.3),(49.8,47.6),(42.4,37.4),(28.6,24.4),(19.6,14.8),(9.3,6.7)]:
    arrow(50,y1,50,y2)
plt.subplots_adjust(left=0.02,right=0.98,top=0.99,bottom=0.01)
for ext in ("pdf","png"): fig.savefig(f"{OUT}/workflow_diagram.{ext}",dpi=200,bbox_inches="tight")
print("WROTE workflow_diagram.pdf / .png (8 models)")
