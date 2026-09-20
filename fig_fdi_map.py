import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm
OUT="Dental last submissions/figures"
d=json.load(open("/tmp/claude-8063/-home-s222393187-Dental--claude-worktrees-dental-folder-access-1f4b32/410b55cb-df29-4b8b-966f-8f9691ffbedc/scratchpad/fdi_recall.json"))
FDI=d["fdi"]; REC=dict(zip(FDI,d["recall"])); GT=dict(zip(FDI,d["gt_count"]))
# arch layout (panoramic orientation: patient right = viewer left)
UP=[18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28]
LO=[48,47,46,45,44,43,42,41, 31,32,33,34,35,36,37,38]
cmap=plt.get_cmap("YlOrRd_r")  # low recall = red (bad), high = pale
vmin,vmax=0.10,0.48
fig,ax=plt.subplots(figsize=(11,4.4))
def draw(row_fdi,y,label):
    for i,f in enumerate(row_fdi):
        x=i*1.0 + (0.5 if i>=8 else 0)  # small gap at midline
        r=REC[f]; col=cmap((r-vmin)/(vmax-vmin))
        ax.add_patch(FancyBboxPatch((x,y),0.9,0.9,boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=0.8,edgecolor="#333",facecolor=col))
        tc="black" if r>0.28 else "white"
        ax.text(x+0.45,y+0.60,str(f),ha="center",va="center",fontsize=8,color=tc,fontweight="bold")
        ax.text(x+0.45,y+0.30,f"{r:.2f}",ha="center",va="center",fontsize=7.5,color=tc)
    ax.text(-0.7,y+0.45,label,ha="right",va="center",fontsize=9,fontweight="bold")
draw(UP,1.15,"Upper"); draw(LO,0.0,"Lower")
ax.text(3.75,2.25,"patient right (Q1/Q4)",ha="center",fontsize=8,style="italic",color="#555")
ax.text(12.25,2.25,"patient left (Q2/Q3)",ha="center",fontsize=8,style="italic",color="#555")
ax.axvline(8.25,0.05,0.9,color="#aaa",ls=":",lw=1)
ax.set_xlim(-2.2,16.7); ax.set_ylim(-0.4,2.5); ax.axis("off")
ax.set_title("Per-tooth pathology detection recall on DENTEX (few-shot, mean of 8 models)\n"
             "Redder = more missed; models detect posterior (premolar/molar) pathology better than anterior teeth",fontsize=10)
sm=cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin,vmax)); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,fraction=0.03,pad=0.01); cb.set_label("mean recall",fontsize=8)
plt.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_fdi_recall_map.{e}",dpi=200,bbox_inches="tight")
print("WROTE fig_fdi_recall_map")
