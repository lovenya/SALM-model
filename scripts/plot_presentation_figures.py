import json, pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path("/scratch/l/lovenya/salm_model_outputs/runs")
OUT = pathlib.Path("/scratch/l/lovenya/salm_model_outputs/presentation_plots"); OUT.mkdir(exist_ok=True)
INK,MUT,GRID = "#0F1519","#5A6B72","#D8E0E3"
TEAL,OCHRE,RED,GREEN = "#1F7A8C","#B5761C","#9E4038","#4A7A52"

def H(rel):
    p = R/rel/"history.json"
    return json.loads(p.read_text()) if p.exists() else None
def series(h,k):
    pts=[(e["epoch"],e[k]) for e in h if e.get(k) is not None] if h else []
    return (tuple(p[0] for p in pts), tuple(p[1] for p in pts))

runs = {
 "L2 (32.3M), diffusion loss only":      ("s2a_audioldm_whisper_xformer/long100", RED),
 "L0 (0.66M), diffusion loss only":      ("s2a_audioldm_whisper_mean_linear/long100_L0", "#C08A84"),
 "L0f (0.66M, U-Net frozen)":            ("s2a_audioldm_whisper_mean_linear/long100_L0f", "#D6B3AE"),
 "L0 + O2 (0.66M + alignment)":          ("s2a_audioldm_whisper_mean_linear/long100_L0_o2", OCHRE),
 "L2 + O2 (32.3M + alignment)":          ("s2a_audioldm_whisper_xformer/long100_L2_o2", TEAL),
}
def style(ax):
    ax.set_facecolor("white")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9); ax.grid(True,color=GRID,lw=.6,alpha=.7); ax.set_axisbelow(True)
    ax.title.set_color(INK); ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)

# ---- FIG 1: the loss never converged, in ANY run -------------------------------
fig,ax = plt.subplots(figsize=(9.5,4.8),dpi=200); style(ax)
for lbl,(rel,c) in runs.items():
    h=H(rel); x,y = series(h,"val_loss")
    if x: ax.plot(x,y,color=c,lw=1.5,label=f"{lbl}  ({y[0]:.4f} → {y[-1]:.4f})")
ax.set_title("The diffusion loss never converged — in any run, with or without the fix",fontsize=12.5,pad=12)
ax.set_xlabel("epoch"); ax.set_ylabel("validation ε-prediction MSE")
leg=ax.legend(frameon=False,fontsize=8.5,loc="upper right")
for t in leg.get_texts(): t.set_color(INK)
fig.text(.012,.02,"Every curve is flat. The two runs that LEARNED (orange, teal) look identical here to the three that did not.",
         fontsize=8.5,color=MUT)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(OUT/"01_loss_never_converged.png",facecolor="white"); plt.close(fig)

# ---- FIG 2: the metric that DID move -------------------------------------------
fig,ax = plt.subplots(figsize=(9.5,4.8),dpi=200); style(ax)
for lbl,(rel,c) in runs.items():
    h=H(rel); x,y = series(h,"val_clap_r1")
    if x: ax.plot(x,y,color=c,lw=1.8,label=f"{lbl}  → {y[-1]:.3f}")
ax.axhline(0.0625,color=INK,ls="--",lw=1.2,alpha=.75)
ax.annotate("chance = 1/batch = 0.0625",xy=(50,0.0625),xytext=(0,7),textcoords="offset points",
            fontsize=9,color=INK)
ax.set_title("The metric we added, and what it showed immediately",fontsize=12.5,pad=12)
ax.set_xlabel("epoch"); ax.set_ylabel("val_clap_r1  (scene retrieval accuracy)")
ax.set_ylim(0,0.65)
leg=ax.legend(frameon=False,fontsize=8.5,loc="center right")
for t in leg.get_texts(): t.set_color(INK)
fig.text(.012,.02,"Same runs, same epochs, same data as the previous figure. The two with the auxiliary loss separate from chance within ~10 epochs.",
         fontsize=8.5,color=MUT)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(OUT/"02_metric_that_moved.png",facecolor="white"); plt.close(fig)

# ---- FIG 3: what "converging" actually looked like -----------------------------
fig,ax = plt.subplots(figsize=(9.5,4.8),dpi=200); style(ax)
for lbl,rel,c in [("L2 + O2 (32.3M)","s2a_audioldm_whisper_xformer/long100_L2_o2",TEAL),
                  ("L0 + O2 (0.66M)","s2a_audioldm_whisper_mean_linear/long100_L0_o2",OCHRE)]:
    h=H(rel); x,y=series(h,"train_align")
    if x: ax.plot(x,y,color=c,lw=1.8,label=f"{lbl}  {y[0]:.3f} → {y[-1]:.3f}")
ax.set_yscale("log")
ax.set_title("The one curve that genuinely converged: the alignment term",fontsize=12.5,pad=12)
ax.set_xlabel("epoch"); ax.set_ylabel("train  (1 − cos)   [log scale]")
leg=ax.legend(frameon=False,fontsize=9); 
for t in leg.get_texts(): t.set_color(INK)
fig.text(.012,.02,"The 32.3M conditioner drives it to 0.014 (train cosine 0.986) — it can memorise the targets. The 0.66M one cannot, and generalises better.",
         fontsize=8.5,color=MUT)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(OUT/"03_alignment_converged.png",facecolor="white"); plt.close(fig)

print("wrote:")
for p in sorted(OUT.glob("*.png")): print("  ",p)
