"""Publication figures for the Neurocomputing MIRA manuscript.

All shapes and text are native vector elements.  The diagrams are based on
paper/lifted_cavity/{main,method_lct}.tex and experiments/lifted_cavity/
{models,lct,anchors}.py.  Calibration cells and sensor icons are illustrative.
The shared noise example is evaluated from its stated Gaussian model, not from
an experimental result file.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as P
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path("C:/Windows/Fonts/times.ttf")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 16,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.8, "savefig.facecolor": "white",
})
INK = "#202529"
GREY = "#b9c0c7"
PALE = "#f3f5f7"
BLUE = "#2776b6"
BLUE_L = "#e7f0f8"
TEAL = "#377d79"
TEAL_L = "#e8f3f1"
PURPLE = "#79559b"
PURPLE_L = "#f0eaf5"
ORANGE = "#bd7b29"
ORANGE_L = "#fbf0df"
RED = "#a95d61"
RED_L = "#f9ebec"
SENSORS = [BLUE, TEAL, GREY, PURPLE, ORANGE]
LIGHTS = [BLUE_L, TEAL_L, PALE, PURPLE_L, ORANGE_L]


def canvas(h):
    f = plt.figure(figsize=(14.4, h))
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, h)
    ax.axis("off")
    return f, ax


def txt(ax, x, y, s, size=16, ha="center", va="center", color=INK, weight="normal", **kw):
    kw.setdefault("zorder", 6)
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color,
                   fontweight=weight, **kw)


def box(ax, x, y, w, h, color="white", edge=INK, lw=0.9, radius=.055, z=1, style="solid"):
    p = P.FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0,rounding_size={radius}",
                        facecolor=color, edgecolor=edge, linewidth=lw,
                        linestyle=style, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=INK, lw=1.0, style="solid", scale=13, z=3):
    p = P.FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale,
                          linewidth=lw, color=color, linestyle=style,
                          shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(p)
    return p


def route(ax, points, color=INK, lw=1.0, style="solid", z=3):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, linestyle=style, zorder=z)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, style=style, z=z)


def title(ax, x, y, label, text):
    txt(ax, x, y, f"({label})  {text}", 19, ha="left", weight="bold")


def sensor(ax, x, y, color, missing=False, scale=1):
    w, h = .44 * scale, .53 * scale
    box(ax, x-w/2, y-h/2, w, h, color=PALE if missing else "white",
        edge=GREY if missing else color, radius=.035, lw=1.0)
    ax.add_patch(P.Circle((x, y+.06*scale), .095*scale, facecolor="white",
                          edgecolor=GREY if missing else color, lw=.9))
    for dx in [-.1, 0, .1]:
        ax.plot([x+dx*scale, x+dx*scale], [y-h/2, y-h/2-.1*scale],
                color=GREY if missing else color, lw=.9)
    if missing:
        ax.plot([x-.14*scale, x+.14*scale], [y-.14*scale, y+.14*scale],
                color=GREY, lw=1.1)
    else:
        xx = np.linspace(x-.12*scale, x+.12*scale, 30)
        yy = y-.13*scale+.025*scale*np.sin(np.linspace(0, 3*np.pi, 30))
        ax.plot(xx, yy, color=color, lw=.9)


def matrix_glyph(ax, x, y, color, n=2, cw=.14):
    for r in range(n):
        for c in range(n):
            ax.add_patch(P.Rectangle((x+c*cw, y+r*cw), cw, cw,
                                    facecolor=color, edgecolor="white", lw=.55,
                                    alpha=.24+.18*(r==c)))
    ax.add_patch(P.Rectangle((x, y), n*cw, n*cw, facecolor="none", edgecolor=color, lw=.65))


def joint_ellipse(ax, x, y, w=2.0, h=1.0):
    arrow(ax, (x-w*.48, y-h*.43), (x+w*.5, y-h*.43), lw=.8, scale=8)
    arrow(ax, (x-w*.48, y-h*.43), (x-w*.48, y+h*.5), lw=.8, scale=8)
    for k, alpha in [(1.0,.07),(.68,.13),(.36,.22)]:
        ax.add_patch(P.Ellipse((x, y), w*.66*k, h*.68*k, angle=24,
                              facecolor=TEAL, alpha=alpha, edgecolor=TEAL,
                              lw=.75))
    txt(ax, x+w*.53, y-h*.43, "$y$", 15)
    txt(ax, x-w*.48, y+h*.63, "$h$", 15)


def save(f, name):
    for ext in ["pdf", "svg", "png"]:
        f.savefig(OUT / f"{name}.{ext}", dpi=360, bbox_inches=None, pad_inches=0)
    plt.close(f)


def overview():
    f, ax = canvas(8.6)
    # Calibration matrix with a clearly stated crop, never presented as results.
    title(ax, .3, 8.32, "a", "Deployment calibration")
    box(ax, .3, 5.16, 4.1, 2.87, color="white", edge="#aeb6be")
    txt(ax, 2.35, 7.78, r"Support $\mathcal{D}=\{(x_i,m_i,y_i)\}_{i=1}^{48}$", 17)
    x0, y0, cw, ch = .78, 5.69, .48, .245
    for j in range(6):
        txt(ax, x0+(j+.5)*cw, y0+7.25*ch,
            f"$x_{j+1}$" if j<5 else "$y$", 15)
    miss = {(0,2),(1,0),(2,4),(3,1),(4,2),(5,3),(6,0)}
    shade = [BLUE,TEAL,PURPLE,BLUE,TEAL]
    rng = np.random.default_rng(14)
    for i in range(7):
        for j in range(6):
            c = ORANGE if j==5 else shade[j]
            a = .12 + .31*rng.random()
            if (i,j) in miss:
                c, a = GREY, .32
            ax.add_patch(P.Rectangle((x0+j*cw,y0+(6-i)*ch),cw-.045,ch-.025,
                                    facecolor=c,alpha=a,edgecolor="none"))
            if (i,j) in miss:
                txt(ax,x0+(j+.45)*cw,y0+(6-i+.43)*ch,"·",13,color="#7e8790")
    txt(ax, 3.90, 6.50, r"$\vdots$", 25)
    txt(ax, 2.35, 5.38, "Illustrative cells, 48 rows cropped", 14, color="#59616a")

    title(ax, .3, 4.86, "b", "Query sensor subset")
    box(ax, .3, 2.33, 4.1, 2.27, edge="#aeb6be")
    sx = np.linspace(.88,3.82,5)
    for j,x in enumerate(sx):
        sensor(ax,x,3.91,SENSORS[j],missing=j==2,scale=1.04)
        txt(ax,x,3.37,f"$x_{j+1}$",16,color=SENSORS[j] if j!=2 else "#7c858d")
        box(ax,x-.29,2.91,.58,.29,color=LIGHTS[j],edge=SENSORS[j],radius=.025,lw=.65)
        txt(ax,x,3.055,"?" if j==2 else r"$\tilde{x}$",14,color="#8b939b" if j==2 else SENSORS[j])
    txt(ax,2.35,2.57,r"$S=\{1,2,4,5\}$; sensor 3 is absent",16)

    # The latent state is emphasized, with an actual bivariate Gaussian glyph.
    box(ax,.3,.44,4.1,1.47,color=TEAL_L,edge=TEAL)
    txt(ax,.55,1.65,"Lift the latent state",17,ha="left",weight="bold")
    txt(ax,.55,1.23,r"$z=(y,h)\in\mathbb{R}^{1+K}$",19,ha="left")
    txt(ax,.55,.80,"Target and shared nuisance",14,ha="left")
    joint_ellipse(ax,3.35,1.14,w=1.15,h=.66)

    title(ax,4.95,8.32,"c","Fit the Gaussian anchor")
    box(ax,4.95,6.43,4.2,1.6,color=TEAL_L,edge=TEAL)
    txt(ax,7.05,7.70,"Incomplete support only",17,weight="bold")
    txt(ax,7.05,7.29,"Gaussian EM, then factor analysis",16)
    txt(ax,7.05,6.82,r"$A,\ D,\ \psi,\ \mu_x,\ \mu_y,\ v_y$",21)
    arrow(ax,(4.4,7.17),(4.95,7.17))

    title(ax,4.95,6.08,"d","Choose a learned site route")
    box(ax,4.95,3.28,4.2,2.52,color="#fafafa",edge="#aeb6be")
    box(ax,5.18,4.64,3.74,.90,color=BLUE_L,edge=BLUE)
    txt(ax,7.05,5.28,"Lifted cavity network",18,weight="bold")
    txt(ax,7.05,4.93,r"Shared $f_\theta$; $R=3$ cavity updates",16)
    txt(ax,7.05,4.36,"or",15,color="#657079")
    box(ax,5.18,3.50,3.74,.65,color=PURPLE_L,edge=PURPLE)
    txt(ax,7.05,3.94,"Lifted cavity transformer",17,weight="bold")
    txt(ax,7.05,3.66,"Query cell head; one site update",15)
    arrow(ax,(7.05,6.43),(7.05,5.80))
    arrow(ax,(4.4,5.40),(4.95,5.40))
    arrow(ax,(4.4,3.82),(4.95,3.82))

    title(ax,9.70,8.32,"e","One site per observed sensor")
    box(ax,9.70,3.28,4.4,4.75,edge="#aeb6be")
    txt(ax,11.90,7.72,r"$t_j(z)\;\longleftrightarrow\;(\Lambda_j,\eta_j)$",21)
    rowys=[7.19,6.66,6.13,5.60,5.07]
    for j,y in enumerate(rowys):
        c=SENSORS[j]
        box(ax,10.02,y-.20,3.75,.39,color=LIGHTS[j],edge=c,lw=.7,radius=.025)
        txt(ax,10.22,y,f"{j+1}",15,color=c,weight="bold")
        if j==2:
            txt(ax,11.93,y,"Absent sensor: no factor",15,color="#7c858d")
        else:
            txt(ax,10.95,y,r"$w_j,\ o_j,\ \psi_j$",17,color=c)
            matrix_glyph(ax,12.04,y-.14,c,n=2,cw=.14)
            txt(ax,12.50,y,r"$\Lambda_j$",17,color=c)
            box(ax,13.03,y-.14,.14,.28,color=c,edge=c,lw=.5,radius=0)
            txt(ax,13.42,y,r"$\eta_j$",17,color=c)
    txt(ax,11.90,4.39,r"$\Lambda_j=\dfrac{w_jw_j^{\top}}{\psi_j},\qquad\eta_j=\dfrac{w_jo_j}{\psi_j}$",22)
    txt(ax,11.90,3.62,r"Rank one sites over the same $z=(y,h)$",16)
    arrow(ax,(9.15,4.67),(9.70,4.67))

    title(ax,4.95,2.89,"f","Add natural parameters and marginalize")
    box(ax,4.95,.44,9.15,2.18,color=ORANGE_L,edge=ORANGE)
    txt(ax,7.18,2.20,r"$\Lambda=\Lambda_0+\sum_{j\in S}\Lambda_j$",21)
    txt(ax,7.18,1.64,r"$\eta=\eta_0+\sum_{j\in S}\eta_j$",21)
    txt(ax,7.18,.95,r"$m=\Lambda^{-1}\eta,\quad V=\Lambda^{-1}$",21)
    arrow(ax,(9.30,1.53),(9.90,1.53),color=ORANGE)
    txt(ax,11.93,2.25,r"$q(y)=\mathcal{N}(m_y,V_{yy})$",22)
    xx=np.linspace(-2.9,2.9,250)
    yy=np.exp(-xx**2/2)
    X=10.37+(xx+2.9)/5.8*3.1
    Y=.86+yy*.91
    ax.fill_between(X,.86,Y,color=ORANGE,alpha=.18)
    ax.plot(X,Y,color=ORANGE,lw=1.6)
    arrow(ax,(10.31,.86),(13.60,.86),lw=.8,scale=9)
    txt(ax,13.73,.86,"$y$",17)
    txt(ax,11.93,.62,"Predictive target distribution",15)
    arrow(ax,(11.90,3.28),(11.90,2.62))
    save(f,"architecture_overview")


def cavity():
    f,ax=canvas(8.6)
    col=[.3,5.05,9.8]; w=4.3
    top=4.93; ht=2.98; bot=.58; hb=3.42
    title(ax,.3,8.28,"a","Form the joint cavity")
    title(ax,5.05,8.28,"b","Predict the sensor reading")
    title(ax,9.8,8.28,"c","Shared correction network")
    for x,c in zip(col,[TEAL_L,BLUE_L,PURPLE_L]):
        box(ax,x,top,w,ht,color=c,edge=TEAL if x==col[0] else BLUE if x==col[1] else PURPLE)
    txt(ax,2.45,7.58,r"$q(z)=\mathcal{N}(m,V)$",21)
    joint_ellipse(ax,2.45,6.72,w=2.3,h=.91)
    txt(ax,2.45,5.97,r"$\Lambda_{-j}=\Lambda-\Lambda_j$",21)
    txt(ax,2.45,5.47,r"$\eta_{-j}=\eta-\eta_j$",21)
    txt(ax,2.45,5.14,r"$q_{-j}(z)=\mathcal{N}(m_{-j},V_{-j})$",19)
    arrow(ax,(4.6,6.28),(5.05,6.28))

    # A visual residual, with explicit schematic status, is easier to read than a fabricated scatter.
    txt(ax,7.2,7.58,r"Condition on the other sensors",17,weight="bold")
    ax.plot([5.69,8.72],[6.89,6.89],color="#626e77",lw=.85)
    ax.add_patch(P.Circle((6.54,6.89),.065,color=BLUE))
    ax.add_patch(P.Circle((8.07,6.89),.065,color=RED))
    txt(ax,6.54,7.17,r"$\hat{x}_j$",19,color=BLUE)
    txt(ax,8.07,7.17,r"$\tilde{x}_j$",19,color=RED)
    ax.annotate("",(8.07,6.56),(6.54,6.56),arrowprops=dict(arrowstyle="<->",lw=.9,color=INK))
    txt(ax,7.30,6.28,r"Residual $r_j=\tilde{x}_j-\hat{x}_j$",18)
    txt(ax,7.20,5.88,r"$\hat{x}_j=A_j(m_{-j,y}-\mu_y)+D_j^{\top}m_{-j,h}$",18)
    txt(ax,7.20,5.50,r"$v_j=w_j^{\top}V_{-j}w_j+\psi_j$",21)
    txt(ax,7.20,5.13,r"Use $r_j$, $r_j/\sqrt{v_j}$, and $\log v_j$",15)
    arrow(ax,(9.35,6.28),(9.8,6.28))

    txt(ax,11.95,7.59,r"Support scatter $\rightarrow e_j\in\mathbb{R}^{16}$",18,weight="bold")
    # Small DeepSets scatter encoding inset.
    xx=np.array([0,.14,.28,.42,.56,.71,.83,1.])
    yy=.12+.70*xx+.10*np.array([0,1,-.5,.9,-.9,.3,-.4,.2])
    ax.plot([10.10,11.14],[6.74,6.74],color="#78818a",lw=.7)
    ax.plot([10.10,10.10],[6.74,7.32],color="#78818a",lw=.7)
    ax.scatter(10.14+.90*xx,6.79+.55*yy,s=10,color=PURPLE,alpha=.65,edgecolors="none")
    arrow(ax,(11.25,7.03),(11.58,7.03),color=PURPLE,lw=.8,scale=10)
    box(ax,11.65,6.77,2.11,.48,color="white",edge=PURPLE,radius=.025,lw=.7)
    txt(ax,12.705,7.01,"DeepSets encoder",15)
    txt(ax,11.95,6.50,"Reading, anchor, cavity statistics",16)
    txt(ax,11.95,6.20,"Residuals and observed fraction",16)
    # Shared two hidden layer network, shown as a real architectural glyph.
    for layer,x in enumerate([10.68,11.53,12.38,13.23]):
        ys=np.linspace(5.51,5.94,3 if layer in [0,3] else 4)
        if layer<3:
            y2=np.linspace(5.51,5.94,3 if layer==2 else 4)
            for ya in ys:
                for yb in y2:
                    ax.plot([x,x+.85],[ya,yb],color=PURPLE,alpha=.18,lw=.6)
        for y in ys:
            ax.add_patch(P.Circle((x,y),.055,color="white",ec=PURPLE,lw=.8,zorder=4))
    txt(ax,11.95,5.16,r"$f_\theta:\quad(\delta_j^{\rm off},\delta_j^{\rm slope},\delta_j^D,s_j)$",20)

    title(ax,.3,4.33,"f","Train with target likelihood")
    title(ax,5.05,4.30,"e","Fuse and marginalize")
    title(ax,9.8,4.30,"d","Reconstruct the Gaussian site")
    for x,c,ec in zip(col,[ORANGE_L,TEAL_L,BLUE_L],[ORANGE,TEAL,BLUE]):
        box(ax,x,bot,w,hb,color=c,edge=ec)
    route(ax,[(14.10,5.10),(14.28,5.10),(14.28,3.67),(14.10,3.67)],lw=.9)
    txt(ax,11.95,3.65,r"$\bar y=m_{-j,y}$",19)
    txt(ax,11.95,3.13,r"$w_j=(A_j+\delta_j^{\rm slope},\ D_j+\delta_j^D)$",20)
    txt(ax,11.95,2.62,r"$g_j(\bar y)=A_j(\bar y-\mu_y)+\delta_j^{\rm off}$",19)
    txt(ax,11.95,2.10,r"$o_j=\tilde{x}_j-g_j(\bar y)+w_{j,y}\bar y$",20)
    txt(ax,11.95,1.59,r"$\psi_j=\psi_j^{(0)}\exp\{4\tanh(s_j/4)\}>0$",19)
    txt(ax,11.95,.98,r"$\Lambda_j=w_jw_j^{\top}/\psi_j,\quad\eta_j=w_jo_j/\psi_j$",20)
    arrow(ax,(9.8,2.28),(9.35,2.28))
    txt(ax,7.20,3.62,"Sum the observed replacement sites",17,weight="bold")
    txt(ax,7.20,3.08,r"$\Lambda=\Lambda_0+\sum_{j\in S}\Lambda_j$",21)
    txt(ax,7.20,2.54,r"$\eta=\eta_0+\sum_{j\in S}\eta_j$",21)
    txt(ax,7.20,1.97,r"$m=\Lambda^{-1}\eta,\quad V=\Lambda^{-1}$",20)
    txt(ax,7.20,1.35,r"$q(y)=\mathcal{N}(m_y,V_{yy})$",23)
    txt(ax,7.20,.84,"Absent sensors contribute zero",16)
    arrow(ax,(5.05,2.28),(4.6,2.28))
    txt(ax,2.45,3.61,"After the final refinement round",17,weight="bold")
    txt(ax,2.45,2.96,r"$\mathcal{L}=\mathbb{E}_{\tau,\mathcal{D},x_S,y}[-\log q_\theta(y)]$",21)
    xx=np.linspace(-2.6,2.6,150); yy=np.exp(-xx**2/2)
    xp=.98+(xx+2.6)/5.2*2.8
    yp=1.53+.8*yy
    ax.fill_between(xp,1.53,yp,color=ORANGE,alpha=.19)
    ax.plot(xp,yp,color=ORANGE,lw=1.4)
    ax.plot([2.99,2.99],[1.5,2.25],color=RED,lw=1.05)
    txt(ax,3.04,2.46,"$y$",18,color=RED)
    txt(ax,2.45,1.16,"Zero output head recovers the anchor",15)
    # Loop in the clear interpanel gutter; all sites are updated synchronously.
    route(ax,[(5.05,3.85),(4.80,3.85),(4.80,4.68),(2.45,4.68),(2.45,4.93)],color=TEAL,lw=.9,style="dashed")
    txt(ax,3.75,4.68,r"Repeat $R$ rounds ($R=3$ here)",14,color=TEAL,
        bbox=dict(facecolor="white",edgecolor="none",pad=2.3))
    save(f,"cavity_refinement")


def analytical():
    f,ax=canvas(5.2)
    txt(ax,.3,4.96,"Analytical example: shared noise changes how a sensor should be weighted",19,ha="left",weight="bold")
    title(ax,.3,4.54,"a","Two correlated sensors")
    title(ax,5.05,4.54,"b","Coefficient of the first reading")
    title(ax,9.8,4.54,"c","Lifted sites preserve the nuisance")
    a=np.array([1.,1.]); d=np.array([1.,.5]); sigma=.05
    cov=np.outer(a,a)+np.outer(d,d)+sigma**2*np.eye(2)
    beta=a@np.linalg.inv(cov)
    single=a/(a*a+d*d+sigma**2)
    like_var=d*d+sigma**2
    poe=(a/like_var)/(1+(a*a/like_var).sum())
    for x in [.3,5.05,9.8]:
        box(ax,x,.38,4.3,3.88,color="white",edge="#aeb6be")
    # The exact generative graph.
    for x,y,label,c,fc in [(1.35,3.61,"$y$",ORANGE,ORANGE_L),(3.45,3.61,"$u$",TEAL,TEAL_L),
                            (1.35,1.91,"$x_1$",BLUE,BLUE_L),(3.45,1.91,"$x_2$",PURPLE,PURPLE_L)]:
        ax.add_patch(P.Circle((x,y),.28,fc=fc,ec=c,lw=1.0,zorder=4))
        txt(ax,x,y,label,22)
    for start,end in [((1.35,3.33),(1.35,2.19)),((1.55,3.41),(3.25,2.11)),
                      ((3.25,3.41),(1.55,2.11)),((3.45,3.33),(3.45,2.19))]:
        arrow(ax,start,end,lw=.85)
    txt(ax,.96,2.72,"1",16)
    txt(ax,3.82,2.72,"0.5",16,color=TEAL)
    txt(ax,2.07,3.07,"1",16,color=ORANGE,
        bbox=dict(fc="white",ec="none",pad=1.0))
    txt(ax,2.68,3.07,"1",16,color=TEAL,
        bbox=dict(fc="white",ec="none",pad=1.0))
    txt(ax,2.45,1.31,r"$x_1=y+u+0.05\epsilon_1$",20)
    txt(ax,2.45,.91,r"$x_2=y+0.5u+0.05\epsilon_2$",20)
    txt(ax,2.45,.57,r"Independent $y,u,\epsilon_1,\epsilon_2\sim\mathcal{N}(0,1)$",16)
    # Exact bars and labels; no stochastic sampling.
    plotax=f.add_axes([5.68/14.4,1.69/5.2,3.18/14.4,2.07/5.2])
    values=[single[0],poe[0],beta[0]]
    labs=["Bayes\nfirst sensor", "Scalar sites\nboth sensors", "Bayes\nboth sensors"]
    colors=[BLUE,"#929aa2",ORANGE]
    plotax.axhline(0,color=INK,lw=.7)
    plotax.bar(np.arange(3),values,color=colors,width=.53,alpha=.80,zorder=3)
    for i,v in enumerate(values):
        plotax.text(i,v+(.07 if v>0 else -.07),f"{v:+.4f}",ha="center",va="bottom" if v>0 else "top",fontsize=16,color=colors[i])
    plotax.set_ylim(-1.14,.75)
    plotax.set_yticks([-1,-.5,0,.5]); plotax.tick_params(axis="y",labelsize=13,length=3)
    plotax.set_xticks(np.arange(3),labs,fontsize=13)
    plotax.spines[["top","right"]].set_visible(False)
    plotax.spines["bottom"].set_visible(False)
    plotax.tick_params(axis="x",length=0)
    plotax.set_ylabel(r"Weight on $x_1$",fontsize=15,labelpad=0)
    txt(ax,7.20,.84,r"Pair Bayes mean: $-0.9588x_1+1.9467x_2$",17)
    txt(ax,7.20,.53,"The second sensor makes noise subtraction possible",13.8)

    # Exact likelihood bands and posterior in the lifted latent space.
    ga=f.add_axes([10.42/14.4,1.67/5.2,3.10/14.4,2.07/5.2])
    reading=np.array([.6,.9]); W=np.stack((a,d),axis=1)
    prec=np.eye(2)+W.T@W/sigma**2
    V=np.linalg.inv(prec); m=V@(W.T@reading/sigma**2)
    ys=np.linspace(-.3,2.,400)
    u1=reading[0]-ys; u2=2*(reading[1]-ys)
    ga.fill_between(ys,u1-1.96*sigma,u1+1.96*sigma,color=BLUE,alpha=.14)
    ga.fill_between(ys,u2-1.96*sigma/.5,u2+1.96*sigma/.5,color=PURPLE,alpha=.14)
    ga.plot(ys,u1,color=BLUE,lw=1.0,label=r"$t_1(y,u)$")
    ga.plot(ys,u2,color=PURPLE,lw=1.0,label=r"$t_2(y,u)$")
    vals,vecs=np.linalg.eigh(V); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
    theta=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]))
    for k,alpha in [(np.sqrt(5.991),.10),(np.sqrt(2.279),.20)]:
        ga.add_patch(P.Ellipse(m,2*k*np.sqrt(vals[0]),2*k*np.sqrt(vals[1]),angle=theta,
                              fc=ORANGE,ec=ORANGE,alpha=alpha,lw=1.0,zorder=4))
    ga.scatter([m[0]],[m[1]],s=12,color=ORANGE,zorder=5)
    ga.set_xlim(-.1,1.9); ga.set_ylim(-1.5,1.5)
    ga.set_xticks([0,1]); ga.set_yticks([-1,0,1])
    ga.set_xlabel("Target $y$",fontsize=15,labelpad=1)
    ga.set_ylabel("Nuisance $u$",fontsize=15,labelpad=1)
    ga.tick_params(labelsize=13,length=3)
    ga.spines[["top","right"]].set_visible(False)
    handles=[Line2D([0],[0],color=BLUE,lw=1.2),Line2D([0],[0],color=PURPLE,lw=1.2),P.Patch(fc=ORANGE,alpha=.35)]
    ga.legend(handles,[r"$t_1$",r"$t_2$",r"$q(y,u)$"],loc="upper right",fontsize=12,frameon=True,
              facecolor="white",edgecolor="none",handlelength=1.1,borderpad=.3)
    txt(ax,11.95,3.98,r"Query $x_1=0.6,\ x_2=0.9$",16)
    txt(ax,11.95,.79,r"$w_1=(1,1),\ w_2=(1,0.5),\ \psi_j=0.0025$",17)
    txt(ax,11.95,.51,"Bands: 95% sensor noise; ellipses: Gaussian posterior",13)
    save(f,"shared_noise_example")
    return {"a":a.tolist(),"d":d.tolist(),"sigma":sigma,
            "bayes_singleton_weights":single.tolist(),"bayes_pair_weights":beta.tolist(),
            "scalar_site_pair_weights":poe.tolist(),"illustrative_query":reading.tolist(),
            "joint_posterior_mean":m.tolist(),"joint_posterior_covariance":V.tolist()}


if __name__ == "__main__":
    overview()
    cavity()
    analytic=analytical()
    provenance={
        "source_files":["paper/lifted_cavity/main.tex","paper/lifted_cavity/method_lct.tex",
                        "experiments/lifted_cavity/models.py","experiments/lifted_cavity/lct.py",
                        "experiments/lifted_cavity/anchors.py"],
        "notes":["Calibration cells, query icons and network glyphs are illustrative.",
                 "The analytical figure is computed from the stated Gaussian model, not experimental measurements.",
                 "Cavity updates are synchronous and the site head starts at zero.",
                 "The transformer has one site update, rather than three cavity rounds.",
                 "No sensor permutation claim is made for the transformer.",
                 "Export size is 14.4 inches wide; text is designed for 180 mm reproduction.",
                 "PDF and SVG contain vector shapes; PNG width is 5184 pixels."],
        "analytical_values":analytic,
    }
    (OUT/"figure_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(json.dumps({"figures":["architecture_overview","cavity_refinement","shared_noise_example"],
                      "formats":["pdf","svg","png"],"analytical_values":analytic},indent=2))
