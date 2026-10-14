"""Architecture figure of the Universal Pool Transformer for the Neurocomputing manuscript.

Reuses the drawing helpers and palette of draw_figures.py so that all diagrams share one visual style.
The diagram follows experiments/lifted_cavity/upt.py, lct.py and anchors.py (MIRA repository, branch
lifted-site-pool-v5): cell tokens, alternating row and column attention, a lifted site head on observed query cells,
a free head and a gate on the target token, exact fusion of lifted sites and a linear opinion pool. The support cells,
the query and the plotted densities are illustrative, not experimental results.

    python tools/draw_upt_architecture.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.patches as P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_figures import (canvas, txt, box, arrow, route, title, sensor, matrix_glyph, joint_ellipse, save,  # noqa: E402
                          INK, GREY, PALE, BLUE, BLUE_L, TEAL, TEAL_L, PURPLE, PURPLE_L, ORANGE, ORANGE_L,
                          RED, RED_L, SENSORS, LIGHTS)

GREEN = "#4f8a3c"
GREEN_L = "#edf5e9"


def cells(ax, x0, y0, cw=0.46, ch=0.31):
    """Support rows and one query row as a grid of cell tokens (grey = missing)."""
    missing = {(1, 2), (3, 0), (4, 3), (6, 1), (2, 4)}
    nrow = 7
    for r in range(nrow):
        for c in range(5):
            miss = (r, c) in missing
            ax.add_patch(P.Rectangle((x0 + c * cw, y0 - r * ch), cw, ch, facecolor=PALE if miss else LIGHTS[c],
                                     edgecolor="white", lw=1.2, zorder=2))
            if miss:
                txt(ax, x0 + c * cw + cw / 2, y0 - r * ch + ch / 2, "$\\cdot$", 13, color="#8b949c")
        ax.add_patch(P.Rectangle((x0 + 5 * cw + 0.12, y0 - r * ch), cw, ch, facecolor=ORANGE_L, edgecolor="white",
                                 lw=1.2, zorder=2))
    for c in range(5):
        txt(ax, x0 + c * cw + cw / 2, y0 + ch + 0.17, f"$x_{c + 1}$", 15)
    txt(ax, x0 + 5 * cw + 0.12 + cw / 2, y0 + ch + 0.17, "$y$", 15)
    yq = y0 - nrow * ch - 0.33
    qmiss = {2}
    for c in range(5):
        miss = c in qmiss
        ax.add_patch(P.Rectangle((x0 + c * cw, yq), cw, ch, facecolor=PALE if miss else "white",
                                 edgecolor=GREY if miss else SENSORS[c] if c != 2 else GREY, lw=1.3, zorder=2))
        if miss:
            txt(ax, x0 + c * cw + cw / 2, yq + ch / 2, "$\\cdot$", 13, color="#8b949c")
    ax.add_patch(P.Rectangle((x0 + 5 * cw + 0.12, yq), cw, ch, facecolor="white", edgecolor=ORANGE, lw=1.3, zorder=2))
    txt(ax, x0 + 5 * cw + 0.12 + cw / 2, yq + ch / 2, "?", 14, color=ORANGE, weight="bold")
    return yq


def density(ax, x0, y0, w, h, mu, sd, color, lw=1.6, fill=None, alpha=.25, weight=1.0):
    xs = np.linspace(-3.2, 3.2, 300)
    ys = weight * np.exp(-0.5 * ((xs - mu) / sd) ** 2) / sd
    return xs, ys


def main():
    f, ax = canvas(8.2)

    # (a) cells
    title(ax, .3, 7.92, "a", "Support and query as cells")
    box(ax, .3, 3.62, 4.05, 4.0, color="white", edge="#aeb6be")
    yq = cells(ax, .62, 6.86)
    txt(ax, 3.98, 5.85, "48\nsupport\nrows", 12.5, color="#4a535b")
    txt(ax, 3.98, yq + .15, "query", 12.5, color="#4a535b")
    txt(ax, 2.33, 3.86, r"grey: missing cell; sensor 3 is absent from the query", 12, color="#4a535b")

    # (b) backbone
    title(ax, 4.75, 7.92, "b", "Cell token transformer")
    box(ax, 4.75, 3.62, 3.75, 4.0, color="white", edge="#aeb6be")
    box(ax, 5.05, 6.18, 3.15, .95, color=PALE, edge=GREY, radius=.05)
    txt(ax, 6.62, 6.8, "attention across the cells", 14.5)
    txt(ax, 6.62, 6.46, "of each row", 14.5)
    box(ax, 5.05, 4.95, 3.15, .95, color=PALE, edge=GREY, radius=.05)
    txt(ax, 6.62, 5.57, "attention across the rows", 14.5)
    txt(ax, 6.62, 5.23, "of each column", 14.5)
    arrow(ax, (6.62, 6.18), (6.62, 5.93), lw=.9, scale=10)
    txt(ax, 6.62, 4.6, r"$\times\,L$ blocks;  query rows attend", 13.5, color="#4a535b")
    txt(ax, 6.62, 4.27, "to support rows only", 13.5, color="#4a535b")
    txt(ax, 6.62, 3.88, "value or missing embedding + column embedding", 12, color="#4a535b")
    arrow(ax, (4.35, 5.62), (4.75, 5.62), lw=1.1)

    # (c) heads
    title(ax, 8.9, 7.92, "c", "Three heads on the query row")
    box(ax, 8.9, 3.62, 5.2, 4.0, color="white", edge="#aeb6be")
    box(ax, 9.15, 5.9, 4.7, 1.45, color=BLUE_L, edge=BLUE, radius=.05)
    txt(ax, 11.5, 7.07, "Lifted site head (each observed query cell $j$)", 14.5, color=BLUE, weight="bold")
    txt(ax, 11.5, 6.66, r"corrections $\delta^{\rm off}_j,\ \delta^{\rm slope}_j,\ \delta^{D}_j,\ s_j$", 15)
    txt(ax, 11.5, 6.22, r"to the anchor site $\Rightarrow$ $(\Lambda_j,\eta_j)$, zero at initialization", 13.5)
    box(ax, 9.15, 4.92, 4.7, .82, color=ORANGE_L, edge=ORANGE, radius=.05)
    txt(ax, 11.5, 5.48, "Free head (target token)", 14.5, color=ORANGE, weight="bold")
    txt(ax, 11.5, 5.13, r"$p_{\rm free}(y)$: Gaussian or three component mixture", 13.5)
    box(ax, 9.15, 3.88, 4.7, .88, color=GREEN_L, edge=GREEN, radius=.05)
    txt(ax, 11.5, 4.47, "Gate (target token)", 14.5, color=GREEN, weight="bold")
    txt(ax, 11.5, 4.1, r"$\pi=\sigma(g)$, gate bias $+4$ at initialization", 13.5)
    arrow(ax, (8.5, 5.62), (8.9, 5.62), lw=1.1)

    # (d) anchor
    title(ax, .3, 3.2, "d", "Anchor fitted on the support")
    box(ax, .3, .25, 4.05, 2.65, color=TEAL_L, edge=TEAL)
    txt(ax, 2.33, 2.5, "incomplete support only", 14.5, weight="bold")
    txt(ax, 2.33, 2.1, "EM with missing entries, then", 14)
    txt(ax, 2.33, 1.76, "factor analysis over $z=(y,h)$", 14)
    txt(ax, 2.33, 1.28, r"$x_j = w_j^{\top}z + c_j + e_j,\ \ e_j\sim\mathcal{N}(0,\psi_j)$", 15)
    txt(ax, 2.33, .72, "scale free: target standardized", 13, color="#4a535b")
    txt(ax, 2.33, .44, "by the support mean and deviation", 13, color="#4a535b")
    route(ax, [(4.1, 2.9), (4.1, 3.42), (8.62, 3.42), (8.62, 6.45), (9.15, 6.45)], color=TEAL, lw=1.1, style="dashed")

    # (e) exact fusion
    title(ax, 4.75, 3.2, "e", "Exact fusion of lifted sites")
    box(ax, 4.75, .25, 3.75, 2.65, color="white", edge="#aeb6be")
    txt(ax, 6.0, 2.42, r"$\Lambda=\Lambda_0+\sum_{j\in S}\Lambda_j$", 15.5)
    txt(ax, 6.0, 1.92, r"$\eta=\eta_0+\sum_{j\in S}\eta_j$", 15.5)
    txt(ax, 6.0, 1.36, r"$q(z)=\mathcal{N}(\Lambda^{-1}\eta,\Lambda^{-1})$", 15.5)
    txt(ax, 6.0, .78, "a missing sensor", 13, color="#4a535b")
    txt(ax, 6.0, .5, "adds no factor", 13, color="#4a535b")
    joint_ellipse(ax, 7.62, 1.55, w=1.25, h=1.25)
    txt(ax, 7.62, .5, r"marginal $p_{\rm site}(y)$", 13, color="#4a535b")

    # (f) linear pool
    title(ax, 8.9, 3.2, "f", "Linear opinion pool")
    box(ax, 8.9, .25, 5.2, 2.65, color=ORANGE_L, edge=ORANGE)
    txt(ax, 11.5, 2.5, r"$p_{\rm UPT}(y)=\pi\,p_{\rm site}(y)+(1-\pi)\,p_{\rm free}(y)$", 16)
    txt(ax, 11.5, 2.05, r"$-\log p_{\rm UPT}\leq\min\{-\log p_{\rm site}-\log\pi,\ -\log p_{\rm free}-\log(1-\pi)\}$", 13)
    xs = np.linspace(-3.2, 3.2, 300)
    ps = np.exp(-0.5 * ((xs + .5) / .55) ** 2) / .55
    pf = np.exp(-0.5 * ((xs - .9) / .8) ** 2) / .8
    pi = .6
    pu = pi * ps + (1 - pi) * pf
    sx, sy, X0, Y0 = .62, .55, 11.5, .48
    ax.plot(X0 + sx * xs, Y0 + sy * ps, color=BLUE, lw=1.2, zorder=4)
    ax.plot(X0 + sx * xs, Y0 + sy * pf, color=ORANGE, lw=1.2, zorder=4)
    ax.fill_between(X0 + sx * xs, Y0, Y0 + sy * pu, color=INK, alpha=.08, zorder=3)
    ax.plot(X0 + sx * xs, Y0 + sy * pu, color=INK, lw=1.8, zorder=5)
    ax.plot([X0 - sx * 3.2, X0 + sx * 3.2], [Y0, Y0], color=INK, lw=.8)
    txt(ax, X0 - 1.62, 1.16, r"$p_{\rm site}$", 13, color=BLUE)
    txt(ax, X0 + 1.72, 1.02, r"$p_{\rm free}$", 13, color=ORANGE)
    txt(ax, X0 + .3, 1.6, r"$p_{\rm UPT}$", 13)

    # flows from the heads into fusion and pool
    route(ax, [(9.15, 6.0), (8.75, 6.0), (8.75, 3.0), (7.6, 3.0), (7.6, 2.9)], color=BLUE, lw=1.1)
    arrow(ax, (8.5, 1.55), (8.9, 1.55), color=BLUE, lw=1.1)
    route(ax, [(13.85, 5.33), (14.25, 5.33), (14.25, 2.95), (13.5, 2.95), (13.5, 2.9)], color=ORANGE, lw=1.1)
    route(ax, [(13.85, 4.32), (14.05, 4.32), (14.05, 3.1), (12.6, 3.1), (12.6, 2.9)], color=GREEN, lw=1.1)
    save(f, "upt_architecture")


if __name__ == "__main__":
    main()
    print("wrote figures/upt_architecture.{pdf,svg,png}")
