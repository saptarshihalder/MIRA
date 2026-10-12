# Protocol v12 results: the lifted-site transformer is the best partner for TabPFN (October 13, 2026)

**Source of numbers:** `artifacts/reports/lifted_cavity_paper/runs/confirm_v12.json`, computed by `confirm12.py`.

**Protocol:** fdef392, committed before any of the PFN-L or FA-Gaussian pools existed.

**Data:** the four v10 networks, whose v10/v11 results were already known. The question is about mechanism, not fresh
confirmation. 5,303 episodes where TabPFN v2 is defined, in 112 network-windows.

## Question

Every partner below is pooled with TabPFN v2 by the same per-episode cross-fit stacking (v11). Is UPT-SF a better
partner than the alternatives?

| Partner | What it is |
|---|---|
| PFN-L | Plain transformer: UPT's backbone, size and training, but no lifted sites. |
| FA-Gaussian | Lifted sites in closed form, with no learning. |

## Endpoints (gain = other pool − UPT-SF pool; positive favours the UPT-SF pool)

| ID | Comparison | All sensors | 3 dropped | 6 dropped |
|---|---|---|---|---|
| Y1 | vs PFN-L ⊕ TabPFN | **+0.125** [+0.085, +0.164] pass | **+0.120** [+0.079, +0.161] pass | **+0.085** [+0.054, +0.117] pass |
| Y2 | vs FA-Gaussian ⊕ TabPFN | **+0.097** [+0.078, +0.116] pass | **+0.115** [+0.099, +0.131] pass | **+0.117** [+0.103, +0.131] pass |

**All six endpoints pass.**

## Every pool against TabPFN v2 alone (secondary)

| Pool | All sensors | 3 dropped | 6 dropped |
|---|---|---|---|
| UPT-SF ⊕ TabPFN (v11) | +0.171 | +0.245 | +0.604 |
| PFN-L ⊕ TabPFN | +0.046 | +0.125 | +0.519 |
| FA-Gaussian ⊕ TabPFN | +0.074 | +0.130 | +0.488 |

## Mean NLL on TabPFN-defined episodes (mean stacking weight on the partner)

| Network | Condition | UPT-SF ⊕ | PFN-L ⊕ | FA ⊕ | TabPFN v2 |
|---|---|---|---|---|---|
| USGS NY | all sensors | **−3.699** (.28) | −3.671 (.29) | −3.674 (.15) | −3.634 |
| USGS NY | 6 dropped | **−3.146** (.43) | −3.139 (.40) | −2.980 (.38) | −2.879 |
| USGS WA | all sensors | −5.048 (.17) | **−5.051** (.18) | −5.042 (.12) | −4.971 |
| USGS WA | 6 dropped | −4.571 (.27) | **−4.581** (.23) | −4.523 (.25) | −4.414 |
| DWD pressure | all sensors | **−0.595** (.91) | −0.125 (.88) | −0.332 (.84) | −0.152 |
| DWD pressure | 6 dropped | **−0.228** (.97) | +0.114 (.97) | −0.091 (.96) | +1.462 |
| AURN PM2.5 | all sensors | **−0.561** (.68) | −0.525 (.65) | −0.442 (.41) | −0.432 |
| AURN PM2.5 | 6 dropped | **−0.352** (.79) | −0.322 (.72) | −0.231 (.73) | +0.074 |

## Reading

1. **Pooling alone already helps a lot, especially under sensor loss.** Any reasonable partner pooled with TabPFN v2
   beats TabPFN v2: +0.05 to +0.07 nats with all sensors, and about +0.5 with six sensors dropped. TabPFN's
   predictive degrades with missing inputs, and stacking hands the weight to the partner.
2. **The lifted-site transformer is the best partner, by a significant margin, at every sensor-loss level.**
   - It beats the plain transformer pool by +0.085 to +0.125 nats.
   - It beats the closed-form site pool by +0.097 to +0.117 nats.
   - The margin is largest where the structured, sensor-wise model matters: DWD pressure +0.47 vs PFN-L, AURN PM2.5
     +0.04, New York +0.03.
   - On Washington rivers, where TabPFN carries 80% of the weight, the three partners tie.
3. **Both ingredients of UPT matter.** Learned corrections beat the closed-form sites (Y2), and lifted sites beat the
   plain transformer (Y1).

**Paper claim supported:** combining a foundation model with a structured, missing-sensor-exact predictor by the
linear pool makes it robust to sensor failure. The lifted-site transformer is the partner of choice.
