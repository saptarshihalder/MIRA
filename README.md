# MIRA lifted cavity: frozen evaluation panels

This is the data branch for `experiments/lifted_cavity` on branch `lifted-cavity`. Each `.pt` file is one evaluation
panel, holding:
- the tasks or episodes;
- the masks;
- the closed-form reference predictions.

`experiments/lifted_cavity/colab_v4.py` checks every file against `SHA256SUMS` before use. Shipping the panels means
every machine scores the same tasks.

## Synthetic panels

Built with `evaluate2.py panel --seed S --tasks N --sensors P [--nonlin --support]`. They need no data.

| File | Protocol | Tasks | Sensors | Nonlinearity | Support rows |
|---|---|---|---|---|---|
| v2_s20261201 … s20261207 | v2 (F1–F7) | 256 or 128 | 5, 8 or 16 | 0, .4 or .8 | 16, 48 or 96 |
| v2_s20261301, s20261303 | v3 (G1, G3) | 256, 128 | 5, 16 | .4 | 48 |
| v2_s20261401, s20261403 | v4 (H1, H3) | 256, 128 | 5, 16 | .4 | 48 |

## Real panels

Built with `eval_real.py build --dataset D --seed S`. These are the test episodes.

| File | Data | Target |
|---|---|---|
| real_beijing_s2027 | Beijing Multi-Site Air Quality | PM2.5 |
| real_beijing_no2_s3031, real_beijing_co_s3031 | Beijing Multi-Site Air Quality | NO2, CO |
| real_beijing_{pm10,so2,o3}_s4041 | Beijing Multi-Site Air Quality | PM10, SO2, O3 |
| real_airq_{co,no2}_s2027 | Air Quality (Italian city) | CO, NO2 |

The s4041 panels are dequantized (protocol v4, amendment 1).

## Attribution and licence

The real panels are derived from the datasets below and are redistributed under CC BY 4.0. Changes made to the
data:
- log transform;
- episode sampling and masking;
- uniform dequantization of integer readings (s4041 panels only).

Sources:
- S. Chen. *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository, 2017.
  https://doi.org/10.24432/C5RK5G
- S. Vito. *Air Quality* [Dataset]. UCI Machine Learning Repository, 2008. https://doi.org/10.24432/C59K5F
