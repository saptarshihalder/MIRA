# policy native gpu v1

| Method | Mean NLL |
|---|---:|
| pretrained_fixed | 0.318891 |
| native_finetuned | 0.311691 |
| native_scratch | 0.307248 |
| native_linear | 0.307697 |
| frozen | 0.313739 |
| moment | 0.333232 |
| cv_select | 0.329009 |
| native_guard | 0.317943 |

Actual Tesla T4 run: XGBoost source training and three correction models on CUDA.
The fine-tuned primary, scratch neural and linear scorers received 400 updates each.
Native source/meta-training/validation/development patients are disjoint. The source backbone
uses 4,096 labels; fixed source context128; native episodes32 training/16 validation/32 development.
Each episode has64 labeled target support and256 queries. Final epochs are fixed;
only validation labels select the operational guard threshold.

Primary fine-tuning loses to scratch and linear models. Descriptive scratch gain over frozen
is .006491 [.002546,.010437], but its gain over linear is .000449 [-.000388,.001285].
Synthetic pretraining and a distinctive nonlinear advantage are not established.
Intervals condition on this single dataset, trained source backbone and initialization.

Independent audit passes: all256 probability arrays, patient boundaries and hashes.
NLL discrepancy <=2.98e-8; interval discrepancy <=1.61e-9; NumPy scorer <=1.79e-7.
Preserve all14 warnings: one XGBoost host-device prediction fallback and13 low-class-count CV warnings.
Do not claim every inference kernel ran on CUDA. No verified hospital/date split or clinical claim.

Source: UCI Diabetes130 (CC BY4.0), https://doi.org/10.24432/C5230J.
Official download hashes/token audit: artifacts/manifests/diabetes_native_source.json.
A proposed70/15/15 preparation split was not adopted; executed protocol uses60/20/20.

Frozen protocol commit68f5dcd; Modal app ap-yd9YbpSmYeKwFYgpgepUj5.
One successful remote invocation,16.877 wrapper seconds. Pre-dispatch Windows console failure
is retained within the same$.50 reservation. No remote retry. Total reservations11.05/26;
3 reproduction reserve protected. Provider display rounds to$.00; final invoice unverified.
Runtime active-compute estimate ~$.00351 excludes startup/build/other charges.

An earlier CPU native screen failed at a baseline optimizer limit after2 completed episodes;
its partial outputs/failure manifest remain. This GPU screen uses separately frozen controls.
