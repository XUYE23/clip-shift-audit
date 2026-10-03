# Evaluation protocol

1. Fix the image encoder, prompt templates, vocabulary, score and temperature.
2. Select a known-only calibration split. Avoid duplicates and near duplicates
   across calibration and test. Keep person/session/source grouping where relevant.
3. Compute score s, where larger values indicate in-distribution.
4. Sort n calibration scores. Set k=floor(alpha*(n+1)).
   Reject s < kth score; if k=0, accept all and report threshold=null.
5. Evaluate once on held-out known and unknown samples.

For exchangeable calibration/test known scores and a fixed scoring function,
strict rejection below the kth order statistic gives marginal false rejection
at most k/(n+1). Ties are conservative. This does not guarantee detection of OOD
samples, individual subgroup coverage, or performance after distribution shift.
Choosing a method using the same test report contaminates final evaluation; use
a separate development split for model/method selection.

AUROC uses known samples as positives and average ranks for equal scores.
FPR@95 uses the lowest observed known score needed to reach at least 95% TPR;
ties may make TPR exceed 95%. It is a ranking metric derived on test labels.
Deployment acceptance metrics use the independent calibration threshold.
Selective accuracy excludes unknown samples; unknown acceptance is separately shown.

The synthetic demo has 6 prototypes, dimension 32, 240 calibration, 240 held-out
known and 160 unknown vectors. Known noise differs between calibration and test
to expose coverage under mild shift. It is a deliberately simple fixture.

For a real release benchmark, publish dataset version/license, known/unknown
class lists, split hashes, checkpoint identifier, prompts, hardware, seed,
dependency versions, raw per-sample scores, and repeated-run uncertainty.
No user resume or private project data belongs in a public benchmark.
