# Cover Letter — Submission to *Applied Sciences* (MDPI)

**Date:** 13 October 2026

**To:** The Editorial Office, *Applied Sciences*, MDPI

**Manuscript title:** Auditable Anomaly Detection for Unattended Delivery Lockers:
A Calibrated Rule Pipeline with Field-Footage Boundaries and a Pre-Deployment Audit

**Manuscript type:** Article
**Suggested section:** Computing and Artificial Intelligence

Dear Editor,

I am pleased to submit the enclosed manuscript for consideration as an Article in
*Applied Sciences*.

**What the paper does.** Unattended delivery lockers are now common public infrastructure
on campuses and in residential areas, yet their operation is supervised only passively, if
at all: cameras, when present, serve after-the-fact forensics rather than proactive
alerting. The manuscript develops and evaluates an auditable anomaly-detection layer for
such facilities under a deliberately realistic budget — one CPU-only personal computer,
free and open-source components, fully local processing. The system estimates the
open/closed state of every compartment from calibrated image regions, tracks users with
explicit guards against measured detector failure modes, and raises six event types through
per-compartment rules whose thresholds are all named, documented parameters; consequently
every alarm can be audited after the fact, since it names a compartment, a measured
quantity and a threshold.

**Why it fits *Applied Sciences*.** The work is applied in the journal's sense: it
addresses a concrete societal and industrial problem (supervision of the fast-growing
unattended last-mile service infrastructure), builds a complete deployable system rather
than a model in isolation, and reports its engineering envelope as measured quantities —
per-stage CPU costs, achievable frame rates, and the multi-view scaling limit. The
Featured Application field describes the deployment-ready audit tools that accompany the
pipeline.

**Novelty and relation to prior literature.** Modern video-anomaly-detection research is
dominated by learned normality models; this manuscript deliberately takes a different,
complementary position — an auditable, calibration-first rule pipeline for facilities whose
operational concerns are enumerable in advance and must be attributable to a specific
compartment and person. Beyond the system itself, the paper contributes (i) a
field-footage boundary study on 15 real recordings from real locker installations,
including a validated real door-left-open episode; (ii) a two-part pre-deployment audit
procedure (geometric coverage and colour-domain collision) that predicts, before alarms
are enabled, which rules will misfire at a new site; and (iii) an analytic method for
bounding rule-threshold sensitivity from recorded event timelines without re-running any
video. All evidence classes are explicitly separated, and the same-distribution results
are presented as verification of the implementation rather than as a generalization claim.

**Compliance and declarations.** The manuscript is original, has not been published
previously and is not under consideration elsewhere; I am the sole author and approve the
submission. The figures are embedded in the manuscript and also supplied in the separate
compressed file as required, with a resolution of at least 1100 pixels in width (300 dpi),
and a graphical abstract (1100 × 560 px) is provided. All experiments were run on a single
personal computer; no proprietary or restricted data is used. The public datasets employed
(COCO val2017, CUHK Avenue) are cited; the field recordings are publicly available videos
used as analysis inputs only, with their published titles recorded in the evaluation logs.
The use of an AI-based assistant during manuscript preparation is disclosed in the
Acknowledgments section, and I take full responsibility for the content. The author
declares no conflicts of interest.

I hope the manuscript meets the standards of *Applied Sciences* and I look forward to the
review process.

Yours sincerely,

**Jinhui Huang**
College of Computer and Artificial Intelligence, Northeast Forestry University,
Harbin 150040, China
E-mail: 2088239568@qq.com
ORCID: 0009-0004-3070-5092
