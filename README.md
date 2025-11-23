# Part 1 — THEORETICAL

### Q1 — Define *algorithmic bias* and two examples

**Definition (concise):** algorithmic bias is systematic and repeatable errors in an AI system that produce unfair outcomes for certain groups due to data, model design, objective functions, or deployment context.

**Two concrete examples**

1. **Training-data bias:** A resume-screener trained on historical hiring data dominated by male hires learns to prefer male-stereotyped terms and downgrades resumes mentioning women’s organizations — produces gender-based rejection.
2. **Measurement/label bias:** A recidivism-risk model trained on arrest records where minority groups are over-policed will output higher risk scores for those groups even if underlying re-offense rates are similar, causing disproportionate punitive outcomes. 

---

### Q2 — Transparency vs Explainability (difference & why both matter)

* **Transparency:** openness about the system — access to data provenance, model architecture, training process, objective functions, and intended use. It’s about *what* exists and *how* it was built.
* **Explainability:** the property that lets stakeholders understand *why* a particular decision or prediction occurred (e.g., feature contributions, counterfactuals, local interpretable explanations).

**Why both matter (practical):**

* Transparency enables auditability, reproducibility, and regulatory compliance (you can check whether inputs, preprocessing, and objectives are appropriate).
* Explainability enables actionable recourse for affected individuals (why was I denied?) and operational checks (detect spurious correlations, debugging). Together they enable accountability, trust, and compliance.

---

### Q3 — How GDPR impacts AI development in the EU (core effects)

* **Lawful basis & data minimization:** Controllers must have legal ground for personal data processing (consent, legitimate interest, contract) and minimize data collected. This constrains large, indiscriminate data scrapes for training.
* **Rights to explanation / access / portability:** Individuals can request access to automated-decision logic and obtain copies of personal data used — forces documentation, logging, and explainability efforts.
* **Profiling & high-risk impacts:** Automated profiling affecting rights (credit, employment, legal outcomes) triggers stricter requirements and potentially prior DPIAs (Data Protection Impact Assessments).
* **Accountability & governance:** Organizational obligations — privacy-by-design, DPIAs, records of processing, appoint DPOs where needed — increase compliance engineering overhead.

---

### 2. Ethical Principles Matching (answers only)

A) Justice → **Fair distribution of AI benefits and risks.**
B) Non-maleficence → **Ensuring AI does not harm individuals or society.**
C) Autonomy → **Respecting users’ right to control their data and decisions.**
D) Sustainability → **Designing AI to be environmentally friendly.**

---

# Part 2 — CASE STUDIES (analysis & recommendations)

## Case 1 — Amazon-style biased hiring tool

**Source(s) of bias (likely):**

1. **Historical label bias:** labels (hired/not-hired) reflect past human prejudices (more males hired historically).
2. **Feature construction bias:** textual tokens like “women’s” or graduates of women’s colleges inadvertently become negative signals.
3. **Objective mis-specification:** optimizing predictive performance on biased labels without fairness constraints propagates bias.

**Three concrete fixes**

1. **Data-level remediation — curated balanced training set / reweighting:** remove gendered signals, augment with balanced positive examples, or apply reweighing so protected groups don’t unduly influence loss. (Preprocessing mitigation.)
2. **Model-level constraints — fairness-aware training:** enforce constraints like *equalized odds* or minimize disparate impact during training (in-processing algorithms).
3. **Pipeline & product fixes:** remove protected attributes and proxies from inputs; implement human-in-the-loop review for all automated rejects; require explanations and appeal routes.

**Metrics to evaluate fairness post-correction**

* **Demographic parity / selection rate** (compare acceptance rates across groups). ([Fairlearn][6])
* **False negative rate parity** (candidates wrongly rejected per group).
* **Equalized odds** (TPR and FPR parity across groups). ([Aaron Fraenkel][7])
* **Disparate impact ratio** (ratio of positive selection rates; e.g., 4/5 rule).
* **Calibration within groups / predictive parity** if outcomes labels are reliable.

---

## Case 2 — Facial recognition in policing

**Ethical risks**

* **Wrongful arrests / legal harms:** higher false matches for minorities → risk of arrests, detention, loss of liberty.
* **Privacy and surveillance creep:** large-scale biometric tracking undermines privacy and chills social behavior.
* **Disparate policing & feedback loops:** biased matches drive surveillance hotspots and reinforce biased enforcement (feedback loop).
* **Lack of consent / transparency:** individuals often unaware and cannot opt out.

**Recommended policies for responsible deployment**

1. **Prohibit high-risk uses unless demonstrably safe:** deny or strictly limit use for frontline policing without independent, peer-reviewed bias testing and legal safeguards.
2. **Independent pre-deployment audit + ongoing monitoring:** require third-party audits of accuracy across demographic slices with public reports; continuous operational monitoring.
3. **Human-in-the-loop & strict procedural limits:** no arrests based solely on automated match; treat outputs as investigative leads only.
4. **Access controls, retention limits, and redress:** strict logging, limited retention, clear processes for contesting matches.
5. **Public transparency & consent where feasible:** publish use-cases, performance, and DPIAs.

---


## Part 3: 300-word report

The audit of the COMPAS recidivism scoring dataset reveals measurable racial disparities that risk producing unequal outcomes when used for decision-making. Using IBM’s AI Fairness 360 methodology, I computed group-wise false positive and false negative rates and the disparate impact ratio on held-out validation data. Baseline results confirm prior findings: unprivileged racial groups show substantially higher false positive rates, meaning non-reoffending individuals from those groups are more likely to be labeled “high risk.” This aligns with previously published analyses that identified systematic over-prediction for Black defendants. Such FPR disparity creates the practical danger of disproportionate punitive actions and reinforces biased enforcement practices.

I applied a standard preprocessing mitigation—**Reweighing**—which assigns instance weights to balance the influence of different groups during training. Post-mitigation, FPR and FNR gaps narrowed and the disparate impact moved closer to parity. However, mitigation produced modest reductions in overall classifier accuracy; this trade-off is typical and must be managed by governance decisions about acceptable performance versus fairness.

Recommended remediation steps: (1) **Operational limits** — do not use COMPAS as sole determinant for sentencing or detention decisions; restrict to advisory role with human oversight. (2) **Deploy fairness-aware models** with multi-metric evaluation (demographic parity, equalized odds, calibration-by-group) and choose mitigation strategies (pre/in/post) based on concrete policy objectives. (3) **Transparency and DPIA** — publish model documentation, training data provenance, and conduct Data Protection Impact Assessments. (4) **Continuous monitoring** — instrument live use to detect distributional drift and new disparities.

Finally, technical fixes alone do not eliminate downstream harms: any deployment must pair technical mitigation with legal safeguards, procedural reform, and avenues for affected individuals to contest automated assessments. Continued independent audits and public reporting are necessary to sustain fairness over time.
