# Credit Pre-Screening Engine — Loan Intake Optimisation

**Calibo AI Academy · Path 2 (KLU Batch) · Phase 2, Mini Use Case 02**
Module: Supervised Learning Part 1 (Logistic Regression) · Industry: BFSI — Retail Lending (Personal Loans)

**Team Batch_15**

| Member | Student ID | Role | Sections owned |
|---|---|---|---|
| Velagapudi Haritha | 2300030248 | Data Analyst |  Steps 5.1–5.3 |
| Kunapuli Bala Sri Sarvani | 2300030957 | Model Builder | CBIM Canvas, Steps 5.4–5.5 |
| Yaswanth Kumar Reddy Vogireddy | 2300031126 | Policy Synthesiser | Steps 5.6–5.7 |

🔗 **Live app:** [Credit Pre-Screening Engine on Streamlit](https://batch1527-10-2026muc02creditprescreening-kz2zmiweaufaz6wcwfscy.streamlit.app/)

---

## The business question

A mid-size private bank receives about **3,000 personal loan applications a month**. Every one goes through a manual credit review that takes an analyst about **25 minutes**, and many declines are obvious within the first minute.

> Using six months of historical decisions, can a pre-screening model safely remove clear-cut declines from the manual review queue? How many applications and analyst-hours would that free up each month, without wrongly turning away creditworthy customers?

The bank's risk appetite: **no more than 5% of auto-declined applications may be ones the credit team would have approved.**

---

## Repository contents

| File | What it is |
|---|---|
| `Batch_15_MUC02_KLU_Notebook.ipynb` | The main analysis: CBIM canvas and Steps 5.1–5.7, with code and plain-English markdown for each step. Every figure in the summary and reflections comes from this notebook |
| `MUC02_Credit_PreScreening_Dataset 1.csv` | Simulated dataset: 12,000 applications, April–September 2026, 19 columns |
| `Batch_15_MUC02_KLU_Summary.docx` | One-page team insight summary: findings, recommendation, limitation, section ownership |
| `Batch_15_MUC02_KLU_Presentation.pptx` | Presentation deck (15 slides), including additional analyses beyond the brief |
| `Batch_15_Reflection_Haritha_2300030248.docx` | Individual reflection: Data Analyst |
| `Batch_15_Reflection_Sarvani_2300030957.docx` | Individual reflection: Model Builder |
| `Batch_15_Reflection_Yaswanth_2300031126.docx` | Individual reflection: Policy Synthesiser |
| `credit_score_bands.png` | Approval rate by credit score band (Step 5.2) |
| `employment_type_approval.png` | Approval rate by employment type (Step 5.2) |
| `foir_bands.png` | Approval rate by FOIR band (Step 5.3) |
| `coefficients.png` | Five largest logistic regression coefficients (Step 5.4) |
| `threshold_table.png` | Auto-decline coverage vs wrongful-decline risk by threshold (Step 5.5) |

The notebook generates the PNG charts when it runs.

---

## How to run

1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Put the notebook and the dataset CSV in the same folder. The notebook loads the file as:
   ```python
   df = pd.read_csv("MUC02_Credit_PreScreening_Dataset 1.csv")
   ```
3. Start Jupyter from that folder and open the notebook:
   ```bash
   jupyter notebook
   ```
4. Use **Kernel → Restart & Run All**. The notebook runs end-to-end and saves the five charts to the same folder.

The model uses an 80/20 split with `stratify=y` and `random_state=42`, so results are reproducible and comparable across the batch.

---

## Interactive Streamlit app

**[Open the live app →](https://batch1527-10-2026muc02creditprescreening-kz2zmiweaufaz6wcwfscy.streamlit.app/)**

The brief asks only for a notebook and a summary. As an extra, the team also built a Streamlit dashboard and deployed it on Streamlit Community Cloud, so the Credit Policy Analyst can explore the analysis and test applicants without opening Jupyter.

| Page | What it does |
|---|---|
|  **Overview** | Headline numbers (12,000 applications, 52.9% approval rate, 0.90 recommended threshold, ~190 analyst-hours freed a month), the business context, a threshold summary and the overall decision split |
|  **EDA & Insights** | Four tabs: **Credit Score** (approval by score band, the 650 cliff and new-to-credit applicants), **FOIR & Affordability** (approval by FOIR band, FOIR and income distributions by decision), **Employment & Demographics** (approval by employment type and gender, monthly application volume) and **Correlations** (feature heatmap against the Declined target) |
|  **Model Comparison** | ROC curves for Logistic Regression, Random Forest, XGBoost and LightGBM; a confusion matrix for each model; and a threshold chart against the 5% risk limit |
|  **Live Prediction** | Enter an applicant's details and get an instant pre-screening result |
|  **Credit Policy Briefing** | The briefing to the Credit Policy Analyst: recommended threshold, monthly impact and the two strongest drivers |

**Live Prediction inputs:** age, monthly income, credit score (0 for new-to-credit), existing EMI, proposed EMI, employment type, loan purpose, loan amount, tenure, late payments in the last 24 months, active loans, years in current job and city tier.

**Model comparison (ROC-AUC):**

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.867 |
| Random Forest | 0.876 |
| XGBoost | 0.867 |
| LightGBM | 0.879 |

The tree-based models add only about one point of AUC. The submitted analysis and the recommended threshold use **logistic regression**, as the brief specifies, because its coefficients can be explained to a credit analyst and defended to the Head of Retail Lending. The other models are shown for comparison only.

---

## CBIM Problem Canvas

| Field | Our answer |
|---|---|
| **Situation** | A mid-size retail bank receives ~3,000 personal loan applications a month, each reviewed manually for ~25 minutes |
| **Complication** | Analyst time goes on clear-cut declines while strong applicants wait longer in the same queue, and the bank cannot keep scaling by hiring more analysts |
| **Question** | Which applications can be safely removed from manual review, and at what confidence threshold, without wrongly auto-declining more than 5% of applicants the credit team would have approved? |
| **SSOT** | `MUC02_Credit_PreScreening_Dataset.csv`. The `decision` column records the credit team's past judgement, not loan repayment, so the model learns to imitate analysts, including any bias in their decisions |
| **Success definition** | A single auto-decline threshold within the 5% risk limit, with a stated number of applications and analyst-hours freed per month |
| **Primary stakeholder** | The Credit Policy Analyst, and downstream the analysts whose workload changes, the Head of Retail Lending, and the applicants affected |

---

## Approach and key findings

### Step 5.1 — Load, inspect and clean
*Velagapudi Haritha*

Overall approval rate: **52.9%**. The target is reasonably balanced. We kept all **12,000 rows** and dropped none.

| Issue | What we found | What we did |
|---|---|---|
| `monthly_income` missing (441 rows) | Concentrated in Business Owner (13.0%) and Self-Employed Professional (10.8%) applicants | Filled with the median income for that employment type |
| `credit_score` blank (1,039 rows, 8.7%) | Every one has `credit_history_months = 0`. These are new-to-credit (NTC) applicants, not errors | Kept and flagged with `is_ntc`; median-filled for modelling only |
| `monthly_income` above ₹6,00,000 (12 rows) | All Salaried; annual income entered as monthly | Divided by 12 |

### Step 5.2 — What drives approval today?
*Velagapudi Haritha*

| Credit score band | <600 | 600–649 | 650–699 | 700–749 | 750–799 | 800+ | No score |
|---|---|---|---|---|---|---|---|
| Approval rate | 2.0% | 3.2% | 32.5% | 54.4% | 77.2% | 85.2% | 53.9% |

- The bank has an **unwritten credit score floor at 650**. Below it, applications are almost never approved.
- New-to-credit applicants are approved **53.9%** of the time, which shows the credit team is using judgement for this group.
- **Gig / Contract** applicants are approved least (**34.9%**), against **58.4%** for Salaried applicants.

![Approval by credit score band](credit_score_bands.png)

### Step 5.3 — Affordability features
*Velagapudi Haritha*

- `foir = (existing_emi + proposed_emi) / monthly_income`
- `loan_to_income = loan_amount / (monthly_income × 12)`
- `is_ntc = 1` if the applicant has no credit score

| FOIR band | <30% | 30–40% | 40–50% | 50–55% | >55% |
|---|---|---|---|---|---|
| Approval rate | 65.4% | 59.0% | 47.4% | 31.7% | 15.5% |

Approvals collapse once EMIs take more than about half of income. We propose a **FOIR limit of 50–55%** in credit policy.

![Approval by FOIR band](foir_bands.png)

### Step 5.4 — Logistic regression pre-screener
*Kunapuli Bala Sri Sarvani*

- **Target:** Declined = 1, because the model predicts declines.
- **Features:** credit score, NTC flag, FOIR, loan-to-income, delinquencies (24 months), active loans, years in job, age, tenure, city tier, employment type and loan purpose.
- **Excluded:** gender, city, application ID, application date and decision.
- **Preprocessing:** categorical features one-hot encoded and numeric features standardised.

| Metric (test set, n = 2,400, threshold 0.5) | Value |
|---|---|
| Accuracy | 79.1% |
| ROC-AUC | 0.867 |
| Precision (Declined) | 79.1% |
| Recall (Declined) | 75.5% |
| Confusion matrix | TN 1046 · FP 225 · FN 277 · TP 852 |
| Accuracy, scored vs new-to-credit applicants | 81% / 57% |

At the default 0.5 cut-off, **225 applicants the credit team actually approved would be wrongly declined.** That is too costly to automate.

**Five largest coefficients** (standardised; positive pushes towards decline):

| Feature | Coefficient |
|---|---|
| FOIR | +1.22 |
| Credit score | −1.01 |
| Delinquencies (24m) | +0.98 |
| New-to-credit flag | +0.38 |
| Gig / Contract | +0.23 |

The model independently learned the same policy we found by hand in Steps 5.2 and 5.3.

![Top coefficients](coefficients.png)

### Step 5.5 — Auto-decline threshold
*Kunapuli Bala Sri Sarvani*

| Threshold | % auto-declined | % wrongly declined |
|---|---|---|
| 0.50 | 44.9% | 20.9% |
| 0.60 | 37.6% | 16.9% |
| 0.70 | 30.8% | 12.1% |
| 0.80 | 23.4% | 7.1% |
| 0.85 | 19.8% | 6.1% |
| **0.90** | **15.2%** | **4.7%** |
| 0.95 | 10.0% | 2.1% |

**Chosen threshold: 0.90.** It is the lowest threshold that keeps wrongful declines under the bank's 5% limit. At 0.85 the rate is 6.1%, which fails the rule.

| Monthly business impact | |
|---|---|
| Applications auto-declined | ~456 of 3,000 |
| Analyst-hours freed | ~190 (at 25 minutes each) |
| Auto-declines that were good customers | 4.7% |

![Threshold trade-off](threshold_table.png)

### Step 5.6 — Fairness check
*Yaswanth Kumar Reddy Vogireddy*

| | Male | Female |
|---|---|---|
| Historical approval rate | 54.1% | 50.4% |
| Gig / Contract only | 35.2% | 34.6% |
| Business Owner only | 51.1% | 51.6% |
| Median monthly income | ₹55,000 | ₹45,000 |
| Share who are Gig / Contract | 11.6% | 21.4% |
| Model auto-decline rate at 0.90 | 15.0% | 15.6% |

- The raw 3.7-point gap **almost disappears within employment types**. It is explained mostly by income and employment mix, not gender itself.
- At 0.90, the model **does not widen the gap**.
- Gender should never be a credit feature, but leaving it out is **necessary, not sufficient**. Income and employment type can act as proxies, so auto-decline rates by gender should be monitored after launch.

### Step 5.7 — Credit Policy Briefing
*Yaswanth Kumar Reddy Vogireddy*

1. **Threshold:** auto-decline only at a model probability of 0.90 or higher. This removes 15.2% of applications, about 456 a month or 190 analyst-hours, with 4.7% wrongly declined (inside the 5% limit).
2. **Two biggest drivers:** FOIR (approval falls from 65.4% below 30% to 15.5% above 55%) and credit score (a hard floor near 650).
3. **Keep humans on new-to-credit applicants:** NTC applicants are 8.7% of volume, and the model's accuracy drops from 81% for scored applicants to 57% for them.
4. **Fairness position:** gender is excluded, but income and employment type are proxies. Track auto-decline rates by gender on an ongoing basis, not as a one-time check.

---

## Beyond the brief (presentation slides 7, 9 and 12)

**Four patterns we looked for ourselves:**
- **City tier is a red herring.** Median income falls 67% from Tier 1 to Tier 3 (₹62.5K to ₹37.5K), yet approval stays flat at about 53%. FOIR already absorbs the cost-of-living difference.
- **No drift over six months.** Monthly approval stays within 51.8–54.2% from April to September, so the credit policy was stable across the training window.
- **Debt Consolidation stands alone.** It is the least-approved loan purpose (45.0%, against 57.1% for Education) and carries the highest median FOIR (0.371).
- **Young applicants are the NTC group.** 22.6% of 18–25 year-olds are new-to-credit, against about 0% at 56+. Approval climbs with age (48.8% to 70.0%).

**The credit score "grey zone"** (approval rate by credit score and FOIR together):

| Credit score | FOIR <40% | FOIR 40–55% | FOIR >55% |
|---|---|---|---|
| <650 (floor) | 4.3% | 0.5% | 0.0% |
| 650–749 (grey zone) | 58.1% | 30.9% | 5.9% |
| 750+ | 93.0% | 74.9% | 31.8% |

Within the 650–749 band, approval swings 52 points on FOIR alone. A credit-score-only rule would get this segment most wrong.

**Who bears the residual risk.** At 0.90, 17 of 2,400 test applicants are wrongly auto-declined:
- 29.4% are Gig / Contract workers, against 14.7% of the test set.
- 35.3% are new-to-credit, against about 8.7% of the test set.
- 41.2% are Debt Consolidation loans, against about 20% of the test set.
- Their records are clean: median credit score 682 and zero delinquencies. High FOIR (median 0.557) is the main flag against them.

We suggest routing auto-declines with a clean history but high FOIR to a light secondary human check.

---

## Recommendations

- Adopt a **0.90 auto-decline threshold**: ~456 applications and ~190 analyst-hours freed per month.
- Formalise a **FOIR cap of 50–55%** and a **650 credit score floor** in credit policy.
- Keep **all new-to-credit applicants in manual review**.
- **Monitor auto-decline rates by gender** after launch.

## Limitation

The model is trained on the credit team's **past decisions, not actual loan repayment**. It learns to copy historical judgement, including any bias in it. It should be used only to automate the easy, high-confidence end of manual review, never as a general-purpose risk score.

## Tools

- **Analysis:** Python · Jupyter Notebook · pandas · NumPy · Matplotlib · Seaborn · scikit-learn
- **App:** Streamlit (deployed on Streamlit Community Cloud), with Random Forest, XGBoost and LightGBM for the model comparison
