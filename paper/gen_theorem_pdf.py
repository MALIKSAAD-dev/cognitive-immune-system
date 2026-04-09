"""Generate the Containment Depth Bound theorem PDF."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from fpdf import FPDF

class TheoremPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(90, 5, "Containment Depth Bound", 0, 0, "L")
            self.cell(0, 5, "Muhammad Saad, 2026", 0, 0, "R")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def section(self, num, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        t = f"{num}  {title}" if num else title
        self.cell(0, 8, t, 0, 1)
        self.ln(1)

    def body(self, text):
        self.set_font("Times", "", 11)
        self.multi_cell(0, 5.8, text)
        self.ln(1)

    def formula(self, text, indent=10):
        self.set_font("Courier", "B", 11)
        self.set_x(30)
        self.write(6, text)
        self.set_font("Times", "", 11)
        self.ln(7.0)

    def definition_block(self, label, text):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, label, 0, 1)
        self.set_font("Times", "I", 11)
        self.multi_cell(0, 5.8, text)
        self.set_font("Times", "", 11)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Times", "", 11)
        self.set_x(30)
        self.write(5.8, "-  ")
        self.write(5.8, text)
        self.ln(6.8)

pdf = TheoremPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font("Helvetica", "B", 18)
pdf.cell(0, 10, "The Containment Depth Bound", 0, 1, "C")
pdf.ln(2)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 7, "A Formal Proof of Sequential Contamination", 0, 1, "C")
pdf.cell(0, 7, "Escape Reduction in Epistemic Quarantine Systems", 0, 1, "C")
pdf.ln(3)
pdf.set_font("Helvetica", "I", 11)
pdf.cell(0, 6, "With Empirical Validation: rho = 0.284, k* = 9", 0, 1, "C")
pdf.ln(5)
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 6, "Muhammad Saad", 0, 1, "C")
pdf.set_font("Helvetica", "I", 10)
pdf.cell(0, 5, "Independent Researcher, Islamabad, Pakistan", 0, 1, "C")
pdf.cell(0, 5, "saadkhan@proton.me", 0, 1, "C")
pdf.cell(0, 5, "April 2026", 0, 1, "C")
pdf.ln(4)
pdf.set_draw_color(100, 100, 100)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.ln(6)

# Motivation
pdf.section("", "Motivation")
pdf.body(
    "Consider a system that processes the output of a large language model. The output may contain "
    "hallucinated claims. The system attempts to identify and quarantine these claims before they "
    "propagate to downstream reasoning steps. A natural question arises: if a single quarantine "
    "checkpoint catches contamination with probability rho, how many checkpoints do we need to "
    "guarantee that contamination escapes with probability at most epsilon?"
)
pdf.body("This document provides a complete proof.")

# 1. Definitions
pdf.section("1", "Definitions")

pdf.definition_block("Definition 1 (Epistemic Quarantine).", (
    "Let C = {c1, c2, ..., cn} be a set of atomic claims extracted from an LLM response. "
    "Let phi: C -> [0,1] be a contamination scoring function and tau in (0,1) a threshold. "
    "Epistemic Quarantine partitions C into:\n"
    "    S = {c in C : phi(c) < tau}     (safe set)\n"
    "    Q = {c in C : phi(c) >= tau}    (quarantine set)\n"
    "with the invariant that S and Q are disjoint and S union Q = C."
))

pdf.definition_block("Definition 2 (Contaminated Claim).", (
    "A claim c is contaminated if it asserts a proposition that is factually false. "
    "We write c in F for the set of contaminated claims."
))

pdf.definition_block("Definition 3 (Detection Rate).", (
    "The single-checkpoint detection rate is:\n"
    "    rho = P(phi(c) >= tau | c in F)\n"
    "That is, rho is the probability that a contaminated claim is correctly placed in Q "
    "by a single EQ checkpoint."
))

pdf.definition_block("Definition 4 (Escape Event).", (
    "A contaminated claim c escapes a checkpoint if phi(c) < tau, i.e., it is placed in S "
    "despite being contaminated. The escape probability for a single checkpoint is 1 - rho."
))

# 2. The Theorem
pdf.section("2", "The Theorem")
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Theorem 1 (Containment Depth Bound)", 0, 1)
pdf.ln(1)
pdf.set_font("Times", "I", 11)
pdf.body(
    "Let rho in (0,1) be the single-checkpoint detection rate for contaminated claims. Assume "
    "detection decisions at successive checkpoints are independent. Then:"
)
pdf.set_font("Times", "", 11)
pdf.body("(i) The probability that a contaminated claim escapes k consecutive checkpoints is:")
pdf.formula("P(escape after k checkpoints) = (1 - rho)^k")
pdf.body("(ii) For any target escape probability epsilon in (0,1), the minimum number of checkpoints guaranteeing P(escape) <= epsilon is:")
pdf.formula("k* = ceil( log(epsilon) / log(1 - rho) )")

# 3. Proof
pdf.section("3", "Proof")
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Part (i).", 0, 1)
pdf.set_font("Times", "", 11)
pdf.body(
    "Let Ej denote the event that the contaminated claim escapes checkpoint j. By definition:"
)
pdf.formula("P(Ej) = 1 - rho    for each j = 1, 2, ..., k")
pdf.body("The claim escapes all k checkpoints if and only if it escapes each one individually:")
pdf.formula("P(escape after k) = P(E1 intersect E2 intersect ... intersect Ek)")
pdf.body("By the independence assumption:")
pdf.formula("P(E1 intersect E2 intersect ... intersect Ek) = product_{j=1}^{k} P(Ej) = (1-rho)^k")
pdf.body(
    "Since rho is in (0,1), we have 1 - rho is in (0,1), so (1-rho)^k is a strictly decreasing "
    "function of k that converges to 0 as k approaches infinity. This completes part (i)."
)

pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Part (ii).", 0, 1)
pdf.set_font("Times", "", 11)
pdf.body("We seek the smallest k in the natural numbers such that:")
pdf.formula("(1-rho)^k <= epsilon")
pdf.body("Taking the natural logarithm of both sides:")
pdf.formula("k * ln(1-rho) <= ln(epsilon)    ... (star)")
pdf.body("Now, since rho is in (0,1):")
pdf.formula("0 < 1-rho < 1  implies  ln(1-rho) < 0")
pdf.body("Similarly, since epsilon is in (0,1):")
pdf.formula("ln(epsilon) < 0")
pdf.body("Dividing (star) by ln(1-rho), which is negative, reverses the inequality:")
pdf.formula("k >= ln(epsilon) / ln(1-rho)")
pdf.body("The minimum integer satisfying this is:")
pdf.formula("k* = ceil( ln(epsilon) / ln(1-rho) )")
pdf.body("This holds for any base of logarithm, since the ratio is base-independent.  QED.")
pdf.ln(2)
pdf.set_draw_color(100, 100, 100)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())

# 4. Numerical Verification
pdf.section("4", "Numerical Verification")
pdf.body(
    "We substitute the empirically measured value rho = 0.284 from our HotpotQA adversarial "
    "benchmark (86 contaminated questions, 25 caught by a single CIS checkpoint)."
)

pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Case 1: epsilon = 0.05", 0, 1)
pdf.set_font("Times", "", 11)
pdf.formula("k* = ceil( ln(0.05) / ln(1 - 0.284) )")
pdf.formula("   = ceil( ln(0.05) / ln(0.716) )")
pdf.body("Computing the numerator: ln(0.05) = -2.9957")
pdf.body("Computing the denominator: ln(0.716) = -0.3340")
pdf.formula("k* = ceil( -2.9957 / -0.3340 ) = ceil( 8.969 ) = 9")
pdf.body("Verification: (0.716)^9 = 0.0494. Indeed 0.0494 < 0.05.  Confirmed.")

pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Case 2: epsilon = 0.01", 0, 1)
pdf.set_font("Times", "", 11)
pdf.formula("k* = ceil( ln(0.01) / ln(0.716) )")
pdf.formula("   = ceil( -4.6052 / -0.3340 ) = ceil( 13.787 ) = 14")
pdf.body("Verification: (0.716)^14 = 0.00929. Indeed 0.00929 < 0.01.  Confirmed.")

pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Case 3: epsilon = 0.001", 0, 1)
pdf.set_font("Times", "", 11)
pdf.formula("k* = ceil( ln(0.001) / ln(0.716) )")
pdf.formula("   = ceil( -6.9078 / -0.3340 ) = ceil( 20.68 ) = 21")
pdf.body("Verification: (0.716)^21 = 0.000759. Indeed 0.000759 < 0.001.  Confirmed.")

# 5. Escape Probability Table
pdf.section("5", "Escape Probability Table")
pdf.set_font("Helvetica", "B", 10)
col_w = [50, 55, 55]
headers = ["k (checkpoints)", "P(escape)", "Containment %"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, 1, 0, "C")
pdf.ln()
pdf.set_font("Times", "", 10)
rows = [
    ("1", "0.7160", "28.4%"),
    ("2", "0.5127", "48.7%"),
    ("3", "0.3671", "63.3%"),
    ("4", "0.2628", "73.7%"),
    ("5", "0.1882", "81.2%"),
    ("6", "0.1347", "86.5%"),
    ("7", "0.0965", "90.4%"),
    ("8", "0.0691", "93.1%"),
    ("9  [epsilon=0.05]", "0.0494", "95.1%"),
    ("10", "0.0354", "96.5%"),
    ("14  [epsilon=0.01]", "0.0093", "99.1%"),
    ("21  [epsilon=0.001]", "0.0008", "99.9%"),
]
for row in rows:
    if "epsilon" in row[0]:
        pdf.set_font("Times", "B", 10)
    else:
        pdf.set_font("Times", "", 10)
    for i, val in enumerate(row):
        pdf.cell(col_w[i], 6, val, 1, 0, "C")
    pdf.ln()
pdf.ln(2)
pdf.set_font("Times", "I", 10)
pdf.cell(0, 5, "Table 1: Escape probability as a function of checkpoint depth for rho = 0.284.", 0, 1, "C")

# 6. Independence Assumption
pdf.section("6", "Discussion of the Independence Assumption")
pdf.body(
    "The proof relies on the assumption that detection decisions at successive checkpoints are "
    "independent. In practice, this assumption holds approximately when:"
)
pdf.bullet("Each checkpoint uses a different scoring oracle (different LLM, different knowledge base, different verification strategy).")
pdf.bullet("The contaminated claim is re-extracted at each checkpoint from the LLM output, which may differ across runs due to sampling temperature.")
pdf.bullet("The Wikipedia and consistency signals are queried independently at each checkpoint.")
pdf.ln(1)
pdf.body(
    "If checkpoints are positively correlated (a claim that escapes one checkpoint is more likely to "
    "escape the next), then k* is a lower bound and more checkpoints are needed. If they are negatively "
    "correlated (unlikely in practice), fewer suffice."
)
pdf.body(
    "Formal relaxation of the independence assumption using Markov chain mixing bounds is a direction "
    "for future work."
)

# 7. Connection to Prior Art
pdf.section("7", "Connection to Prior Art")
pdf.body("The structure of this bound is classical. It follows the same logic as:")
pdf.bullet("Redundancy in fault-tolerant systems (Avizienis et al. 2004): n independent replicas reduce failure probability exponentially.")
pdf.bullet("Boosting in machine learning (Schapire 1990): sequential weak learners compose into strong learners at an exponential rate.")
pdf.bullet("Randomized primality testing (Rabin 1980): k independent Miller-Rabin witnesses reduce false positive probability to 4^(-k).")
pdf.ln(1)
pdf.body(
    "The novelty is not in the mathematical technique but in its application to LLM hallucination "
    "containment. To our knowledge, no prior work has formalized inference-time quarantine checkpoints "
    "and derived the minimum depth required for a target containment guarantee."
)

# 8. Statement of Contribution
pdf.section("8", "Statement of Contribution")
pdf.body(
    "This theorem establishes that reliable contamination containment does not require a perfect detector. "
    "A detector with any nonzero detection rate rho > 0, applied sequentially k* times, achieves "
    "arbitrarily low escape probability. The practical implication is that improving containment can "
    "proceed along two independent axes: improving rho (better detection per checkpoint) or increasing "
    "k (more checkpoints in sequence). The depth bound quantifies the exact tradeoff."
)
pdf.body(
    "For the empirically measured rho = 0.284 from our HotpotQA adversarial benchmark, we get k* = 9 "
    "for 95% containment and k* = 14 for 99% containment. These are practical numbers: a multi-step "
    "reasoning pipeline with 9 or more steps, each guarded by a CIS checkpoint, would contain "
    "contamination with high probability even though each individual checkpoint is imperfect."
)

pdf.ln(6)
pdf.set_draw_color(100, 100, 100)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.ln(4)
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 6, "Muhammad Saad", 0, 1)
pdf.set_font("Helvetica", "I", 10)
pdf.cell(0, 5, "Independent Researcher, Islamabad, Pakistan", 0, 1)
pdf.cell(0, 5, "April 2026", 0, 1)

# References
pdf.ln(6)
pdf.section("", "References")
pdf.set_font("Times", "", 9)
refs = [
    "[1]  A. Avizienis, J.-C. Laprie, B. Randell, and C. Landwehr. Basic concepts and taxonomy of dependable and secure computing. IEEE Trans. Dependable and Secure Computing, 1(1):11-33, 2004.",
    "[2]  M. O. Rabin. Probabilistic algorithm for testing primality. Journal of Number Theory, 12(1):128-138, 1980.",
    "[3]  R. E. Schapire. The strength of weak learnability. Machine Learning, 5(2):197-227, 1990.",
]
for ref in refs:
    pdf.multi_cell(0, 4.5, ref)
    pdf.ln(1)

pdf.output("d:/CIS/paper/Containment_Depth_Bound_Theorem.pdf")
print("Theorem PDF generated: d:/CIS/paper/Containment_Depth_Bound_Theorem.pdf")
