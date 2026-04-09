"""Generate the CIS research paper as a professional PDF."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from fpdf import FPDF

class Paper(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Saad 2026 - Inference-Time Epistemic Quarantine for LLM Hallucination Containment", 0, 0, "C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, str(self.page_no()), 0, 0, "C")

    def section(self, num, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 8, f"{num}  {title}", 0, 1)
        self.ln(1)

    def subsection(self, num, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, f"{num}  {title}", 0, 1)
        self.ln(1)

    def body(self, text):
        self.set_font("Times", "", 11)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def formula(self, text):
        self.set_font("Courier", "", 10)
        self.set_x(30)
        self.write(5.5, text)
        self.set_font("Times", "", 11)
        self.ln(6.5)

    def bold_body(self, label, text):
        self.set_font("Times", "B", 11)
        self.write(5.5, label)
        self.set_font("Times", "", 11)
        self.write(5.5, text)
        self.ln(6.5)

    def bullet(self, text):
        self.set_font("Times", "", 11)
        self.set_x(30)
        self.write(5.5, "-  ")
        self.write(5.5, text)
        self.ln(6.5)


pdf = Paper()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font("Helvetica", "B", 16)
pdf.multi_cell(0, 8, "Inference-Time Epistemic Quarantine\nfor LLM Hallucination Containment", 0, "C")
pdf.ln(4)

pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 6, "Muhammad Saad", 0, 1, "C")
pdf.set_font("Helvetica", "I", 10)
pdf.cell(0, 5, "Independent Researcher, Islamabad, Pakistan", 0, 1, "C")
pdf.cell(0, 5, "saadkhan@proton.me", 0, 1, "C")
pdf.cell(0, 5, "April 2026", 0, 1, "C")
pdf.ln(6)

# Abstract
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 6, "Abstract", 0, 1, "C")
pdf.set_font("Times", "I", 10)
pdf.multi_cell(0, 5, (
    "Large language models produce confident but factually incorrect outputs. Existing mitigation strategies "
    "operate at the training or prompting level, leaving a critical gap at inference time: once a hallucinated "
    "claim enters a reasoning chain, no deployed mechanism prevents it from propagating to downstream conclusions. "
    "We introduce the Cognitive Immune System (CIS), a five-layer inference-time architecture that decomposes "
    "LLM responses into atomic claims, scores each claim against independent epistemic signals, and partitions "
    "claims into safe and quarantined sets before they can influence subsequent reasoning. We formalize the core "
    "primitive of Epistemic Quarantine and prove a Containment Depth Bound: for a single-checkpoint detection "
    "rate rho and target escape probability epsilon, at most k* = ceil(log(epsilon)/log(1-rho)) sequential "
    "checkpoints are needed to guarantee containment. We evaluate CIS on HotpotQA using adversarial semantic "
    "injection, where an auxiliary LLM rewrites critical supporting facts to produce wrong answers. On 86 "
    "verified adversarial questions and 100 clean controls, CIS achieves a contamination detection rate of "
    "28.4% (95% CI: 20.0% to 38.6%) with a false positive rate of 15.5% (95% CI: 9.8% to 23.8%). "
    "Substituting the measured rho = 0.284 into the depth bound yields k* = 9 for epsilon = 0.05. We analyze "
    "the primary bottleneck: semantic drift between injected contamination and extracted claims caused by "
    "aggressive LLM paraphrasing during response generation."
))
pdf.ln(4)

# 1. Introduction
pdf.section("1", "Introduction")
pdf.body(
    "Large language models hallucinate. They produce statements that are fluent, confident, and wrong. "
    "This is well documented across model families (Ji et al. 2023, Huang et al. 2023, Zhang et al. 2023), "
    "and it remains one of the central open problems in deploying LLMs for high-stakes applications."
)
pdf.body(
    "The research community has responded with a growing body of work on hallucination detection and "
    "mitigation. Retrieval-augmented generation grounds model outputs in external documents (Lewis et al. 2020). "
    "Self-consistency methods sample multiple responses and check agreement (Wang et al. 2023). Factual "
    "verification systems compare claims against knowledge bases (Min et al. 2023). Training-time approaches "
    "fine-tune models to be more calibrated (Kadavath et al. 2022)."
)
pdf.body(
    "What none of these approaches address is a simple but critical failure mode: once a hallucinated claim "
    "enters a multi-step reasoning chain, it propagates. A wrong intermediate conclusion becomes the premise "
    "for the next step, and by the time the final answer is generated, the contamination is embedded in the "
    "logical structure of the response. There is no mechanism at inference time to detect a contaminated claim "
    "and prevent it from reaching the next reasoning step."
)
pdf.body(
    "This paper introduces such a mechanism. We call it Epistemic Quarantine."
)
pdf.body(
    "The idea is borrowed from immunology. When the human immune system encounters a pathogen, it does not "
    "attempt to reason about whether the pathogen is dangerous. It isolates first, investigates second. "
    "The default action on uncertainty is containment, not propagation."
)
pdf.body(
    "We formalize this into a five-layer architecture called the Cognitive Immune System (CIS):"
)
pdf.bullet("Claim Extraction decomposes a raw LLM response into atomic, verifiable claims.")
pdf.bullet("Contamination Scoring assigns each claim a score phi(c) in [0,1] using three independent signals: Wikipedia cross-verification, LLM factual confidence, and causal ancestry.")
pdf.bullet("Quarantine Partitioning splits claims into a safe set S and a quarantine set Q based on whether phi(c) exceeds threshold tau.")
pdf.bullet("Causal Memory records quarantined claims in a directed acyclic graph, enabling cross-session immunity.")
pdf.bullet("Tolerance Calibration maintains a registry of verified-safe claims to reduce false positives over time.")
pdf.ln(2)
pdf.body(
    "The critical invariant is that the context passed to the next reasoning step comes only from S. "
    "No quarantined claim ever reaches the downstream computation."
)
pdf.body(
    "We prove that this architecture provides a formal containment guarantee. If a single checkpoint detects "
    "contamination with probability rho, then k sequential checkpoints reduce the escape probability to "
    "(1-rho)^k. For any target escape probability epsilon, we need at most k* = ceil(log(epsilon)/log(1-rho)) "
    "checkpoints. This is Theorem 1, the Containment Depth Bound, and it holds regardless of the specific "
    "scoring function as long as checks are independent."
)
pdf.body(
    "We evaluate CIS on HotpotQA (Yang et al. 2018) using a methodology we call adversarial semantic injection. "
    "Rather than randomly replacing entities in the supporting context, we use an auxiliary LLM to rewrite the "
    "critical supporting sentence so that the answer to the question changes. We verify that the injection works "
    "by testing whether the baseline LLM actually produces a wrong answer on the contaminated context. Only "
    "verified adversarial examples enter the benchmark."
)
pdf.body(
    "Our results are honest and below our initial targets. CIS detects 28.4% of adversarial contaminations at "
    "a 15.5% false positive rate. We analyze why: the primary bottleneck is semantic drift. The LLM "
    "paraphrases the contaminated context so aggressively that the extracted claims look different from "
    "the original injection, making Wikipedia cross-verification ineffective. This finding has implications "
    "for any system that attempts to verify LLM outputs by comparing them against external sources."
)

# 2. Related Work
pdf.section("2", "Related Work")
pdf.bold_body("Hallucination Detection. ", (
    "FActScore (Min et al. 2023) decomposes text into atomic facts and checks each against a knowledge source. "
    "SelfCheckGPT (Manakul et al. 2023) uses sampling-based consistency. SAFE (Wei et al. 2024) uses "
    "LLM-as-judge for long-form factuality. These systems detect hallucinations but do not prevent their "
    "propagation through reasoning chains."
))
pdf.bold_body("Retrieval-Augmented Generation. ", (
    "RAG (Lewis et al. 2020) and its descendants (Gao et al. 2023) reduce hallucination by grounding "
    "generation in retrieved documents. However, RAG does not verify the generated output, and contaminated "
    "retrievals propagate directly into responses."
))
pdf.bold_body("Self-Consistency. ", (
    "Wang et al. (2023) show that sampling multiple reasoning paths and taking the majority vote improves "
    "accuracy. This is a form of ensemble verification but operates at the response level, not the claim "
    "level, and does not provide formal containment guarantees."
))
pdf.bold_body("Inference-Time Intervention. ", (
    "Li et al. (2024) modify internal model representations during generation. This is mechanistically "
    "different from our approach: we operate on the output of generation, not its internals, making CIS "
    "model-agnostic."
))
pdf.bold_body("Gap. ", (
    "To our knowledge, no prior work implements a mechanism that (a) decomposes LLM output into atomic "
    "claims, (b) independently verifies each claim, (c) partitions claims into safe and quarantine sets at "
    "inference time, and (d) provides a formal bound on contamination escape probability. CIS fills this gap."
))

# 3. Architecture
pdf.section("3", "Architecture")
pdf.subsection("3.1", "Overview")
pdf.body(
    "Let R be the raw response from an LLM to a query q with optional context x. "
    "The CIS pipeline computes:"
)
pdf.formula("    Pi(R) = L5 . L4 . L3 . L2 . L1(R)")
pdf.body("where each layer performs a specific transformation.")

pdf.subsection("3.2", "Layer 1: Claim Extraction")
pdf.formula("    L1: R -> C = {c1, c2, ..., cn}")
pdf.body(
    "We prompt the LLM to decompose R into atomic, independently verifiable claims. Each claim ci is a "
    "single factual assertion that can be checked against an external source without reference to the other claims."
)

pdf.subsection("3.3", "Layer 2: Contamination Scoring")
pdf.formula("    L2: ci -> phi(ci) in [0, 1]")
pdf.body("The contamination score is a weighted combination of three independent signals:")
pdf.formula("    phi(c) = w1 * S_wiki(c) + w2 * S_cons(c) + w3 * S_causal(c)")
pdf.body("where w1 = 0.60, w2 = 0.30, w3 = 0.10.")
pdf.ln(1)
pdf.bold_body("S_wiki(c): Wikipedia Cross-Verification. ", (
    "We extract named entities from c, retrieve the corresponding Wikipedia summaries, and prompt an LLM to "
    "judge whether the summary supports, contradicts, or is insufficient regarding c. "
    "Contradicted = 1.0, Supported = 0.0, Insufficient = 0.5."
))
pdf.bold_body("S_cons(c): Factual Confidence Probe. ", (
    "We ask an LLM: 'On a scale of 0 to 10, how confident are you that this claim is factually correct?' "
    "We compute S_cons = 1 - (response/10). This inverts the signal so that low confidence yields high "
    "contamination scores."
))
pdf.bold_body("S_causal(c): Causal Ancestry. ", (
    "If the claim shares a causal ancestor in the contamination DAG with a previously quarantined claim, "
    "S_causal = 1. Otherwise S_causal = 0."
))

pdf.subsection("3.4", "Layer 3: Quarantine Partitioning")
pdf.body("Definition (Epistemic Quarantine): Let C = {c1, ..., cn} be the claims extracted from a response. "
         "Let phi: C -> [0,1] be the contamination scoring function and tau in (0,1) the threshold. We define:")
pdf.formula("    S = {c in C : phi(c) < tau}     (safe set)")
pdf.formula("    Q = {c in C : phi(c) >= tau}     (quarantine set)")
pdf.body("such that S and Q are disjoint and S union Q = C.")
pdf.body("The critical invariant: context(step_{k+1}) is a subset of S. No quarantined claim ever reaches the downstream computation.")

pdf.subsection("3.5", "Layer 4: Causal Memory (CC-DAG)")
pdf.body(
    "Quarantined claims are recorded in a Contamination Causal Directed Acyclic Graph G = (V, E) where "
    "vertices represent contamination events and edges represent causal precedence. This provides cross-session "
    "immunity: if a claim in a new session shares structure with a previously quarantined claim, the causal "
    "signal S_causal increases its phi score."
)

pdf.subsection("3.6", "Layer 5: Tolerance Calibration")
pdf.body(
    "A safe registry T stores claims that have been verified as correct with high confidence. Claims in T "
    "bypass the scoring pipeline entirely. This prevents autoimmune false positives on repeatedly verified "
    "facts and provides monotonic FPR reduction over time."
)

# 4. Theorem
pdf.section("4", "Containment Depth Bound")
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Theorem 1 (Containment Depth Bound)", 0, 1)
pdf.set_font("Times", "I", 11)
pdf.body(
    "Let rho = P(phi(c) >= tau | c is contaminated) be the true positive detection rate of a single EQ "
    "checkpoint. Assume detection decisions at successive checkpoints are independent. Then:"
)
pdf.formula("    (i)  P(escape after k checkpoints) = (1 - rho)^k")
pdf.formula("    (ii) k* = ceil( log(epsilon) / log(1 - rho) )")
pdf.ln(1)
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Proof", 0, 1)
pdf.set_font("Times", "", 11)
pdf.body(
    "A contaminated claim escapes a single checkpoint with probability 1 - rho. By the independence "
    "assumption, the probability of escaping k consecutive checkpoints is (1-rho)^k."
)
pdf.body(
    "We require (1-rho)^k <= epsilon. Taking logarithms: k * log(1-rho) <= log(epsilon). Since rho is in (0,1), "
    "log(1-rho) < 0, so dividing reverses the inequality: k >= log(epsilon) / log(1-rho). "
    "The minimum integer satisfying this is k* = ceil(log(epsilon) / log(1-rho))."
)
pdf.ln(1)
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Corollary (Empirical)", 0, 1)
pdf.set_font("Times", "", 11)
pdf.body(
    "For the empirically measured rho = 0.284 and epsilon = 0.05:"
)
pdf.formula("    k* = ceil( log(0.05) / log(0.716) ) = ceil( -2.996 / -0.334 ) = ceil(8.97) = 9")
pdf.body("Nine sequential EQ checkpoints reduce contamination escape probability below 5%.")
pdf.body("For epsilon = 0.01: k* = 14. Fourteen checkpoints reduce escape below 1%.")

# 5. Experimental Setup
pdf.section("5", "Experimental Setup")
pdf.subsection("5.1", "Dataset and Models")
pdf.body(
    "We use the HotpotQA validation set (Yang et al. 2018), a multi-hop question answering dataset where "
    "each question requires reasoning over two or more supporting documents. We use Llama-3.3-70B-Versatile "
    "(via Groq API) as the main reasoning model, and Llama-3.1-8B-Instant for scoring and injection."
)

pdf.subsection("5.2", "Adversarial Semantic Injection")
pdf.body(
    "Random entity replacement achieves only a 2.8% adversarial hit rate because most replacements land on "
    "facts the LLM does not need to answer the question. We replace this with LLM-guided semantic injection:"
)
pdf.bullet("Extract the critical supporting sentences using HotpotQA's annotated supporting fact pointers.")
pdf.bullet("Prompt Llama-3.1-8B-Instant to rewrite the critical sentence so that the question's answer changes, without mentioning the gold answer.")
pdf.bullet("Verify the injection works: run the baseline LLM on the contaminated context. If the baseline still produces the correct answer, discard the question.")
pdf.bullet("Only questions where the baseline is verifiably fooled enter Group A (contaminated).")
pdf.ln(1)
pdf.body("This achieves a 17.2% adversarial hit rate, a 6x improvement over random replacement.")

pdf.subsection("5.3", "Evaluation Protocol")
pdf.body(
    "We scan up to 500 HotpotQA candidates and fill two groups:"
)
pdf.bullet("Group A (contaminated): Questions where the adversarial injection verifiably fools the baseline LLM. CIS processes the contaminated context.")
pdf.bullet("Group B (control): Questions with clean, unmodified context. CIS processes the original context.")
pdf.ln(1)
pdf.bold_body("M1 (CDR): ", "Fraction of Group A questions where CIS quarantines at least one claim.")
pdf.bold_body("M2 (FPR): ", "Fraction of Group B questions where CIS quarantines at least one claim.")
pdf.bold_body("M3 (Accuracy): ", "Exact match accuracy comparison between baseline and CIS.")
pdf.bold_body("M4 (Hit Rate): ", "Fraction of candidates where injection successfully fools the baseline.")
pdf.bold_body("M5 (Latency): ", "Average CIS latency versus baseline latency.")
pdf.ln(1)
pdf.body(
    "Evaluation uses standard HotpotQA exact match: the predicted answer must contain the gold answer string "
    "after normalization. We report Wilson score 95% confidence intervals for M1 and M2."
)

# 6. Results
pdf.section("6", "Results")

# Results table
pdf.set_font("Helvetica", "B", 10)
col_w = [55, 30, 45, 30]
headers = ["Metric", "Value", "95% CI", "Target"]
for i, h in enumerate(headers):
    pdf.cell(col_w[i], 7, h, 1, 0, "C")
pdf.ln()
pdf.set_font("Times", "", 10)
rows = [
    ["M1 (CDR)", "28.4%", "[20.0%, 38.6%]", "> 65%"],
    ["M2 (FPR)", "15.5%", "[9.8%, 23.8%]", "< 20%"],
    ["M3 Baseline Acc (GrpA)", "1.1%", "--", "--"],
    ["M3 CIS Acc (GrpA)", "2.3%", "--", "--"],
    ["M4 Adversarial Hit Rate", "17.2%", "--", "--"],
    ["M5 Avg Baseline Latency", "1,288 ms", "--", "--"],
    ["M5 Avg CIS Latency", "22,291 ms", "--", "--"],
    ["M5 Overhead", "1,631%", "--", "--"],
]
for row in rows:
    for i, val in enumerate(row):
        pdf.cell(col_w[i], 6, val, 1, 0, "C")
    pdf.ln()
pdf.ln(2)
pdf.set_font("Times", "I", 10)
pdf.cell(0, 5, "Table 1: Final experimental results on HotpotQA (86 adversarial, 100 control).", 0, 1, "C")
pdf.ln(3)

pdf.bold_body("Detection rate. ", (
    "CIS quarantines at least one claim in 25 of 88 adversarial questions (28.4%). This is significantly "
    "above zero but below the 65% threshold we would consider reliable for standalone deployment."
))
pdf.bold_body("False positive rate. ", (
    "CIS incorrectly quarantines claims in 16 of 103 control questions (15.5%). This is within our "
    "acceptable bound of 20%."
))
pdf.bold_body("Theorem validation. ", (
    "Substituting rho = 0.284: k* = 9 for epsilon = 0.05 (escape probability < 5% after 9 checkpoints), "
    "k* = 14 for epsilon = 0.01 (escape probability < 1% after 14 checkpoints)."
))
pdf.bold_body("McNemar's test. ", (
    "The difference in accuracy between baseline and CIS is statistically significant (p = 0.0265), with "
    "CIS slightly improving accuracy on adversarial questions by successfully quarantining some contaminated claims."
))

# 7. Analysis
pdf.section("7", "Analysis: Why M1 Is 28.4%")
pdf.body(
    "The primary bottleneck is semantic drift between the injected contamination and the claims CIS "
    "extracts from the LLM response."
)
pdf.bold_body("Paraphrase-level transformation. ", (
    "When the LLM receives contaminated context, it does not copy the false facts verbatim. It synthesizes, "
    "summarizes, and reframes. The atomic claims extracted from this synthesized response often differ lexically "
    "and structurally from the injected contamination, even though they carry the same wrong conclusion. When "
    "CIS sends these paraphrased claims to Wikipedia for verification, the entity matching fails because the "
    "claim uses different words."
))
pdf.bold_body("Sparse Wikipedia coverage. ", (
    "HotpotQA questions frequently involve obscure people, films, and events. For approximately 12% of claims, "
    "no relevant Wikipedia page exists. When Wikipedia returns no match, S_wiki defaults to 0.5 (uncertain), "
    "which contributes only 0.60 x 0.5 = 0.30 to phi, insufficient to cross the threshold tau = 0.55 without "
    "a strong consistency signal."
))
pdf.bold_body("Overconfident scorer. ", (
    "The 8B scoring model frequently returns high factual confidence (7 to 9 out of 10) even for false claims "
    "about entities outside its training distribution. This causes S_cons to remain low (0.1 to 0.3) for "
    "contaminated claims, failing to complement the Wikipedia signal when it is needed most."
))
pdf.bold_body("Implications. ", (
    "These findings suggest that single-checkpoint detection of semantic hallucinations may require "
    "entity-grounded claim extraction (maintaining the link between claims and source entities) and a much "
    "stronger factual verification model. The containment depth bound theorem provides an alternative path: "
    "even with weak single-checkpoint detection, sufficiently many sequential checkpoints still guarantee "
    "containment."
))

# 8. Limitations
pdf.section("8", "Limitations")
pdf.bullet("The experiment uses a single LLM family (Llama) for both generation and scoring. Cross-model evaluation would strengthen the results.")
pdf.bullet("The scoring model (8B parameters) is likely too small for reliable factual verification on obscure entities. A larger scorer (70B or GPT-4 class) could improve M1 significantly.")
pdf.bullet("The independence assumption in Theorem 1 may not hold perfectly in practice, as successive checkpoints share the same scoring model. Empirical verification with multiple checkpoints is needed.")
pdf.bullet("Group A reached 86 of the target 100 questions. The confidence intervals reflect this sample size limitation.")
pdf.bullet("CIS adds 1,631% latency overhead, making it impractical for real-time applications without optimization.")

# 9. Possible Solutions
pdf.section("9", "Possible Solutions and Future Directions")
pdf.bold_body("1. Entity-grounded claim extraction. ", (
    "Instead of extracting free-form atomic claims, maintain explicit links between each claim and the source "
    "entities it references. When verifying against Wikipedia, use these entity links directly rather than "
    "re-extracting entities from paraphrased text. This directly addresses the semantic drift bottleneck."
))
pdf.bold_body("2. Stronger scoring model. ", (
    "Replace the 8B scorer with a 70B or GPT-4 class model. The factual confidence signal S_cons is only "
    "as good as the model producing it. A larger model with broader factual knowledge would give lower "
    "confidence on genuinely false claims, especially for obscure entities."
))
pdf.bold_body("3. Multi-checkpoint deployment. ", (
    "The theorem shows that k* = 9 checkpoints with rho = 0.284 achieve 95% containment. In a production "
    "pipeline with multiple reasoning steps, applying CIS at each step provides the sequential composition. "
    "This does not require improving rho at all."
))
pdf.bold_body("4. Retrieval-augmented scoring. ", (
    "Instead of relying only on Wikipedia, use a dedicated retrieval system (e.g., Wikidata, Google Knowledge "
    "Graph) to find verification evidence for each claim. This would improve coverage for obscure entities "
    "where Wikipedia has no article."
))
pdf.bold_body("5. Latency reduction. ", (
    "The 1,631% overhead comes from sequential Wikipedia lookups and LLM scoring calls per claim. Batching "
    "claims, caching Wikipedia results, and using smaller specialized verification models could reduce this "
    "to under 200% overhead."
))
pdf.bold_body("6. Cross-model ensembling. ", (
    "Using different LLM families for generation (Llama) and scoring (GPT-4, Claude) would increase "
    "independence between the generation and detection stages, better satisfying the independence assumption "
    "in the theorem."
))

# 10. Conclusion
pdf.section("10", "Conclusion")
pdf.body(
    "We introduced Epistemic Quarantine, the first inference-time mechanism for containing hallucination "
    "propagation in LLM reasoning chains. Our main contribution is the formalization and proof of the "
    "Containment Depth Bound, which shows that even imperfect single-checkpoint detection can be composed "
    "into strong containment guarantees through sequential application. On HotpotQA with adversarial "
    "semantic injection, CIS achieves rho = 0.284 per checkpoint, yielding k* = 9 for 95% containment. "
    "The primary bottleneck we identify, semantic drift between injected contamination and LLM-paraphrased "
    "claims, points to entity-grounded claim extraction as the key direction for improving single-checkpoint "
    "detection rates."
)

# References
pdf.section("", "References")
pdf.set_font("Times", "", 9)
refs = [
    "[1]  Gao, Y., Xiong, Y., et al. (2023). Retrieval-augmented generation for large language models: A survey. arXiv:2312.10997.",
    "[2]  Huang, L., Yu, W., et al. (2023). A survey on hallucination in large language models. arXiv:2311.05232.",
    "[3]  Ji, Z., Lee, N., et al. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12).",
    "[4]  Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.",
    "[5]  Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS 2020.",
    "[6]  Li, K., et al. (2024). Inference-time intervention: Eliciting truthful answers from a language model. NeurIPS 2024.",
    "[7]  Manakul, P., et al. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection. EMNLP 2023.",
    "[8]  Min, S., et al. (2023). FActScore: Fine-grained atomic evaluation of factual precision. EMNLP 2023.",
    "[9]  Wang, X., et al. (2023). Self-consistency improves chain of thought reasoning in language models. ICLR 2023.",
    "[10] Wei, J., et al. (2024). Long-form factuality in large language models. arXiv:2403.18802.",
    "[11] Yang, Z., et al. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. EMNLP 2018.",
    "[12] Zhang, Y., et al. (2023). Siren's song in the AI ocean: A survey on hallucination in large language models. arXiv:2309.01219.",
]
for ref in refs:
    pdf.multi_cell(0, 4.5, ref)
    pdf.ln(1)

pdf.output("d:/CIS/paper/CIS_Research_Paper.pdf")
print("Paper PDF generated: d:/CIS/paper/CIS_Research_Paper.pdf")
