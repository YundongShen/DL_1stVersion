# The Geometry of Edit Necessity: Learning What a Requirement Change Entails

---

## 1. Introduction

Pre-trained code language models learn strong semantic representations and achieve strong performance on retrieval and repository-level code generation tasks\cite{arambula2025slice}.   However, semantic similarity alone is insufficient for software evolution\cite{maveli2025can}. Given a requirement change, multiple edits may appear equally relevant to the requirement text while differing in whether maintainers would consider them necessary. Repository-level coding agents frequently generate such semantically plausible but unnecessary edits, often described as scope creep\cite{jimenez2024swe}.   In practice, the boundary is not strictly defined: developers do not optimize for minimal patches, and maintainers may disagree on whether defensive checks, auxiliary refactors, or dependency-related changes should be included\cite{wen2025repository}. Nevertheless, within a shared repository, developers tend to follow stable modification patterns. For similar requirement changes, some edits are routinely included while others are routinely excluded. These patterns are implicitly encoded in repository evolution histories.

We study whether these patterns can be learned as structure in representation space. Specifically, we ask whether a model can distinguish edits with similar semantic affinity to a requirement by learning which edits are entailed by the requirement in the context of the repository. We hypothesize that pre-trained models already capture a substantial portion of semantic necessity\cite{beger2025coconut}, but fail on structural necessity: edits required because of repository dependencies or behavioral constraints rather than textual similarity alone.

To study this problem, we introduce Edit Entailment Learning (EEL), a self-supervised contrastive framework trained on SWE-bench.   EEL jointly embeds four entities: requirements ([REQ]), diff hunks ([HUNK]), source code units being modified ([ORIG]), and test functions ([TEST]). Requirement embeddings capture change intent, [ORIG] encodes repository context that may not appear in the requirement text, and [TEST] approximates behavioral constraints expected by maintainers. We derive three edit tiers directly from SWE-bench structure: test-covered maintainer edits (T1), maintainer-endorsed but test-uncovered edits (T2), and LLM-generated edits outside the maintainer patch (T3). Training uses a MultiPair InfoNCE objective over these relations.

Our experiments show that semantic similarity alone has a clear ceiling. An untrained UniXCoder encoder already recovers 80.6\% of T2 edits, indicating that semantic necessity substantially emerges during pre-training.   Standard contrastive learning with random negatives provides little improvement. In contrast, EEL improves T2 recall to 92.4\% and produces clearly separated regions in embedding space between maintainer-endorsed and non-entailed edits. These results suggest that edit necessity has learnable geometric structure beyond semantic similarity alone. This paper makes the following contributions:
\begin{itemize}
\item We study edit necessity as a representation learning problem and show that semantically related edits and maintainer-endorsed edits form distinguishable regions in embedding space, revealing a gap between semantic similarity and requirement entailment.
\item We introduce Edit Entailment Learning (EEL), a self-supervised framework that learns repository-conditioned edit representations by jointly modeling requirements, edits, repository context, and behavioral signals.
\item We construct a three-tier benchmark over SWE-bench containing maintainer and LLM-generated edits with automatically derived necessity labels.  
\end{itemize}

---

## 2. Related Work

\subsection{Repository-level code generation and patch evaluation}

Repository-level code generation is commonly evaluated through execution-based correctness. SWE-bench defines a patch as correct if it resolves the associated fail-to-pass tests\cite{jimenez2024swe}, and subsequent coding-agent systems largely optimize for this objective \cite{yang2026swe}. However, passing tests does not determine whether all generated edits are necessary. Recent work has begun to examine this gap. Toward Understanding Scope Creep in LLM-Generated Patches shows that LLM-generated fixes frequently modify code outside maintainer-endorsed scope \cite{sajadi2025ai}, while Can Semantic Similarity Evaluate Patch Quality? finds that semantic similarity alone is insufficient for assessing repository-level patches \cite{sajadi2025ai}. hese works identify limitations of current evaluation signals, but they do not model how a candidate edit relates to the modified source code and the tests that define expected behavior.

≈

Pre-trained code models such as CodeBERT \cite{feng2020codebert}, GraphCodeBERT \cite{guo2020graphcodebert}, and UniXCoder \cite{guo2022unixcoder} learn aligned representations between natural language and source code and form the basis of modern code retrieval systems. These models capture semantic correspondence effectively, which explains their strong zero-shot performance in our setting. Prior work on code change representations, such as CC2Vec \cite{hoang2020cc2vec}, models what a change does rather than whether it is necessary given a requirement and repository context. More recent repository-level retrieval approaches also focus primarily on semantic alignment between requirements and code \cite{jiang2025aligncoder}. In contrast, we study the distinction between semantically related edits and maintainer-endorsed edits, and explicitly model the modified source unit as part of the representation space.

\subsection{Contrastive representation learning}

Contrastive learning has become a standard approach for improving embedding geometry in language and code representations. SimCSE shows that contrastive objectives substantially improve sentence-level embedding structure \cite{gao2021simcse}, while subsequent work extends similar objectives to code retrieval and repository-level representations. Hard negative selection is also known to strongly affect learned geometry \cite{robinson2020contrastive}. Our work suggests that, in repository-level software evolution, negatives must be hard along the axis of edit necessity itself. LLM-generated edits outside maintainer patches are useful not because they are random alternatives, but because they are semantically plausible while differing in whether maintainers considered them necessary. This provides the contrastive signal required to separate entailed and non-entailed edits in embedding space.

---

## 3. Method

3 Method

Given a requirement R, a set of LLM-generated candidate hunks \mathcal{H}=\{h_1,\dots,h_n\}, and repository context \mathcal{C} (source files and fail-to-pass tests), the task is to rank candidate edits according to how closely they align with the maintainer-approved modification scope of R. The goal is not simply to retrieve semantically related edits, but to separate edits typically retained in the final maintainer patch from edits that are semantically plausible yet usually excluded.

We organize the method around two questions. RQ1: Can the distinction between maintainer-retained and non-retained edits be learned directly from repository evolution history, without manually labeling generated edits? RQ2: Does the learned embedding space reflect different reasons why an edit is retained, such as semantic relevance to the requirement, consistency with expected behavior, or structural dependence on existing repository code?

3.1 Learning Edit Necessity (RQ1)

We derive supervision directly from SWE-bench and LLM-generated patches. For each SWE-bench requirement, we generate candidate patches using the same requirement prompt, then compare generated hunks against the maintainer patch and associated fail-to-pass tests. This produces three edit tiers: generated hunks that overlap with the maintainer patch and whose modified files are covered by at least one fail-to-pass test (T1), generated hunks that overlap with the maintainer patch but are not covered by fail-to-pass tests (T2), and generated hunks that do not overlap with the maintainer patch (T3). T1 and T2 both correspond to edits retained by maintainers, while T3 corresponds to edits generated by the LLM but excluded from the final patch. The main supervision signal is therefore the separation between T1/T2 and T3. The distinction between T1 and T2 is used mainly to analyze different forms of retained edits, especially the difference between behaviorally visible and structurally motivated changes.

Recovering this boundary requires more than requirement-edit similarity alone. Some retained edits are directly implied by the requirement text, while others arise because the modified code interacts with repository structure or expected test behavior. We therefore jointly model four entities within a shared embedding space: the requirement (\texttt{[REQ]}), fail-to-pass tests (\texttt{[TEST]}), the original source unit being modified (\texttt{[ORIG]}), and the generated hunk itself (\texttt{[HUNK]}). All entities are encoded with a shared UniXCoder backbone and entity-specific type tokens. \texttt{[REQ]} captures requirement intent, \texttt{[TEST]} captures expected behavior, \texttt{[ORIG]} captures the structural context of the modified code, and \texttt{[HUNK]} represents the generated edit itself.

We optimize four relationships simultaneously under a shared MultiPair InfoNCE objective: (\texttt{REQ}, \texttt{HUNK}), (\texttt{REQ}, \texttt{TEST}), (\texttt{ORIG}, \texttt{HUNK}), and (\texttt{REQ}, \texttt{ORIG}). Training on only one relationship would bias the space toward a single aspect of edit acceptance, while jointly optimizing all four constrains the space using requirement semantics, expected behavior, repository structure, and generated edits together. Random in-batch samples are used as standard negatives. T3 hunks from the same requirement instance are additionally used as hard negatives. Unlike random negatives, these edits remain semantically related to the requirement while differing in whether they are retained by maintainers. This forces the model to learn the boundary between retained and scope-exceeding edits rather than semantic similarity alone.

We study progressively richer supervision signals through four training variants. M1 uses only positive pairs with random negatives. M2 adds T3 hard negatives from the same requirement instance. M3 further adds negatives from other issues within the same repository. M4 applies tier-aware weighting that emphasizes T1 pairs over T2 pairs.

3.2 Geometry and Inference (RQ2)

If the learned space captures these relationships successfully, different categories of edits should occupy different regions of the embedding space. Edits driven primarily by requirement semantics tend to align with \texttt{[REQ]}; edits constrained by expected behavior align with \texttt{[TEST]}; and structurally motivated edits align with \texttt{[ORIG]}. In contrast, T3 edits remain semantically related to the requirement but lack strong behavioral or structural support, causing them to occupy a different region of the space. This means that edit type is reflected not by a single similarity score, but by a geometric pattern across the three contextual anchors. T1, T2, and T3 edits are therefore expected to occupy distinguishable regions rather than forming a single homogeneous cluster.

At inference time, we rank all LLM-generated candidate edits for a requirement by

\text{score}(h)=
\alpha \cdot \mathrm{sim}(h,r)
+\beta \cdot \mathrm{sim}(h,\bar{t})
+\gamma \cdot \mathrm{sim}(h,\bar{o}),

where r is the requirement embedding, \bar{t} is the mean over test embeddings, and \bar{o} is the mean over embeddings of modified source units. The score directly reflects the learned geometry of semantic relevance, behavioral consistency, and structural compatibility within the repository.
## 4. Experiments

### 4.1 Setup

**Dataset.** We use SWE-bench Full \cite{jimenez2024swe}: 2,291 resolved issues from 12 Python repositories, each with a requirement description, maintainer gold patch, and fail-to-pass tests. Instances are split 80/10/10 into train (1,833), validation (229), and test (229); all metrics are on the test split.

**Three-tier construction.** Gold patch hunks are assigned Tier 1 if any fail-to-pass test file imports a module matching the hunk's file path or shares the same package directory; otherwise Tier 2. Tier-3 hunks are obtained by prompting \texttt{claude-haiku-4-5} on each instance and retaining generated hunks that do not overlap with the gold patch, yielding Tier-3 candidates for 743 instances (32.4\%). Of the 229 test instances, 82 contain Tier-3 hunks; all retrieval metrics are computed on these 82.

**Metrics.** We report three metrics on the 82 instances with Tier-3 candidates. \textit{nDCG@$k$} uses $k = |T_1 \cup T_2|$ with relevance weights 3/2/0 for Tier-1/2/3, measuring overall ranking quality. \textit{T2-Recall} measures the fraction of Tier-2 hunks recovered in the top-$k$ positions, computed on the 47 instances that also contain Tier-2 hunks; this is our primary metric, as Tier-2 edits have no test coverage and cannot be identified by semantic or behavioral signals alone. \textit{Perfect-Sep} is the fraction of instances where every Tier-3 hunk ranks below every gold hunk.

**Evaluation setting.** Each test instance is augmented with up to 50 same-repository gold hunks from other test-split instances as Tier-0 distractors, increasing ranking difficulty with contextually plausible but irrelevant candidates.

### 4.2 Effect of Structured Negatives (RQ1)

The central question of RQ1 is whether edit necessity is learnable from repository history. The answer depends critically on what constitutes a hard negative.

\begin{table}[h]
\centering
\begin{tabular}{llccc}
\hline
Model & Training signal & nDCG@$k$ & T2-Recall & Perfect-Sep \\
\hline
M0 & untrained & 0.590 & 0.678 & 0.280 \\
M1 & + random negatives & 0.595 & 0.664 & 0.244 \\
M2 & + T3 hard negatives & \textbf{0.825} & \textbf{0.839} & \textbf{0.902} \\
M3 & + cross-issue negatives & 0.806 & 0.784 & 0.829 \\
M4 & + tier-aware weighting & 0.812 & 0.791 & 0.805 \\
\hline
\end{tabular}
\caption{Performance across training variants on 82 test instances with Tier-3 candidates (T2-Recall on 47 instances with Tier-2 hunks).}
\end{table}

**M0 establishes a semantic baseline.** The pretrained UniXCoder encoder without fine-tuning achieves nDCG of 0.590 and T2-Recall of 0.678, confirming that pre-trained semantic representations already capture a substantial portion of edit necessity. However, Perfect-Sep is only 0.280: in 72\% of instances, at least one Tier-3 hunk ranks above a gold hunk. The pretrained encoder cannot reliably separate scope-exceeding edits from necessary ones when both are topically related to the same requirement.

**Random negatives do not improve over the baseline.** M1 fine-tunes with random in-batch negatives drawn from other instances. These negatives are already distinguishable from any given requirement by semantic similarity alone — the pretrained encoder separates hunks from different requirements without any additional training. Fine-tuning on this signal does not pressure the model to learn the within-instance necessity boundary, and the resulting scores are nearly unchanged (nDCG 0.595, T2-Recall 0.664; 2b$>$3 remains n.s.). Contrastive training without structurally appropriate negatives adds no information beyond what the pretrained representations already encode.

**Instance-level hard negatives create the boundary.** M2 introduces Tier-3 hunks from the same requirement instance as hard negatives. Unlike random negatives, these edits pass the semantic filter — they are topically related to the requirement — yet were not adopted by maintainers. This forces the model to learn that semantic relevance to the requirement is insufficient to predict adoption: there must be an additional feature that separates what maintainers retain from what they discard. As Section 4.3 shows, that feature is structural grounding — whether an edit modifies code that is structurally implicated in the requirement, captured by the ORIG signal. The effect is sharp: nDCG rises from 0.595 to 0.825, T2-Recall from 0.664 to 0.839, and Perfect-Sep from 0.244 to 0.902. The geometric structure underlying these gains is analysed in Section 4.3.

**Additional supervision does not improve the learned geometry.** M3 was designed to sharpen within-repository geometric structure by adding negatives from other issues in the same repository, hypothesising that finer-grained intra-repo separation would improve necessity discrimination. M4 was designed to make the Tier-1/Tier-2 gradient geometrically explicit by upweighting Tier-1 loss, reflecting the stronger necessity signal of test-covered edits. Neither achieves its goal: M3 nDCG drops to 0.806 and M4 to 0.812, both below M2 (0.825). Cross-issue negatives are already separable by requirement semantics and add no pressure along the necessity axis; tier-aware weighting introduces a gradient within gold that does not bear on the primary retained-versus-excluded boundary. The geometry established by M2 through instance-level hard negatives is already sufficient, and further objectives that operate on different axes degrade rather than refine it.

### 4.3 Geometric Structure of the Learned Space (RQ2)

**Pre-existing structure, amplified by training.** Figure~\ref{fig:3d_compare} compares 3D UMAP projections of M0 (untrained UniXCoder) and M2 (after EEL training) across 849 Django instances. A key observation about M0 is that the four entity types are not randomly scattered: requirements and tests already cluster in overlapping semantic zones, source units form a dense code-language manifold, and hunks sit in between — reflecting semantic associations that UniXCoder absorbed during pre-training. This latent geometry is meaningful: it encodes the intuition that edits related to a requirement should resemble it, and edits that touch existing code should resemble that code. What M0 lacks is the ability to make this intuition discriminative. Accepted and unaccepted hunks overlap substantially in the M0 space, because the encoder has never been exposed to the distinction between maintainer-retained and LLM-generated edits for the same requirement. EEL training on instance-level hard negatives provides exactly this signal: it amplifies the attraction between accepted hunks and their contextual anchors (REQ, TEST, ORIG), while simultaneously repelling unaccepted hunks from all three. The effect is a 5.8× improvement in hunk-tier silhouette score (0.024 → 0.139, measured in the original 256-dimensional cosine space): what was implicit in the pre-trained space becomes geometrically explicit and directly usable as a ranking signal.

**The learned space as a visualization of the entailment score.** Figure~\ref{fig:django_scatter} projects all 2,857 HUNK embeddings from the 849 Django instances onto the $\mathrm{sim}(h,\mathrm{REQ})\times\mathrm{sim}(h,\mathrm{ORIG})$ plane, with $\mathrm{sim}(h,\mathrm{TEST})$ encoded as color intensity. These three dimensions are the exact components of the inference-time entailment score $\alpha\cdot\mathrm{sim}(h,r)+\beta\cdot\mathrm{sim}(h,\bar{t})+\gamma\cdot\mathrm{sim}(h,\bar{o})$: the scatter is a direct visualization of what the scoring formula measures for each hunk. Accepted and unaccepted edits do not separate along a single axis — they occupy structurally distinct regions of the joint space, revealing that edit necessity is genuinely multi-dimensional. All Django gold hunks are Tier 2 (Django's top-level \texttt{tests/} directory is not co-located with any source module, so the coverage heuristic that assigns Tier 1 never fires), providing a clean two-class case study with no label ambiguity. Applying fixed thresholds ($\tau_r=0.65$, $\tau_t=0.40$, $\tau_o=0.80$) to each similarity component yields five proximity regions (Table~\ref{tab:regions}).

\begin{table}[h]
\centering
\small
\begin{tabular}{llrrl}
\hline
Region & Anchors & $n$ & Gold\% & Character \\
\hline
R1 & REQ + TEST  & 1{,}012 & 100\% & Direct fix, behaviorally covered \\
R2 & REQ + ORIG  & 1{,}223 &  99\% & Semantically aligned and structurally grounded \\
R3 & REQ only    &   157   &  99\% & New code path, requirement-aligned \\
R4 & ORIG only   &    84   &  98\% & Structurally induced; requirement-opaque \\
R5 & none        &   381   &  15\% & No anchor alignment; predominantly unaccepted \\
\hline
\end{tabular}
\caption{Proximity regions in the Django embedding space (M2, 2,857 hunks). Gold\% = fraction of accepted (Tier-2) hunks. 99\% of all Tier-3 hunks (324/327) fall in R5.}
\label{tab:regions}
\end{table}

**A gradient of necessity, not a binary boundary.** The region structure reveals that the model has learned to decompose edit necessity along multiple dimensions rather than drawing a single accepted/unaccepted boundary. R1–R4 form a gradient from richly grounded to purely structurally induced: R1 hunks are grounded in requirement semantics, behavioral expectations, and structural context simultaneously; R2 hunks are semantically and structurally grounded but test-invisible; R3 hunks match the requirement but add new code without a structural anchor; R4 hunks are grounded only in structure, with no textual signal from the requirement at all. Each region represents a distinct \emph{reason} why an edit is accepted — and the model has learned to recognize all four. The unaccepted region R5 is where the gradient terminates: 85\% of T3 hunks fall here, lacking grounding in any anchor simultaneously, which means the model has learned that scope-creep edits fail all three tests of necessity at once. This is a qualitatively stronger result than simply ranking gold above T3: the model encodes not just \emph{whether} an edit is accepted, but \emph{why}. R4 is particularly important as evidence of what the ORIG entity contributes: 84 accepted hunks with $s_r\leq0.65$ would be systematically missed by any method relying solely on requirement similarity — BM25, standard retrieval, or M1 — yet are recovered at 98\% purity through structural grounding alone. Accepted hunks that modify existing functions have median $\mathrm{sim}(h,\mathrm{ORIG})=0.924$ versus $0.010$ for Tier-3 hunks (Mann–Whitney $p<10^{-100}$). The existence of this gradient also suggests a path for future refinement: the current model treats all four anchors with fixed weights; training variants that specifically strengthen the ORIG signal for structurally induced edits, or that learn anchor weights conditioned on hunk type, could push T2-Recall further by more precisely targeting the R4 region. Concrete hunk examples for each region are provided in Appendix~\ref{app:region_examples}.

### 4.4 Sources of Geometric Separation

To verify the contribution of each contextual entity, we retrain three M2-equivalent models with one entity removed and set the corresponding inference weight to zero.

\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
Model & nDCG@$k$ & T2-Recall \\
\hline
M2 (full) & \textbf{0.825} & \textbf{0.839} \\
M2 w/o ORIG ($\gamma=0$) & 0.668 & 0.721 \\
M2 w/o TEST ($\beta=0$) & 0.806 & 0.830 \\
M2 w/o REQ ($\alpha=0$) & 0.789 & 0.755 \\
\hline
\end{tabular}
\caption{Entity ablation on 82 test instances. Each variant removes one contextual anchor from both training and the inference score.}
\end{table}

The results directly confirm the geometric analysis in Section~\ref{sec:geometry}. ORIG removal causes the largest drop (nDCG $-$0.157, T2-Recall $-$0.118): the R4 hunks — accepted edits with no semantic alignment to the requirement — lose their only active signal and fall below distractors. This is the strongest evidence that structural context captures a dimension of edit necessity that requirement and behavioral signals cannot reach. TEST removal shows a modest drop in nDCG ($-$0.019) but near-negligible T2-Recall degradation ($-$0.009). The nDCG drop better reflects TEST's actual role: its primary contribution is at the upper end of the necessity gradient, separating T1 hunks (R1: REQ+TEST active) from T2 hunks (R2: REQ+ORIG active) and encoding the distinction between behaviorally constrained and structurally grounded edits. T2-Recall is structurally insensitive to this because T2 hunks by definition lack the behavioral coverage that TEST captures. REQ removal produces an intermediate drop in both metrics (nDCG $-$0.036, T2-Recall $-$0.084), confirming that semantic alignment is necessary but not sufficient on its own. Together, the ablations establish a clear hierarchy: ORIG is the most critical signal for recovering necessary edits, REQ provides complementary semantic coverage, and TEST refines the upper-gradient separation between behaviorally visible and structurally induced changes.

## 6. Conclusion

We presented Edit Entailment Learning, a contrastive framework that learns a four-entity geometric space in which a hunk's position relative to REQ, TEST, and ORIG anchors encodes its degree of edit necessity. EEL achieves nDCG@$k$ of 0.825 and T2-Recall of 0.839 on our three-tier benchmark, recovering structurally necessary edits that requirement-similarity and test-coverage baselines systematically miss. The learned space amplifies a weak but pre-existing tier separation in UniXCoder embeddings (silhouette 0.024 → 0.139) and organises hunks into five interpretable proximity regions that form a coherent necessity gradient. Entity ablations confirm a clear signal hierarchy: ORIG removal causes the largest degradation (nDCG $-$0.157, T2-Recall $-$0.118), establishing that structural context is the primary mechanism for recovering R4 edits — helper changes that are requirement-opaque yet causally necessary; REQ removal produces an intermediate drop (nDCG $-$0.036, T2-Recall $-$0.084); and TEST contributes primarily at the upper end of the gradient, separating behaviorally visible T1 edits from structurally grounded T2 edits (nDCG $-$0.019). Cross-repository evaluation confirms that these gains reflect genuine geometric generalisation rather than repository memorisation, with the largest T2-Recall improvements concentrated in small-exposure repositories.

Two categories of limitation affect the current work. First, the three-tier labelling scheme is entirely automatic and carries systematic noise at both ends. Gold patches sourced from SWE-bench are maintainer-accepted but not exhaustive: some structurally induced changes necessary for correctness were never committed, leaving genuine T2 hunks absent from training targets. At the other end, the LLM-generated T3 corpus contains implementations that are architecturally reasonable but simply not the approach maintainers chose; treating all non-overlapping LLM edits as scope creep introduces false negatives into the negative training signal. Human annotation — asking domain experts to judge which LLM-generated hunks represent a defensible implementation of the stated requirement — would produce a cleaner, continuous necessity signal that the current automatic construction cannot provide. Second, the model itself has several architectural constraints. The MultiPairInfoNCE loss draws negatives uniformly from the in-batch pool, providing no hard negatives that specifically challenge the R4 boundary (structurally similar but causally unrelated code). The inference score is a fixed linear combination $\alpha s_r + \beta s_t + \gamma s_o$ with globally shared weights, which cannot adapt to instances where the requirement is terse, tests are sparse, or the codebase is atypically structured. Finally, because T3 hunks are generated for only 32.4\% of instances, the training signal for scope-creep rejection is unevenly distributed across the dataset.

Future work can address these limitations along three converging directions. Higher-quality data is the most direct lever: human-annotated necessity judgements on LLM-generated hunks would enable training with soft, continuous labels rather than the current hard three-tier assignment, directly attacking the false-negative problem in T3 and the incompleteness problem in T2. With a richer label space, finer-grained necessity gradients become trainable targets — moving beyond three discrete tiers toward a spectrum that distinguishes, for example, critical structural dependencies from low-priority refactors triggered by the same requirement. Region-aware curriculum learning (easy: R1 vs.\ R5; hard: R2 vs.\ R4) and anchor-interaction terms in the scoring function (replacing the linear combination with a learned aggregation) are natural architectural companions to such data improvements. Finally, the four-entity geometry is not inherently tied to patch generation: the same REQ–TEST–ORIG–HUNK structure applies to pull-request review (where REQ is a reviewer comment), to change-request triage, and to detecting over-engineering in AI coding agents operating autonomously on large codebases. Extending EEL to multi-language repositories and to these deployment contexts would establish whether the learned geometry captures a universal property of edit necessity or one specific to the Python open-source maintenance setting studied here.

---

## Appendix

### A. Region Example Hunks (Django)
\label{app:region_examples}

We trace one representative hunk per proximity region to its Django source. Similarity scores are computed by M2: $s_r = \mathrm{sim}(h,\mathrm{REQ})$, $s_t = \mathrm{sim}(h,\mathrm{TEST})$, $s_o = \mathrm{sim}(h,\mathrm{ORIG})$.

**R1 — \texttt{django/core/checks/registry.py} (\texttt{django\_\_django-12396},\ $s_r{=}0.82$,\ $s_t{=}0.82$,\ $s_o{=}0.64$).** The requirement reports that the test runner accesses the production database when a test subset uses only the default database, because all database checks fire regardless of which databases are under test. The fix scopes \texttt{check\_database\_backends} to the databases actually being tested:

```diff
-def check_database_backends(*args, **kwargs):
+def check_database_backends(databases=None, **kwargs):
+    if databases is None:
+        return []
     issues = []
-    for conn in connections.all():
+    for alias in databases:
+        conn = connections[alias]
         issues.extend(conn.validation.check(**kwargs))
```

The change is verbally described in the requirement and is directly exercised by \texttt{tests/check\_framework/tests.py::test\_registered\_check\_did\_run}. Both REQ and TEST anchors are active.

**R2 — \texttt{django/core/management/commands/showmigrations.py} (\texttt{django\_\_django-14513},\ $s_r{=}0.78$,\ $s_t{=}0.00$,\ $s_o{=}0.90$).** The requirement identifies a discrepancy between the visual output of \texttt{showmigrations} and the actual recorded state for squashed migrations. The patch modifies the existing \texttt{show\_list()} method to query the migration recorder directly:

```diff
     loader = MigrationLoader(connection, ignore_no_migrations=True)
+    recorder = MigrationRecorder(connection)
+    recorded_migrations = recorder.applied_migrations()
     graph = loader.graph
```

The hunk semantically matches the requirement and modifies an existing method body (ORIG active). No fail-to-pass test imports \texttt{showmigrations} directly (TEST inactive). This is the region of primary interest: structurally grounded, requirement-aligned edits that are invisible to behavioral tests alone.

**R3 — \texttt{django/db/models/expressions.py} (\texttt{django\_\_django-16092},\ $s_r{=}0.77$,\ $s_t{=}0.00$,\ $s_o{=}0.44$).** The requirement adds \texttt{Field.db\_default} for database-level column defaults. Part of the implementation introduces a new expression class with no prior counterpart in the codebase:

```diff
+class DatabaseDefault(Expression):
+    """Placeholder expression for the database default in an insert query."""
+
+    def as_sql(self, compiler, connection):
+        return "DEFAULT", []
```

The hunk is semantically aligned with the new feature (REQ active) but adds entirely new code rather than modifying an existing function (ORIG below threshold). No test directly covers this new class. R3 hunks represent requirement-described additions where structural grounding is absent because there is no existing code to modify.

**R4 — \texttt{django/db/models/query.py} (\texttt{django\_\_django-16072},\ $s_r{=}0.32$,\ $s_t{=}0.00$,\ $s_o{=}0.83$).** The requirement asks that \texttt{update\_or\_create} update only the fields listed in \texttt{defaults}, not the entire model. Implementing this requires pre-computing the set of non-primary-key concrete field names. The patch adds a cached property for this purpose:

```diff
+    @cached_property
+    def _non_pk_concrete_field_names(self):
+        names = []
+        for field in self.concrete_fields:
+            if not field.primary_key:
+                names.append(field.name)
+                if field.name != field.attname:
+                    names.append(field.attname)
+        return frozenset(names)
```

The requirement never mentions \texttt{\_non\_pk\_concrete\_field\_names} or cached properties. The hunk is a structural dependency of the main fix: \texttt{update\_or\_create} must call this property to resolve which fields to pass to \texttt{save()}. Only the ORIG anchor is active. This region — structurally induced, requirement-opaque — is the primary target of the ORIG entity: edits that semantic or behavioral signals alone cannot recover.

**R5 — \texttt{django/db/backends/base/base.py} (\texttt{django\_\_django-15766},\ T3,\ $s_r{=}0.01$,\ $s_t{=}0.00$,\ $s_o{=}{-}0.09$).** The requirement proposes robust \texttt{on\_commit} handlers that continue executing when a previous handler raises. The LLM generated a backward-compatibility approach that infers a \texttt{robust} flag by inspecting tuple length:

```diff
-    sids, func = current_run_on_commit.pop(0)
-    func()
+    hook_data = current_run_on_commit.pop(0)
+    if len(hook_data) == 2:
+        sids, func = hook_data
+    else:
+        sids, func, robust = hook_data
+    if robust:
+        try:
+            func()
+        except Exception:
+            pass
+    else:
+        func()
```

The maintainers implemented robustness differently — adding \texttt{robust} as an explicit keyword argument to \texttt{on\_commit()} itself. The LLM hunk was not retained. After M2 training it is repelled from all anchors: the backward-compatibility tuple-inspection pattern does not match the structural signature of the codebase's \texttt{on\_commit} implementation, and the approach diverges from the requirement's framing of explicit robust invocation. The dominant T3 pattern in Django's R5 cluster is semantically plausible implementations that take a structurally mismatched approach — a pattern the model learns to separate from structurally grounded edits.
