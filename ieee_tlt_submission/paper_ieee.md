# Benchmarking Knowledge Tracing Methods Across Five Educational Datasets: A Comparative Study of Bayesian, Logistic, and Deep Learning Approaches

## Abstract

Knowledge tracing (KT) models a student's evolving knowledge state to predict future performance, supporting adaptive learning, timely intervention, and mastery-based progression. Despite the rapid growth of deep learning approaches to KT, few recent benchmarks have systematically compared classical and neural methods under a single consistent evaluation protocol with properly implemented baselines. This study benchmarks five KT methods spanning three methodological generations: Bayesian Knowledge Tracing (BKT), Performance Factors Analysis (PFA), Deep Knowledge Tracing (DKT), Self-Attentive Knowledge Tracing (SAKT), and a generic causal Transformer baseline (TransformerKT). All models are evaluated on five public datasets using 5-fold student-level cross-validation, five complementary metrics, and a common prediction mask ensuring identical evaluation subsets. DKT achieves the highest macro-averaged AUC-ROC (0.725), outperforming BKT and PFA (both 0.685) by approximately 4 percentage points. TransformerKT (0.715) closely trails DKT at 2.9 times the computational cost. Student-clustered bootstrap confidence intervals confirm that DKT's advantage is statistically meaningful on most datasets. Classical methods remain competitive on datasets with long interaction histories. These findings provide practitioners with empirically grounded guidance for model selection in knowledge tracing applications.

*Index Terms*-- Knowledge tracing, educational data mining, deep learning, benchmark, learning analytics

## I. Introduction

Understanding what students know and predicting what they will learn next is a central challenge in educational technology. Knowledge tracing (KT) refers to the computational task of modeling a student's evolving knowledge state based on their sequence of interactions with learning materials [1]. Accurate KT models have practical implications for adaptive learning systems, as they enable personalized content selection, timely intervention, and mastery-based progression through curricula [2].

The KT field has evolved through several methodological generations, from classical probabilistic models such as BKT [1] and PFA [5], through recurrent neural networks [7], to attention-based architectures [9], [10] and recent large-language-model approaches [11], [12]. Each generation has claimed improvements over its predecessors, but the evaluation conditions across studies vary widely in datasets, metrics, preprocessing, and prediction targets, making direct comparisons unreliable.

Gervet et al. raised this question directly [8], finding that simpler baselines could match deep learning approaches on some datasets when properly tuned. More recent comparative evaluations have continued this line of inquiry [14], [15], but they typically focus on specific model families or use different evaluation protocols across studies, making direct cross-method conclusions difficult. A gap remains for benchmarks that compare classical and deep learning KT methods under one unified protocol with carefully implemented baselines, multiple complementary metrics, and bootstrap-based inferential support.

This study addresses that gap. The contributions are as follows:

1) A systematic benchmark of five KT methods spanning three methodological generations (Bayesian, logistic, and deep learning), evaluated on five publicly available datasets under identical conditions.
2) A common prediction mask methodology that excludes first skill encounters and chunk-boundary positions, ensuring all models are scored on the identical test subset regardless of their internal processing differences.
3) Student-clustered bootstrap confidence intervals that account for within-student correlation, enabling rigorous pairwise model comparisons.
4) An open-source, fully reproducible implementation with complete seed control.

The remainder of this paper is organized as follows. Section II reviews related work on knowledge tracing methods and benchmarking. Section III describes the methodology, including datasets, models, and evaluation protocol. Section IV presents the experimental results. Section V discusses findings and limitations. Section VI concludes the paper.

## II. Related Work

### A. Classical Knowledge Tracing Methods

The earliest and most influential KT framework is Bayesian Knowledge Tracing (BKT), introduced by Corbett and Anderson [1]. BKT models each skill as a two-state Hidden Markov Model with parameters governing the initial probability of mastery, the probability of learning, and the probabilities of guessing and slipping. BKT remained the dominant approach for nearly two decades and has been extended through student-level individualization [3] and integration with item response theory [4].

A parallel line of work explored logistic regression-based methods. Performance Factors Analysis (PFA), proposed by Pavlik et al. [5], models student performance as a function of skill-specific intercepts and cumulative success and failure counts per skill, drawing on Learning Factors Analysis [6]. PFA offers computational efficiency and interpretability but does not model latent knowledge states explicitly.

### B. Deep Learning Approaches

Deep Knowledge Tracing (DKT), introduced by Piech et al. [7], marked a paradigm shift by using a recurrent neural network (LSTM) to process sequences of student interactions. DKT reported substantial improvements over BKT, but has been criticized for non-interpretability and inconsistencies in predicted knowledge states [8].

Subsequent approaches leveraged attention mechanisms from natural language processing. Pandey and Karypis [9] proposed Self-Attentive Knowledge Tracing (SAKT), which replaces the recurrent architecture with a self-attention mechanism. Ghosh et al. [10] extended this with Attentive Knowledge Tracing (AKT), incorporating monotonic attention and Rasch-model-inspired embeddings. More recently, Jung et al. [11] proposed CLST, aligning a generative language model as a knowledge tracer, Neubauer et al. [12] investigated principled transformer architectures, and Badrinath and Pardos [13] explored neural network parameter generation to optimize BKT.

### C. Benchmarking and Comparative Studies

Comparative evaluation is essential for progress in KT but challenging due to differences in datasets, preprocessing, and evaluation targets across studies. Gervet et al. [8] provided an important early comparison showing that properly tuned classical baselines could match DKT on some datasets. Liu et al. [14] released pyKT, a standardized Python library enabling fairer comparisons across deep KT models. Schmucker et al. [15] conducted a comprehensive evaluation of deep KT methods with a focus on reproducibility. However, these works either focus primarily on deep learning models or do not include the full range of classical baselines with proper implementations. The present study complements this prior work by including both classical and deep learning methods under a single protocol with student-clustered bootstrap inference.

## III. Methodology

### A. Datasets

Five publicly available educational datasets were selected for this benchmark, chosen for their widespread use in prior KT research and their diversity in terms of domain, scale, and student population characteristics. TABLE I presents descriptive statistics for each dataset.

**TABLE I**

*DESCRIPTIVE STATISTICS FOR THE FIVE BENCHMARK DATASETS*

| Dataset | Interactions | Students | Items | Skills | Correct rate | Avg. per student |
|---|---|---|---|---|---|---|
| ASSISTments 2009 | 278,336 | 3,114 | 17,708 | 149 | 0.659 | 89.4 |
| ASSISTments 2015 | 656,154 | 14,228 | 100 | 100 | 0.730 | 46.1 |
| ASSISTments 2017 | 934,638 | 1,708 | 3,162 | 411 | 0.374 | 547.2 |
| Statics 2011 | 189,297 | 282 | 1,223 | 98 | 0.765 | 671.3 |
| Algebra 2005 | 606,983 | 567 | 173,113 | 271 | 0.755 | 1,070.5 |

The ASSISTments datasets (2009, 2015, and 2017) originate from the ASSISTments online tutoring platform for middle and high school mathematics [16]. The three versions differ substantially: ASSISTments 2009 has a moderate number of students with a balanced correct rate; ASSISTments 2015 has the most students but the fewest items per skill; and ASSISTments 2017 has the lowest correct rate (0.374), indicating more challenging content. The Statics 2011 dataset was collected from a college-level engineering course on the Open Learning Initiative platform [17], featuring the smallest student population (N = 282) but the longest average interaction sequences (M = 671.3). The Algebra 2005 dataset originates from the KDD Cup 2010 challenge [18] and contains the most items (173,113) and the longest average sequences (M = 1,070.5).

All datasets were obtained in preprocessed form from a publicly available standardized repository, which provides a consistent schema with five columns: user_id, item_id, timestamp, correct, and skill_id. No missing values were present after preprocessing.

### B. Data Preprocessing

Several preprocessing steps were applied to ensure comparability across datasets and models.

*Minimum interaction threshold.* Students with fewer than five interactions were excluded, following the convention of Piech et al. [7].

*ID remapping.* Skill and item identifiers were remapped to contiguous integers starting from zero, as required by embedding-based models.

*Sequence construction.* For deep learning models, student interaction sequences were segmented into non-overlapping windows of at most 200 interactions. Subsequences shorter than three interactions were discarded. The maximum sequence length of 200 follows the setting of Pandey and Karypis [9]. Classical models (BKT, PFA) received full, unchunked interaction histories.

*Train-test splitting.* A 5-fold cross-validation procedure was employed with splits at the student level. All interactions of a given student appear in either the training or test set, preventing data leakage [19].

*Common prediction mask.* To ensure identical evaluation subsets, a common prediction mask was applied at test time. This mask excludes each student's first encounter with each skill and chunk-boundary positions (positions 0, 200, 400, etc. within each student's history) where deep learning models produce no next-step prediction. Excluding these positions from all models ensures that classical and deep learning approaches are scored on an identical evaluation subset.

### C. Models

Five KT models were evaluated, representing three generations of approaches.

*1) Bayesian Knowledge Tracing (BKT):* BKT was implemented following the original formulation [1]. Each skill is modeled as a two-state Hidden Markov Model with four parameters: initial mastery probability p(L_0), learning transition probability p(T), guessing probability p(G), and slipping probability p(S). Parameters were estimated using the forward-backward algorithm [20] with EM, converging within tolerance 1e-4 over a maximum of 50 iterations per skill. Guessing and slipping were constrained to [0.001, 0.40] following standard identifiability constraints [21]. BKT was trained on full, unchunked interaction histories.

*2) Performance Factors Analysis (PFA):* PFA was implemented following the standard formulation [5] as a logistic regression with three feature categories: skill-specific intercepts via one-hot encoding, skill-gated cumulative success counts, and skill-gated cumulative failure counts, yielding 3N_skills features implemented as a sparse matrix. The model was fit using L-BFGS with L2 regularization (C = 1.0, max 5,000 iterations). PFA was trained on full, unchunked interaction histories.

*3) Deep Knowledge Tracing (DKT):* DKT was implemented following Piech et al. [7] using a single-layer LSTM with 100 hidden units. The input is a one-hot encoding of the interaction (skill combined with correctness). The LSTM output passes through dropout (p = 0.2) and a linear projection with sigmoid activation. Training used the Adam optimizer [22] with learning rate 0.001, batch size 64, and gradient clipping at norm 1.0. Epochs were adapted to dataset size: 15 for smaller datasets, 10 for medium, and 8 for the largest.

*4) Self-Attentive Knowledge Tracing (SAKT):* SAKT was implemented following Pandey and Karypis [9] with an embedding dimension of 64, separate interaction and skill embeddings, positional embeddings, single-head causal attention, layer normalization, and a feedforward network with expansion factor 4. Dropout (p = 0.2) was applied after attention and feedforward layers. Training hyperparameters matched DKT.

*5) TransformerKT:* A generic Transformer encoder baseline with 2 layers, 4 attention heads, and embedding dimension 64. Inputs consist of the sum of skill, correctness, and positional embeddings with causal masking. The encoded context at position t is concatenated with the skill embedding at t+1 and passed through a two-layer feedforward network with ReLU activation. This model does not implement the Rasch-model-based embeddings or monotonic attention of the full AKT architecture [10].

### D. Evaluation Protocol

Five complementary metrics were used, all computed on the common prediction mask.

*AUC-ROC* measures the probability that a model ranks a correct response higher than an incorrect one. It is threshold-independent and the primary metric in KT research [23].

*PR-AUC* is particularly informative for datasets with imbalanced class distributions, focusing on the positive class [24].

*Accuracy* is the proportion of correct predictions at threshold 0.5.

*F1 score* is the harmonic mean of precision and recall at threshold 0.5.

*RMSE* measures the average magnitude of prediction errors between predicted probabilities and binary outcomes.

Means and standard deviations are reported across the five folds. Additionally, 95% student-clustered bootstrap confidence intervals (1,000 resamples) are reported for AUC-ROC and PR-AUC. The clustered bootstrap resamples students with replacement rather than individual interactions, correctly accounting for within-student correlation.

### E. Reproducibility and Computational Environment

Full seed control was implemented for NumPy, PyTorch (including CUDA, cuDNN deterministic mode), and DataLoader worker initialization. The complete code is available at [URL removed for blind review]. All experiments were conducted on a consumer-grade macOS machine (3.9 GHz CPU) using PyTorch 2.8.0 without GPU acceleration.

## IV. Experimental Results

### A. Overall Performance Comparison

TABLE II presents the mean AUC-ROC and standard deviation across five folds for all model-dataset combinations. DKT achieved the highest AUC-ROC on all five datasets. Fig. 1 provides a visual comparison with error bars representing standard deviations.

**TABLE II**

*MEAN AUC-ROC (AND STANDARD DEVIATION) BY MODEL AND DATASET*

| Dataset | BKT | PFA | DKT | SAKT | TransformerKT |
|---|---|---|---|---|---|
| ASSISTments 2009 | .706 (.005) | .701 (.005) | **.735 (.006)** | .723 (.006) | .728 (.006) |
| ASSISTments 2015 | .668 (.002) | .678 (.001) | **.717 (.001)** | .702 (.002) | .707 (.002) |
| ASSISTments 2017 | .641 (.003) | .631 (.004) | **.677 (.003)** | .646 (.003) | .659 (.004) |
| Statics 2011 | .656 (.006) | .661 (.007) | **.698 (.011)** | .676 (.009) | .690 (.010) |
| Algebra 2005 | .753 (.007) | .751 (.008) | **.796 (.006)** | .775 (.007) | .793 (.007) |
| Macro avg. | .685 | .685 | **.725** | .704 | .715 |

Several patterns are evident. First, deep learning methods consistently outperform classical methods across all datasets. Second, the performance gap varies by dataset, ranging from 2.9 AUC percentage points on ASSISTments 2009 to 4.9 points on ASSISTments 2015. Third, among deep learning methods, DKT achieves the highest AUC-ROC, with TransformerKT as a close second.

### B. PR-AUC Analysis

Fig. 2 presents the PR-AUC results. PR-AUC is particularly revealing for ASSISTments 2017, where the 37.4% correct rate makes this metric more sensitive to model differences than AUC-ROC. On this dataset, DKT (0.561) outperforms TransformerKT (0.547) by a wider margin than observed in AUC-ROC. On datasets with higher correct rates (Algebra 2005: 75.5%), PR-AUC values are uniformly high (0.90--0.92) and model differences are compressed.

### C. Multi-Metric Comparison

TABLE III presents the complete multi-metric results.

**TABLE III**

*COMPLETE BENCHMARK RESULTS: MEAN VALUES ACROSS 5-FOLD CROSS-VALIDATION*

| Dataset | Model | AUC-ROC | PR-AUC | Accuracy | F1 | RMSE |
|---|---|---|---|---|---|---|
| ASSISTments 2009 | BKT | .706 | .793 | .707 | .799 | .444 |
| | PFA | .701 | .791 | .694 | .795 | .448 |
| | DKT | .735 | .821 | .716 | .801 | .435 |
| | SAKT | .723 | .812 | .707 | .797 | .441 |
| | TransformerKT | .728 | .815 | .712 | .801 | .438 |
| ASSISTments 2015 | BKT | .668 | .826 | .728 | .834 | .433 |
| | PFA | .678 | .838 | .724 | .834 | .432 |
| | DKT | .717 | .863 | .731 | .835 | .423 |
| | SAKT | .702 | .853 | .728 | .834 | .427 |
| | TransformerKT | .707 | .855 | .732 | .836 | .425 |
| ASSISTments 2017 | BKT | .641 | .517 | .659 | .327 | .467 |
| | PFA | .631 | .499 | .650 | .257 | .469 |
| | DKT | .677 | .561 | .673 | .415 | .458 |
| | SAKT | .646 | .525 | .662 | .341 | .466 |
| | TransformerKT | .659 | .547 | .672 | .374 | .462 |
| Statics 2011 | BKT | .656 | .856 | .767 | .865 | .410 |
| | PFA | .661 | .862 | .767 | .867 | .411 |
| | DKT | .698 | .880 | .770 | .868 | .404 |
| | SAKT | .676 | .869 | .767 | .867 | .408 |
| | TransformerKT | .690 | .875 | .769 | .868 | .405 |
| Algebra 2005 | BKT | .753 | .902 | .799 | .879 | .382 |
| | PFA | .751 | .902 | .796 | .879 | .384 |
| | DKT | .796 | .922 | .809 | .885 | .370 |
| | SAKT | .775 | .913 | .802 | .881 | .377 |
| | TransformerKT | .793 | .921 | .810 | .886 | .370 |

While AUC-ROC differences between models are substantial, accuracy and F1 differences are considerably smaller. On Statics 2011, the AUC-ROC gap between BKT (0.656) and DKT (0.698) is 4.2 percentage points, but the accuracy difference is only 0.3 points (0.767 vs. 0.770). This divergence occurs because accuracy is sensitive to the classification threshold and base rate.

PFA achieves an F1 of only 0.257 on ASSISTments 2017, far below all other models. This occurs because PFA's logistic regression predicts most interactions as incorrect on this low-correct-rate dataset, resulting in very low recall for the positive class.

Fig. 3 presents the RMSE comparison, showing that DKT and TransformerKT achieve the lowest RMSE on every dataset.

### D. Computational Efficiency

Fig. 4 and TABLE IV present the training time per fold.

**TABLE IV**

*AVERAGE TRAINING TIME PER FOLD IN SECONDS*

| Model | ASSIST 09 | ASSIST 15 | ASSIST 17 | Statics 11 | Algebra 05 | Mean |
|---|---|---|---|---|---|---|
| BKT | 45.3 | 141.5 | 194.3 | 24.8 | 71.7 | 95.5 |
| PFA | 3.8 | 1.4 | 17.3 | 11.0 | 38.4 | 14.4 |
| DKT | 33.6 | 79.5 | 33.9 | 9.2 | 22.3 | 35.7 |
| SAKT | 26.9 | 64.5 | 19.9 | 7.3 | 15.0 | 26.7 |
| TransformerKT | 98.1 | 253.3 | 77.2 | 28.8 | 58.6 | 103.2 |

PFA is the fastest model (M = 14.4 s), followed by SAKT (26.7 s), DKT (35.7 s), BKT (95.5 s), and TransformerKT (103.2 s). TransformerKT requires approximately 2.9 times more training time than DKT. Given that DKT outperforms TransformerKT by 0.9 AUC percentage points, the Transformer architecture does not justify its additional computational cost in this benchmark.

### E. Stability and Macro-Averaged Performance

Fig. 5 presents the distribution of AUC-ROC scores across folds for each model and dataset. All models exhibit low variance (standard deviations from 0.001 to 0.011). The largest variance is observed on Statics 2011 (N = 282), reflecting greater sampling variability.

Fig. 6 presents the macro-averaged performance. DKT achieves the highest macro-averaged AUC-ROC (0.725), followed by TransformerKT (0.715), SAKT (0.704), and the classical methods (BKT: 0.685, PFA: 0.685). Notably, BKT and PFA achieve identical macro-averaged AUC-ROC despite fundamentally different modeling approaches.

### F. Confidence Intervals

TABLE V presents 95% student-clustered bootstrap confidence intervals for AUC-ROC. The clustered bootstrap yields intervals approximately 2 to 3 times wider than naive interaction-level resampling. Non-overlapping intervals indicate statistically meaningful differences.

**TABLE V**

*95% STUDENT-CLUSTERED BOOTSTRAP CONFIDENCE INTERVALS FOR AUC-ROC*

| Dataset | BKT | PFA | DKT | SAKT | TransformerKT |
|---|---|---|---|---|---|
| ASSISTments 2009 | [.693, .721] | [.688, .714] | [.722, .749] | [.710, .737] | [.714, .742] |
| ASSISTments 2015 | [.662, .675] | [.672, .685] | [.712, .723] | [.696, .708] | [.701, .713] |
| ASSISTments 2017 | [.634, .648] | [.625, .638] | [.670, .684] | [.638, .654] | [.650, .667] |
| Statics 2011 | [.639, .675] | [.643, .678] | [.684, .710] | [.660, .691] | [.675, .704] |
| Algebra 2005 | [.743, .765] | [.742, .760] | [.787, .805] | [.765, .785] | [.784, .803] |

On most datasets, DKT's confidence interval does not overlap with those of BKT and PFA, confirming a statistically meaningful advantage. The intervals are widest on Statics 2011, reflecting the small student population (N = 282). TransformerKT and DKT intervals overlap on most datasets, indicating that the difference between these two models is not statistically significant.

### G. Dataset-Specific Observations

The relative ordering of datasets by difficulty is consistent across all models. Algebra 2005 yields the highest AUC-ROC for every model, likely due to its long interaction sequences (M = 1,070.5 per student). ASSISTments 2017 is the most challenging dataset, with AUC-ROC values 3 to 12 percentage points lower than other datasets, attributable to its low correct rate (0.374) and high skill count (411).

## V. Discussion

### A. Summary of Findings

This benchmark yields three primary findings. First, deep learning methods generally outperform classical approaches on discrimination metrics. DKT achieves a macro-averaged AUC-ROC advantage of approximately 4 percentage points over BKT and PFA. This advantage is consistent for AUC-ROC and PR-AUC; however, differences in accuracy and F1 are smaller and sometimes reverse on individual datasets, suggesting that the benefit of deep learning is most pronounced for ranking predictions rather than at a fixed decision threshold. These results confirm the general finding of Piech et al. [7] while adding nuance about which evaluation aspects benefit most.

Second, among deep learning methods, the LSTM-based DKT achieves the best overall performance. TransformerKT trails DKT by 0.9 AUC-ROC percentage points on average and requires 2.9 times more computation. SAKT trails by 2.0 percentage points. These results are consistent with Gervet et al. [8] in finding that more complex architectures do not necessarily improve KT performance proportionally to their increased cost.

Third, dataset characteristics substantially influence absolute performance but not the relative ordering of methods. All models perform best on Algebra 2005 and worst on ASSISTments 2017, suggesting that dataset properties (sequence length, number of skills, class balance) are at least as important as model architecture.

### B. Classical Methods Revisited

An important finding is that properly implemented classical methods achieve stronger performance than is sometimes reported. With correct forward-backward EM for BKT and proper categorical skill encoding for PFA, both methods achieve macro-averaged AUC-ROC of 0.685. On Algebra 2005, BKT reaches 0.753, within 4.3 percentage points of DKT. These results suggest that some of the performance gap attributed to deep learning in prior work may reflect implementation differences rather than fundamental limitations of classical approaches.

### C. Practical Implications

For applications where computational resources are limited or interpretability is paramount, BKT and PFA remain reasonable choices, particularly on datasets with long interaction sequences. PFA is particularly attractive when training speed is important (M = 14.4 s per fold). For maximum predictive accuracy, DKT offers the best tradeoff between performance and cost. The attention-based models do not provide sufficient gains to justify their increased complexity in most settings.

### D. Limitations

Several limitations should be acknowledged. First, hyperparameters were selected from literature defaults rather than exhaustive search; dataset-specific tuning could alter relative rankings. Second, TransformerKT is a generic baseline and does not implement the full AKT architecture [10]; the original AKT may achieve higher performance. Third, all experiments were conducted on CPU; on GPU hardware, computational cost differences between deep learning models would be smaller. Fourth, the study omits several recent LLM-augmented methods [11], [12]. Fifth, standard metrics treat all predictions equally; mastery-relevant metrics may be more appropriate for some applications. Finally, the common prediction mask excludes first skill encounters, so cold-start performance is not evaluated.

## VI. Conclusion

This study presented a systematic benchmark of five knowledge tracing methods across five educational datasets under a unified evaluation protocol. The results demonstrate that DKT achieves the highest overall performance (macro-averaged AUC-ROC of 0.725) while offering a favorable balance of accuracy and computational cost. Properly implemented classical methods (BKT, PFA) remain competitive, achieving macro-averaged AUC-ROC of 0.685 and offering substantially lower computational requirements and greater interpretability.

The methodological contributions of this work---the common prediction mask ensuring identical evaluation subsets and student-clustered bootstrap confidence intervals---provide tools for more rigorous KT benchmarking. Several directions for future research emerge: extending the benchmark to include LLM-augmented KT methods [11], evaluating fairness metrics across demographic subgroups [25], conducting benchmarks on non-English-language datasets, and incorporating cold-start evaluation protocols.

## Acknowledgment

The author declares no conflicts of interest. This research received no specific funding.

## References

[1] A. T. Corbett and J. R. Anderson, "Knowledge tracing: Modeling the acquisition of procedural knowledge," *User Model. User-Adapted Interact.*, vol. 4, no. 4, pp. 253--278, 1995.

[2] R. S. Baker and P. S. Inventado, "Educational data mining and learning analytics," in *Learning Analytics: From Research to Practice*, J. A. Larusson and B. White, Eds. New York, NY, USA: Springer, 2014, pp. 61--75.

[3] M. V. Yudelson, K. R. Koedinger, and G. J. Gordon, "Individualized Bayesian knowledge tracing models," in *Proc. 16th Int. Conf. Artif. Intell. Educ.*, 2013, pp. 171--180.

[4] G. Rasch, *Probabilistic Models for Some Intelligence and Attainment Tests*. Copenhagen, Denmark: Danmarks Paedagogiske Institut, 1960.

[5] P. I. Pavlik, H. Cen, and K. R. Koedinger, "Performance factors analysis: A new alternative to knowledge tracing," in *Proc. 14th Int. Conf. Artif. Intell. Educ.*, 2009, pp. 531--538.

[6] H. Cen, K. Koedinger, and B. Junker, "Learning factors analysis: A general method for cognitive model evaluation and improvement," in *Proc. 8th Int. Conf. Intell. Tutoring Syst.*, 2006, pp. 164--175.

[7] C. Piech *et al.*, "Deep knowledge tracing," in *Proc. Adv. Neural Inf. Process. Syst.*, vol. 28, 2015, pp. 505--513.

[8] T. Gervet, K. Koedinger, J. Schneider, and T. Mitchell, "When is deep learning the best approach to knowledge tracing?" *J. Educ. Data Mining*, vol. 12, no. 3, pp. 31--54, 2020.

[9] S. Pandey and G. Karypis, "A self-attentive model for knowledge tracing," in *Proc. 12th Int. Conf. Educ. Data Mining*, 2019.

[10] A. Ghosh, N. Heffernan, and A. S. Lan, "Context-aware attentive knowledge tracing," in *Proc. 26th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2020, pp. 2330--2339.

[11] H. Jung, J. Yoo, Y. Yoon, and Y. Jang, "CLST: Cold-start mitigation in knowledge tracing by aligning a generative language model as a students' knowledge tracer," *J. Educ. Data Mining*, vol. 17, no. 2, pp. 86--117, 2025.

[12] K. Neubauer, Y. Rudolph, and U. Brefeld, "Principled transformers for predictive performance in knowledge tracing," *J. Educ. Data Mining*, vol. 18, no. 1, pp. 89--112, 2026.

[13] A. Badrinath and Z. A. Pardos, "Optimizing Bayesian knowledge tracing with neural network parameter generation," *J. Educ. Data Mining*, vol. 17, no. 1, pp. 41--65, 2025.

[14] Z. Liu *et al.*, "pyKT: A Python library to benchmark deep learning based knowledge tracing models," in *Proc. NeurIPS Datasets Benchmarks Track*, 2022.

[15] R. Schmucker, T. Mitchell, and J. Schneider, "Assessing the knowledge state of online students: New data, new approaches, improved accuracy," in *Proc. 17th Int. Conf. Educ. Data Mining*, 2024.

[16] M. Feng, N. T. Heffernan, and K. R. Koedinger, "Addressing the assessment challenge with an online system that tutors as it assesses," *User Model. User-Adapted Interact.*, vol. 19, no. 3, pp. 243--266, 2009.

[17] K. R. Koedinger, E. A. McLaughlin, and N. T. Heffernan, "A quasi-experimental evaluation of an on-line formative assessment and tutoring system," *J. Educ. Comput. Res.*, vol. 43, no. 4, pp. 489--510, 2010.

[18] J. Stamper, A. Niculescu-Mizil, S. Ritter, G. J. Gordon, and K. R. Koedinger, "Algebra I 2005-2006: Challenge data set from KDD Cup 2010 Educational Data Mining Challenge," 2010. [Online]. Available: https://pslcdatashop.web.cmu.edu/KDDCup/

[19] X. Xiong, S. Zhao, E. G. Van Inwegen, and J. E. Beck, "Going deeper with deep knowledge tracing," in *Proc. 9th Int. Conf. Educ. Data Mining*, 2016, pp. 545--550.

[20] L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," *Proc. IEEE*, vol. 77, no. 2, pp. 257--286, 1989.

[21] R. S. J. d. Baker, A. T. Corbett, and V. Aleven, "More accurate student modeling through contextual estimation of slip and guess probabilities in Bayesian knowledge tracing," in *Proc. 9th Int. Conf. Intell. Tutoring Syst.*, 2008, pp. 406--415.

[22] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learn. Representations*, 2015.

[23] R. Pelanek, "Metrics for evaluation of student models," *J. Educ. Data Mining*, vol. 7, no. 2, pp. 1--19, 2015.

[24] J. Davis and M. Goadrich, "The relationship between Precision-Recall and ROC curves," in *Proc. 23rd Int. Conf. Mach. Learn.*, 2006, pp. 233--240.

[25] N. Verger, A. Lissack, H. Gamboa, and F. Rodrigues, "A comprehensive study on evaluating and mitigating algorithmic unfairness with the MADD metric," *J. Educ. Data Mining*, vol. 16, no. 1, 2024.

## Author Biography

[Author biography removed for blind review.]
