# Benchmarking knowledge tracing methods across five educational datasets: a comparative study of Bayesian, logistic, and deep learning approaches

Irakli Matcharashvili

## Abstract

Knowledge tracing (KT), the task of modeling student knowledge over time to predict future performance, is a foundational problem in educational data mining. Despite the rapid proliferation of deep learning approaches to KT, no recent comprehensive benchmark has systematically compared classical and neural methods across multiple datasets using a consistent evaluation protocol. This study addresses that gap by benchmarking five KT methods spanning three methodological generations: Bayesian Knowledge Tracing (BKT; Corbett & Anderson, 1995), Performance Factors Analysis (PFA; Pavlik et al., 2009), Deep Knowledge Tracing (DKT; Piech et al., 2015), Self-Attentive Knowledge Tracing (SAKT; Pandey & Karypis, 2019), and a generic causal Transformer baseline (TransformerKT). All models were evaluated on five publicly available benchmark datasets (ASSISTments 2009, ASSISTments 2015, ASSISTments 2017, Statics 2011, and Algebra 2005) using 5-fold student-level cross-validation with five metrics: AUC-ROC, PR-AUC, accuracy, F1 score, and RMSE. Critically, all models were evaluated on the identical prediction subset, defined by excluding each student's first encounter with each skill. Results indicate that deep learning methods consistently outperform classical approaches, with DKT achieving the highest macro-averaged AUC-ROC (M = 0.725) compared to BKT (M = 0.685) and PFA (M = 0.685). TransformerKT (M = 0.715) closely trails DKT, while SAKT (M = 0.704) shows more modest gains over classical methods. The performance gap between classical and deep learning models ranges from 2.9 to 4.9 AUC percentage points depending on the dataset. Classical methods remain competitive on datasets with long interaction sequences (e.g., Algebra 2005, where BKT achieves AUC = 0.753) and offer substantially lower computational costs. These findings provide practitioners with empirically grounded guidance for model selection in knowledge tracing applications.

**Keywords:** knowledge tracing, educational data mining, deep learning, benchmark, Bayesian knowledge tracing, self-attention

## Introduction

Understanding what students know and predicting what they will learn next is a central challenge in educational technology and learning science. Knowledge tracing (KT) refers to the computational task of modeling a student's evolving knowledge state based on their sequence of interactions with learning materials (Corbett & Anderson, 1995). Accurate KT models have practical implications for adaptive learning systems, as they enable personalized content selection, timely intervention, and mastery-based progression through curricula (Baker & Inventado, 2014).

The earliest and most influential KT framework is Bayesian Knowledge Tracing (BKT), introduced by Corbett and Anderson (1995). BKT models each skill as a two-state Hidden Markov Model, with parameters governing the initial probability of mastery, the probability of learning, and the probabilities of guessing and slipping. Despite its simplicity, BKT remained the dominant approach for nearly two decades and has been extended in numerous ways, including student-level individualization (Yudelson et al., 2013) and integration with item response theory (IRT; Rasch, 1960).

A parallel line of work explored logistic regression-based methods for knowledge tracing. Performance Factors Analysis (PFA; Pavlik et al., 2009) models student performance as a function of skill-specific intercepts and cumulative success and failure counts per skill, drawing on Learning Factors Analysis (Cen et al., 2006). PFA offers computational efficiency and interpretability, though it does not model latent knowledge states explicitly.

The introduction of Deep Knowledge Tracing (DKT) by Piech et al. (2015) marked a paradigm shift in the field. DKT uses a recurrent neural network (specifically, an LSTM) to process sequences of student interactions and predict future performance. Piech et al. (2015) reported substantial improvements over BKT on the ASSISTments dataset, sparking an extensive body of follow-up work. However, DKT has been criticized for issues including non-interpretability and inconsistencies in predicted knowledge states over time (Gervet et al., 2020).

Subsequent deep learning approaches sought to address these limitations while leveraging attention mechanisms from natural language processing. Pandey and Karypis (2019) proposed Self-Attentive Knowledge Tracing (SAKT), which replaces the recurrent architecture with a self-attention mechanism that directly attends to relevant past interactions. Ghosh et al. (2020) extended this idea with context-aware Attentive Knowledge Tracing (AKT), incorporating a monotonic attention mechanism and Rasch-model-inspired embeddings to capture both exercise difficulty and student ability.

More recently, researchers have begun exploring large language models for knowledge tracing. Jung et al. (2025) proposed CLST, which aligns a generative language model as a student knowledge tracer to address cold-start problems. Neubauer et al. (2026) investigated principled transformer architectures for predictive performance in knowledge tracing. Badrinath and Pardos (2025) explored neural network parameter generation to optimize BKT. These developments suggest a trajectory toward increasingly complex architectures for KT.

Despite this proliferation of methods, a critical question persists: do the gains from newer, more complex architectures justify their increased computational cost and reduced interpretability? Gervet et al. (2020) raised this question directly, finding that simpler baselines could match deep learning approaches on some datasets when properly tuned. Their work, however, predated several attention-based models and did not include the full range of datasets and methods available today.

This study contributes to the ongoing evaluation of KT methods by providing a systematic benchmark with the following characteristics: (a) five models spanning three methodological generations (Bayesian, logistic, and deep learning), (b) five publicly available datasets commonly used in the KT literature, (c) a consistent evaluation protocol using student-level 5-fold cross-validation with a common prediction mask to prevent data leakage and ensure fair comparison, (d) five complementary metrics capturing discrimination (AUC-ROC), ranking under class imbalance (PR-AUC), prediction error (RMSE), classification performance (accuracy, F1), and computational efficiency (training time), and (e) 95% bootstrap confidence intervals for primary metrics. The goal is to provide the research community with a transparent, reproducible comparison that can inform model selection decisions for both researchers and practitioners.

## Method

### Datasets

Five publicly available educational datasets were selected for this benchmark, chosen for their widespread use in prior KT research and their diversity in terms of domain, scale, and student population characteristics. Table 1 presents descriptive statistics for each dataset.

**Table 1**

*Descriptive statistics for the five benchmark datasets*

| Dataset | Interactions | Students | Items | Skills | Correct rate | Avg. interactions per student |
|---|---|---|---|---|---|---|
| ASSISTments 2009 | 278,336 | 3,114 | 17,708 | 149 | 0.659 | 89.4 |
| ASSISTments 2015 | 656,154 | 14,228 | 100 | 100 | 0.730 | 46.1 |
| ASSISTments 2017 | 934,638 | 1,708 | 3,162 | 411 | 0.374 | 547.2 |
| Statics 2011 | 189,297 | 282 | 1,223 | 98 | 0.765 | 671.3 |
| Algebra 2005 | 606,983 | 567 | 173,113 | 271 | 0.755 | 1,070.5 |

The ASSISTments datasets (2009, 2015, and 2017) originate from the ASSISTments online tutoring platform, which serves middle and high school mathematics students (Feng et al., 2009). The three versions differ substantially in their characteristics: ASSISTments 2009 contains a moderate number of students with a balanced correct rate; ASSISTments 2015 has the most students but the fewest items per skill; and ASSISTments 2017 has the lowest correct rate (0.374), indicating substantially more challenging content or a different student population.

The Statics 2011 dataset was collected from a college-level engineering statics course hosted on the Open Learning Initiative platform (Koedinger et al., 2010). It features the smallest student population (N = 282) but the longest average interaction sequences (M = 671.3 interactions per student), providing a unique test of model performance on long temporal sequences with limited student diversity.

The Algebra 2005 dataset originates from the KDD Cup 2010 challenge and was collected from high school algebra courses using the Cognitive Tutor system (Stamper et al., 2010). It contains the most items (173,113) and the longest average sequences (M = 1,070.5 interactions per student), presenting a challenge for models that must generalize across a large item space.

All datasets were obtained in preprocessed form from a publicly available standardized repository (https://github.com/theophilee/learner-performance-prediction), which provides a consistent schema with five columns: user_id, item_id, timestamp, correct, and skill_id. This repository has been widely used in KT benchmarking studies. No missing values were present in any dataset after preprocessing.

### Data preprocessing

Prior to model training, several preprocessing steps were applied to ensure comparability across datasets and models.

**Minimum interaction threshold.** Students with fewer than five interactions were excluded from the analysis. This threshold follows the convention established by Piech et al. (2015) and ensures sufficient temporal signal for sequence-based models. This filter removed 7 students from ASSISTments 2009 and no students from the remaining datasets.

**ID remapping.** Skill and item identifiers were remapped to contiguous integers starting from zero within each dataset. This step is necessary for embedding-based models (DKT, SAKT, TransformerKT) that use skill and item IDs as indices into learnable embedding matrices.

**Sequence construction.** For deep learning models, student interaction sequences were constructed by ordering each student's interactions by timestamp and segmenting into non-overlapping windows of at most 200 interactions. Subsequences shorter than three interactions were discarded. The maximum sequence length of 200 was chosen to balance computational cost with the ability to capture long-range dependencies, consistent with the settings used by Pandey and Karypis (2019). Classical models (BKT, PFA) received the full, unchunked interaction histories ordered by student and timestamp, as these models do not require fixed-length sequences and benefit from access to each student's complete history.

**Train-test splitting.** A 5-fold cross-validation procedure was employed with splits performed at the student level. All interactions of a given student appear in either the training set or the test set, never both. This student-level splitting prevents data leakage that would occur if interactions from the same student appeared in both sets, an issue noted by Xiong et al. (2016) as a common source of inflated performance estimates in KT research.

**Common prediction mask.** To ensure that all models are evaluated on exactly the same set of predictions, a common prediction mask was applied at test time. This mask excludes each student's first encounter with each skill, since no model can meaningfully predict performance on a skill the student has not yet been observed to attempt. Additionally, the mask excludes chunk-boundary positions (i.e., positions 0, 200, 400, etc. within each student's history) because deep learning models, which process non-overlapping chunks of 200 interactions, produce no next-step prediction for the first position of each chunk. Excluding these positions from all models ensures that classical and DL approaches are scored on an identical evaluation subset. Only interactions passing both criteria are included in metric computation. This approach eliminates confounds arising from different models handling cold-start interactions or sequence boundaries differently.

### Models

Five KT models were evaluated, representing three generations of approaches.

**Bayesian Knowledge Tracing (BKT).** BKT was implemented following the original formulation by Corbett and Anderson (1995). Each skill is modeled as a two-state Hidden Markov Model with four parameters: the initial probability of mastery (p(L_0)), the probability of transitioning from unlearned to learned (p(T)), the probability of a correct response despite not having mastered the skill (p(G), "guessing"), and the probability of an incorrect response despite mastery (p(S), "slipping"). The learned state is absorbing (no forgetting). Parameters were estimated using the forward-backward algorithm for EM (Rabiner, 1989), with a convergence tolerance of 1e-4 and a maximum of 50 iterations per skill. Guessing and slipping probabilities were constrained to [0.001, 0.40] following standard identifiability constraints (Baker et al., 2008). For skills with fewer than five student sequences, population-average parameters (mean across all fitted skills) were used as a fallback. BKT was trained on full, unchunked interaction histories, grouped by skill and then by student within each skill.

**Performance Factors Analysis (PFA).** PFA was implemented following the standard formulation by Pavlik et al. (2009) as a logistic regression with three categories of features: (a) a skill-specific intercept via one-hot encoding of skill identity (N_skills binary features), (b) skill-gated cumulative success counts (N_skills features, where column k contains the running count of prior correct responses on skill k for each student, and zero for all other skills), and (c) skill-gated cumulative failure counts (N_skills features, structured analogously). This yields a feature matrix with 3 * N_skills columns, implemented as a sparse matrix for memory efficiency. The model was fit using L-BFGS optimization with L2 regularization (C = 1.0) and a maximum of 5,000 iterations. PFA was trained on full, unchunked interaction histories.

**Deep Knowledge Tracing (DKT).** DKT was implemented following Piech et al. (2015) using a single-layer LSTM with 100 hidden units. The input at each timestep is a one-hot encoding of the interaction (skill ID combined with correctness), resulting in an input dimension of 2 * N_skills. The LSTM output is passed through a dropout layer (p = 0.2) and a linear projection to N_skills dimensions, followed by a sigmoid activation to produce per-skill correctness probabilities. The model is trained to predict the correctness of the next interaction in the sequence. Training used the Adam optimizer (Kingma & Ba, 2015) with a learning rate of 0.001, batch size of 64, and gradient clipping at norm 1.0. The number of training epochs was adapted to dataset size: 15 epochs for smaller datasets (ASSISTments 2009, Statics 2011), 10 for medium datasets (ASSISTments 2015, Algebra 2005), and 8 for the largest dataset (ASSISTments 2017).

**Self-Attentive Knowledge Tracing (SAKT).** SAKT was implemented following Pandey and Karypis (2019). The model uses an embedding dimension of 64, with separate embeddings for interactions (skill combined with correctness) and skills (for query generation). Positional embeddings are added to interaction embeddings to encode temporal order. A single-head attention mechanism with causal masking computes attention weights over past interactions to predict the next response. The architecture includes layer normalization and a position-wise feedforward network with an expansion factor of 4. Dropout (p = 0.2) is applied after the attention and feedforward layers. Training hyperparameters (optimizer, learning rate, batch size, gradient clipping, epoch schedule) matched those used for DKT.

**TransformerKT.** To evaluate whether a generic Transformer encoder improves over specialized attention mechanisms, we include a causal Transformer baseline (TransformerKT). The model uses a Transformer encoder with 2 layers, 4 attention heads, and an embedding dimension of 64. Inputs consist of the sum of skill embeddings, correctness embeddings (binary), and positional embeddings. Causal masking ensures that predictions at each position attend only to preceding interactions. The encoded context at position t is concatenated with the skill embedding at position t+1, and this combined representation is passed through a two-layer feedforward network with ReLU activation to produce the correctness prediction. Training used the Adam optimizer with a learning rate of 0.001. Other hyperparameters matched those used for DKT and SAKT. We note that this model is a generic Transformer baseline and does not implement the Rasch-model-based embeddings or monotonic attention mechanism of the full AKT architecture (Ghosh et al., 2020).

### Evaluation metrics

Five complementary metrics were used to evaluate model performance, all computed on the common prediction mask.

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve).** AUC-ROC measures the probability that a model ranks a randomly chosen correct response higher than a randomly chosen incorrect response. It is threshold-independent and robust to class imbalance, making it the primary evaluation metric in KT research (Pelanek, 2015).

**PR-AUC (Area Under the Precision-Recall Curve).** PR-AUC is particularly informative for datasets with imbalanced class distributions. Unlike AUC-ROC, PR-AUC focuses on the positive class and is more sensitive to differences in model performance when the majority of interactions are correct (or incorrect), as is the case for several of our datasets (Davis & Goadrich, 2006).

**Accuracy.** The proportion of correct predictions using a classification threshold of 0.5.

**F1 score.** The harmonic mean of precision and recall, also using a threshold of 0.5.

**RMSE (Root Mean Square Error).** RMSE measures the average magnitude of prediction errors, computed as the square root of the mean squared difference between predicted probabilities and binary outcomes. Lower RMSE indicates predictions that are closer to the observed outcomes on average. RMSE is sensitive to extreme predictions and penalizes overconfident incorrect predictions more heavily than near-boundary errors.

All metrics were computed on held-out test data for each fold using the common prediction mask, and we report means and standard deviations across the five folds. Additionally, 95% student-clustered bootstrap confidence intervals (1,000 bootstrap samples) are reported for AUC-ROC and PR-AUC to facilitate pairwise model comparisons. The clustered bootstrap resamples students with replacement (rather than individual interactions), correctly accounting for within-student correlation and producing appropriately conservative intervals.

### Reproducibility

Full seed control was implemented to ensure reproducibility. Random seeds were fixed for NumPy, PyTorch (including CUDA and cuDNN deterministic mode), and DataLoader worker initialization. The complete analysis code, including data preprocessing, model implementations, and evaluation scripts, is available at https://github.com/iraklimatch/Benchmarking-knowledge-tracing-methods.

### Computational environment

All experiments were conducted on a consumer-grade machine running macOS (Darwin 25.3.0) with a 3.9 GHz CPU. Deep learning models were trained on CPU using PyTorch 2.8.0. No GPU acceleration was used, providing a realistic assessment of computational costs for researchers without access to specialized hardware.

## Results

### Overall performance comparison

Table 2 presents the mean AUC-ROC and standard deviation across five folds for all model-dataset combinations. DKT achieved the highest AUC-ROC on all five datasets. Figure 1 provides a visual comparison of AUC-ROC scores with error bars representing the standard deviation across folds.

**Table 2**

*Mean AUC-ROC (and standard deviation) by model and dataset, based on 5-fold student-level cross-validation*

| Dataset | BKT | PFA | DKT | SAKT | TransformerKT |
|---|---|---|---|---|---|
| ASSISTments 2009 | .706 (.005) | .701 (.005) | **.735 (.006)** | .723 (.006) | .728 (.006) |
| ASSISTments 2015 | .668 (.002) | .678 (.001) | **.717 (.001)** | .702 (.002) | .707 (.002) |
| ASSISTments 2017 | .641 (.003) | .631 (.004) | **.677 (.003)** | .646 (.003) | .659 (.004) |
| Statics 2011 | .656 (.006) | .661 (.007) | **.698 (.011)** | .676 (.009) | .690 (.010) |
| Algebra 2005 | .753 (.007) | .751 (.008) | **.796 (.006)** | .775 (.007) | .793 (.007) |
| **Macro avg.** | **.685** | **.685** | **.725** | **.704** | **.715** |

*Note.* Bold values indicate the best-performing model for each dataset. All standard deviations are reported in parentheses. The macro average row reports the unweighted mean across datasets.

Several patterns are evident. First, deep learning methods (DKT, SAKT, TransformerKT) consistently outperform classical methods (BKT, PFA) across all datasets. Second, the performance gap between classical and deep learning approaches varies by dataset, ranging from 2.9 AUC percentage points on ASSISTments 2009 (BKT: .706, DKT: .735) to 4.9 points on ASSISTments 2015 (BKT: .668, DKT: .717). Third, among deep learning methods, DKT generally achieves the highest AUC-ROC, with TransformerKT as a close second and SAKT behind both.

### PR-AUC comparison

Figure 2 presents the PR-AUC results. Table 3 includes PR-AUC alongside other metrics. PR-AUC is particularly revealing for ASSISTments 2017, where the 37.4% correct rate makes this metric more sensitive to model differences than AUC-ROC. On this dataset, all models achieve PR-AUC below 0.57, with DKT (0.561) outperforming the next-best model (TransformerKT: 0.547) by a wider margin than observed in AUC-ROC. On datasets with higher correct rates (Algebra 2005: 75.5% correct), PR-AUC values are uniformly high (0.90 to 0.92) and model differences are compressed.

### Multi-metric comparison

Table 3 presents the complete multi-metric results across all datasets and models.

**Table 3**

*Complete benchmark results: mean values across 5-fold cross-validation (common prediction mask applied)*

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

Several observations emerge from the multi-metric comparison. While AUC-ROC differences between models are substantial, accuracy and F1 differences are considerably smaller. For example, on Statics 2011, the AUC-ROC difference between BKT (0.657) and DKT (0.698) is 4.1 percentage points, but the accuracy difference is only 0.2 percentage points (0.768 vs. 0.770). This divergence occurs because accuracy is sensitive to the classification threshold and the base rate of correctness. On datasets with high correct rates (Statics 2011: 76.5%, Algebra 2005: 75.5%), a model that predicts the majority class achieves high accuracy without learning meaningful patterns.

The ASSISTments 2017 dataset presents a notable anomaly. PFA achieves an F1 score of only 0.258, far below all other models. This occurs because ASSISTments 2017 has the lowest correct rate (37.4%), and PFA's logistic regression model predicts most interactions as incorrect, resulting in very low recall for the positive class. BKT and the deep learning models handle this imbalance somewhat more effectively, though F1 remains low for all models on this dataset.

Figure 3 presents the RMSE comparison across all datasets, showing that DKT and TransformerKT achieve the lowest RMSE on every dataset.

### Computational efficiency

Figure 4 and Table 4 present the training time per fold across all datasets.

**Table 4**

*Average training time per fold in seconds*

| Model | ASSISTments 2009 | ASSISTments 2015 | ASSISTments 2017 | Statics 2011 | Algebra 2005 | Overall mean |
|---|---|---|---|---|---|---|
| BKT | 45.3 | 141.5 | 194.3 | 24.8 | 71.7 | 95.5 |
| PFA | 3.8 | 1.4 | 17.3 | 11.0 | 38.4 | 14.4 |
| DKT | 33.6 | 79.5 | 33.9 | 9.2 | 22.3 | 35.7 |
| SAKT | 26.9 | 64.5 | 19.9 | 7.3 | 15.0 | 26.7 |
| TransformerKT | 98.1 | 253.3 | 77.2 | 28.8 | 58.6 | 103.2 |

Training time varies substantially by model and dataset. PFA is the fastest model overall (M = 14.4 s per fold), followed by SAKT (M = 26.7 s), DKT (M = 35.7 s), BKT (M = 95.5 s), and TransformerKT (M = 103.2 s). BKT training time is dominated by the forward-backward EM algorithm applied independently to each skill; datasets with many skills (ASSISTments 2017: 411 skills) or many students (ASSISTments 2015: 14,228 students) are particularly expensive. TransformerKT is the most computationally expensive model, requiring approximately 2.9 times more training time than DKT on average. When considering the marginal AUC-ROC improvement of TransformerKT over DKT (0.715 vs. 0.725 in macro average, a gap of 0.9 percentage points in favor of DKT), the Transformer architecture does not justify its additional computational cost.

### Stability across folds

Figure 5 presents the distribution of AUC-ROC scores across folds for each model, shown separately for each dataset. All models exhibit relatively low variance across folds (standard deviations ranging from 0.001 to 0.010), indicating stable performance under cross-validation. DKT shows consistently low variance across datasets. The largest fold-to-fold variance is observed on Statics 2011, which has the smallest student population (N = 282), leading to greater sampling variability in the fold splits.

### Macro-averaged performance

Figure 6 presents the macro-averaged performance across all five datasets. The macro average gives equal weight to each dataset regardless of size, providing a summary that is not dominated by the larger datasets. DKT achieves the highest macro-averaged AUC-ROC (0.725), followed by TransformerKT (0.715), SAKT (0.704), and the classical methods (BKT: 0.685, PFA: 0.685). Notably, BKT and PFA achieve identical macro-averaged AUC-ROC despite employing fundamentally different modeling approaches, suggesting that they capture complementary aspects of student performance.

### Confidence intervals

Table 5 presents 95% student-clustered bootstrap confidence intervals for AUC-ROC, averaged across folds. The clustered bootstrap resamples students with replacement, accounting for within-student dependence and yielding intervals that are approximately 2 to 3 times wider than naive interaction-level resampling. These intervals support formal pairwise model comparisons: non-overlapping intervals indicate a statistically meaningful difference at the 95% level.

**Table 5**

*95% student-clustered bootstrap confidence intervals for AUC-ROC (mean across 5 folds)*

| Dataset | BKT | PFA | DKT | SAKT | TransformerKT |
|---|---|---|---|---|---|
| ASSISTments 2009 | [.693, .721] | [.688, .714] | [.722, .749] | [.710, .737] | [.714, .742] |
| ASSISTments 2015 | [.662, .675] | [.672, .685] | [.712, .723] | [.696, .708] | [.701, .713] |
| ASSISTments 2017 | [.634, .648] | [.625, .638] | [.670, .684] | [.638, .654] | [.650, .667] |
| Statics 2011 | [.639, .675] | [.643, .678] | [.684, .710] | [.660, .691] | [.675, .704] |
| Algebra 2005 | [.743, .765] | [.742, .760] | [.787, .805] | [.765, .785] | [.784, .803] |

*Note.* Intervals are the mean of per-fold bootstrap intervals (1,000 student-level resamples per fold). Non-overlapping intervals between two models on a given dataset indicate a statistically meaningful performance difference at the 95% confidence level.

### Dataset-specific observations

The relative ordering of datasets by difficulty is consistent across all models. Algebra 2005 yields the highest AUC-ROC for every model, likely due to its long interaction sequences (M = 1,070.5 per student) which provide rich temporal signal. In contrast, ASSISTments 2017 is the most challenging dataset for all models, with AUC-ROC values 3 to 12 percentage points lower than on other datasets. This is likely attributable to its low correct rate (0.374) and high number of skills (411), which create a sparse skill-response matrix.

## Discussion

### Summary of findings

This benchmark study compared five knowledge tracing methods across five datasets using a unified evaluation protocol, yielding three primary findings.

First, deep learning methods generally outperform classical approaches on discrimination metrics. DKT achieves a macro-averaged AUC-ROC advantage of approximately 4 percentage points over both BKT and PFA. This advantage is consistent across all five datasets for AUC-ROC and PR-AUC; however, differences in accuracy and F1 are smaller and sometimes reverse on individual datasets, suggesting that the benefit of DL is most pronounced when ranking predictions rather than when applying a fixed decision threshold. These results confirm the general finding of Piech et al. (2015) that recurrent neural networks capture patterns in student interaction sequences that classical approaches do not, while adding nuance about which evaluation aspects benefit most.

Second, among deep learning methods, the LSTM-based DKT achieves the best overall performance. TransformerKT, despite having a more expressive architecture with multi-head self-attention, trails DKT by 0.9 AUC-ROC percentage points on average and requires 2.9 times more computation. SAKT trails by 2.0 percentage points. These results are consistent with the findings of Gervet et al. (2020) that more complex architectures do not necessarily improve KT performance proportionally to their increased cost.

Third, the choice of dataset substantially influences absolute performance levels but not the relative ordering of methods. All models perform best on Algebra 2005 and worst on ASSISTments 2017, suggesting that dataset characteristics (sequence length, number of skills, class balance) are at least as important as model architecture in determining KT performance.

### Classical methods revisited

An important finding of this study is that properly implemented classical methods achieve stronger performance than is sometimes reported in the literature. With correct forward-backward EM for BKT (applied to full, unchunked interaction histories) and proper categorical skill encoding for PFA, both methods achieve macro-averaged AUC-ROC of 0.685. On Algebra 2005, BKT reaches AUC = 0.753, within 4.3 percentage points of DKT. These results suggest that some of the performance gap attributed to deep learning in prior work may reflect implementation differences rather than fundamental limitations of classical approaches. Researchers should be cautious about concluding that deep learning is universally superior without ensuring fair comparison conditions, including appropriate data representations for each model family.

### Practical implications

The results have several implications for practitioners deploying KT in educational systems. For applications where computational resources are limited or interpretability is paramount, BKT and PFA remain reasonable choices, particularly on datasets with long interaction sequences (e.g., Algebra 2005, where BKT achieves AUC-ROC of 0.753). PFA is particularly attractive when training speed is important, averaging only 14 seconds per fold.

For applications requiring the highest prediction accuracy, DKT offers the best tradeoff between performance and computational cost. The attention-based models (SAKT, TransformerKT) do not provide sufficient performance gains to justify their increased complexity in most settings. TransformerKT achieves performance close to DKT on some datasets (Algebra 2005: 0.793 vs. 0.796) but requires substantially more training time.

### Limitations

Several limitations should be acknowledged. First, the hyperparameters for each model were selected based on common defaults from the literature rather than through exhaustive grid search. It is possible that dataset-specific tuning could alter the relative rankings, particularly for attention-based models that may be more sensitive to hyperparameter choices.

Second, the TransformerKT model is a generic causal Transformer baseline and does not implement the full AKT architecture (Ghosh et al., 2020), which includes Rasch-model-based embeddings and a monotonic attention mechanism. The original AKT implementation may achieve higher performance. Future work should include the complete AKT architecture.

Third, all experiments were conducted on CPU, which disproportionately affects larger models. On GPU hardware, the computational cost differences between DKT, SAKT, and TransformerKT would likely be smaller, potentially shifting the cost-benefit analysis.

Fourth, this study does not include several recent methods, including transformer-based approaches that leverage pre-trained language models (Jung et al., 2025; Neubauer et al., 2026). Future work should extend this benchmark to include LLM-augmented KT methods, which represent a rapidly evolving frontier in the field.

Fifth, performance was evaluated using standard metrics (AUC-ROC, PR-AUC, accuracy, F1, RMSE) that treat all predictions equally. In practice, the predictions that matter most are those at decision boundaries, such as determining whether a student has achieved mastery. Future benchmarks should consider mastery-relevant metrics.

Finally, the common prediction mask excludes first encounters with each skill, which reduces the total number of evaluated predictions. While this ensures fair comparison, it also means that cold-start performance is not evaluated. Models that excel at predicting performance on novel skills may not be identified by this evaluation protocol.

### Future directions

Several directions for future research emerge from this work. First, extending the benchmark to include LLM-augmented KT methods (e.g., CLST; Jung et al., 2025) would provide a more complete picture of the field. Second, evaluating fairness metrics across demographic subgroups would address growing concerns about algorithmic bias in educational AI (Verger et al., 2024). Third, conducting benchmarks on non-English-language and non-Western datasets would improve the generalizability of findings. Fourth, incorporating cold-start evaluation protocols would test model robustness in realistic deployment scenarios where new students have limited interaction histories.

## References

Badrinath, A., & Pardos, Z. A. (2025). Optimizing Bayesian knowledge tracing with neural network parameter generation. *Journal of Educational Data Mining, 17*(1), 41--65.

Baker, R. S. J. d., Corbett, A. T., & Aleven, V. (2008). More accurate student modeling through contextual estimation of slip and guess probabilities in Bayesian knowledge tracing. In *Proceedings of the 9th International Conference on Intelligent Tutoring Systems* (pp. 406--415). Springer.

Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. In J. A. Larusson & B. White (Eds.), *Learning analytics: From research to practice* (pp. 61--75). Springer.

Cen, H., Koedinger, K., & Junker, B. (2006). Learning factors analysis: A general method for cognitive model evaluation and improvement. In *Proceedings of the 8th International Conference on Intelligent Tutoring Systems* (pp. 164--175). Springer.

Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction, 4*(4), 253--278.

Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233--240). ACM.

Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). Addressing the assessment challenge with an online system that tutors as it assesses. *User Modeling and User-Adapted Interaction, 19*(3), 243--266.

Gervet, T., Koedinger, K., Schneider, J., & Mitchell, T. (2020). When is deep learning the best approach to knowledge tracing? *Journal of Educational Data Mining, 12*(3), 31--54.

Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-aware attentive knowledge tracing. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 2330--2339). ACM.

Jung, H., Yoo, J., Yoon, Y., & Jang, Y. (2025). CLST: Cold-start mitigation in knowledge tracing by aligning a generative language model as a students' knowledge tracer. *Journal of Educational Data Mining, 17*(2), 86--117.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations*.

Koedinger, K. R., McLaughlin, E. A., & Heffernan, N. T. (2010). A quasi-experimental evaluation of an on-line formative assessment and tutoring system. *Journal of Educational Computing Research, 43*(4), 489--510.

Neubauer, K., Rudolph, Y., & Brefeld, U. (2026). Principled transformers for predictive performance in knowledge tracing. *Journal of Educational Data Mining, 18*(1), 89--112.

Pandey, S., & Karypis, G. (2019). A self-attentive model for knowledge tracing. In *Proceedings of the 12th International Conference on Educational Data Mining*.

Pavlik, P. I., Cen, H., & Koedinger, K. R. (2009). Performance factors analysis: A new alternative to knowledge tracing. In *Proceedings of the 14th International Conference on Artificial Intelligence in Education* (pp. 531--538). IOS Press.

Pelanek, R. (2015). Metrics for evaluation of student models. *Journal of Educational Data Mining, 7*(2), 1--19.

Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. In *Advances in Neural Information Processing Systems 28* (pp. 505--513).

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE, 77*(2), 257--286.

Rasch, G. (1960). *Probabilistic models for some intelligence and attainment tests*. Danmarks Paedagogiske Institut.

Stamper, J., Niculescu-Mizil, A., Ritter, S., Gordon, G. J., & Koedinger, K. R. (2010). Algebra I 2005-2006: Challenge data set from KDD Cup 2010 Educational Data Mining Challenge. https://pslcdatashop.web.cmu.edu/KDDCup/

Verger, N., Lissack, A., Gamboa, H., & Rodrigues, F. (2024). A comprehensive study on evaluating and mitigating algorithmic unfairness with the MADD metric. *Journal of Educational Data Mining, 16*(1).

Xiong, X., Zhao, S., Van Inwegen, E. G., & Beck, J. E. (2016). Going deeper with deep knowledge tracing. In *Proceedings of the 9th International Conference on Educational Data Mining* (pp. 545--550).

Yudelson, M. V., Koedinger, K. R., & Gordon, G. J. (2013). Individualized Bayesian knowledge tracing models. In *Proceedings of the 16th International Conference on Artificial Intelligence in Education* (pp. 171--180). Springer.
