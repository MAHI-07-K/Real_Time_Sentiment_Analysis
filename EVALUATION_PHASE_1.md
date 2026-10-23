# Evaluation Phase 1 - Baseline Sentiment Ensemble

## Objective

Evaluate the baseline sentiment-classification models on the held-out test set and select the strongest production model.

## Evaluation setup

- Dataset split: 70% training, 15% validation, and 15% testing, using stratified sampling.
- Test-set size: 29,386 text samples.
- Features: TF-IDF unigrams and bigrams, limited to 20,000 features.
- Models tested: Multinomial Naive Bayes, calibrated linear SVM, LightGBM, and their weighted soft-voting ensemble.
- Primary metric: macro-F1, chosen to give equal importance to negative, neutral, and positive classes.
- Data-leakage control: the TF-IDF vectorizer is fitted only on training data; validation and test data are transformed with that saved vocabulary.

## Results

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Naive Bayes | 73.93% | 70.76% |
| Calibrated SVM | 94.61% | 94.03% |
| LightGBM | 93.93% | 93.31% |
| **Weighted ensemble** | **94.75%** | **94.15%** |

The ensemble achieved the strongest overall performance, so it is selected as the Phase 1 baseline model.

## Ensemble design

The ensemble combines class probabilities from all three models. Weights are derived from each model's validation macro-F1 score:

| Model | Weight |
| --- | ---: |
| Naive Bayes | 27.50% |
| SVM | 36.43% |
| LightGBM | 36.07% |

## Class-level performance of the selected ensemble

| Sentiment label | Precision | Recall | F1-score | Test samples |
| --- | ---: | ---: | ---: | ---: |
| Negative (-1) | 94.39% | 86.92% | 90.51% | 6,470 |
| Neutral (0) | 95.28% | 97.92% | 96.58% | 9,920 |
| Positive (1) | 94.51% | 96.23% | 95.36% | 12,996 |

The neutral and positive classes are identified very reliably. Negative sentiment has the lowest recall, meaning some negative comments are predicted as neutral or positive; this is the main area for improvement in later evaluation phases.

## Confusion matrix

Rows are actual labels and columns are predicted labels.

| Actual \ Predicted | Negative | Neutral | Positive |
| --- | ---: | ---: | ---: |
| Negative | 5,624 | 270 | 576 |
| Neutral | 55 | 9,714 | 151 |
| Positive | 279 | 211 | 12,506 |

## Conclusion

Phase 1 establishes a high-performing baseline: the weighted ensemble produces 94.75% accuracy and 94.15% macro-F1 on unseen test data. The baseline model, vectorizer, weights, and complete machine-readable metrics are saved in `models/` as `*.pkl`, `ensemble_weights.json`, and `evaluation.json`.
