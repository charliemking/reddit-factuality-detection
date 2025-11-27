# Logistic Regression Baseline Results

**Model:** TF-IDF + Logistic Regression
**Dataset:** FACTOID (3,193,625 Reddit posts)
**Training Date:** November 2024
**Training Time:** ~21 minutes (CPU)

## Model Configuration

- **Features:** TF-IDF vectors (max 50,000 features, unigrams + bigrams)
- **Classifier:** Logistic Regression (L2 regularization, C=1.0)
- **Class Weights:** Balanced
- **Data Split:** 70% train / 15% validation / 15% test (stratified)
- **Random Seed:** 42

## Performance Summary

### Test Set Performance (477,065 samples)

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 64.11% |
| **Macro F1** | 0.627 |
| **Weighted F1** | 0.641 |
| **ROC-AUC** | 0.671 |

### Per-Class Results

**Non-Factual Posts** (191,763 samples):
- Precision: 55.3%
- Recall: 55.4%
- F1-Score: 0.554

**Factual Posts** (285,302 samples):
- Precision: 70.0%
- Recall: 70.0%
- F1-Score: 0.700

### Confusion Matrix

```
                    Predicted
                Non-Factual  Factual
Actual Non-Fact    106,251   85,512
       Factual      85,715  199,587
```

**Key Observation:** The model shows better performance on factual content (70% F1) compared to non-factual content (55% F1). This may reflect the class imbalance (60% factual vs 40% non-factual in the dataset).

## Feature Analysis

### Top Non-Factual Indicators (Negative Coefficients)

The model strongly associates these terms with non-factual content:

1. **Climate Denial:** "alarmists" (-7.88), "agw" (-4.96), "climate" (-4.21), "cooling" (-4.18), "warming" (-3.61)
2. **Political Bias:** "leader donnie" (-7.22), "orange monster" (-7.10)
3. **Conspiracy Language:** "power establishment" (-4.80), "ipcc" (-4.29)
4. **Divisive Terms:** "feminist" (-3.88), "misandry" (-3.63)

### Top Factual Indicators (Positive Coefficients)

The model associates these with factual content:

1. **Vaccine Discussion:** "novavax" (4.28), "anti vax" (4.26)
2. **Technical References:** "github new" (4.13), "ms mcenany" (5.01)
3. **Specific Names:** "leatherface" (4.24), "catie" (4.16), "twiv" (3.65)

## Key Insights

1. **Baseline Performance:** 64% accuracy provides a reasonable baseline for comparison with more sophisticated models (e.g., DistilBERT).

2. **Style vs. Content:** The model appears to learn associations between language **style** (inflammatory vs. technical) and factuality labels, rather than evaluating factual accuracy directly.

3. **Topic Bias:** Strong associations with specific topics (climate change, politics, vaccines) suggest the model may be learning topic-specific patterns rather than generalizable factuality indicators.

4. **Class Imbalance Effects:** Better performance on the majority class (factual posts) is expected given the 60-40 class distribution.

## Limitations

1. **Surface-Level Features:** TF-IDF captures word frequencies but not semantic meaning or context
2. **No Word Order:** Bigrams provide limited sequential information
3. **Topic Dependence:** May not generalize well to new topics not seen during training
4. **Style Bias:** Inflammatory language != non-factual (and vice versa)

## Next Steps

1. **DistilBERT Comparison:** Evaluate whether transformer-based models improve performance by capturing semantic context
2. **Reuters Validation:** Test whether predictions align with professionally curated news content
3. **Cross-Topic Evaluation:** Assess generalization to unseen topics
4. **Error Analysis:** Examine misclassified examples to understand failure modes

## Files in This Directory

- `logreg_model.pkl` - Trained model (391KB)
- `tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer (1.7MB)
- `test_metrics.json` - Detailed performance metrics
- `classification_report.txt` - Per-class performance breakdown
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve
- `top_features.csv` - Most influential features ranked by coefficient magnitude
- `test_predictions.csv` - Model predictions on test set (24MB, 477,065 rows)

## Reproducibility

To reproduce these results:

```bash
python -m src.train_logreg \
    --factoid_path data/factoid_clean.csv \
    --output_dir results/logreg_full \
    --max_features 50000 \
    --ngram_range 1 2 \
    --random_seed 42
```

## Conclusion

This TF-IDF + Logistic Regression baseline achieves **64.11% accuracy** on 477K test samples, demonstrating that simple lexical features capture some patterns of factuality in Reddit posts. However, the strong association with specific topics and inflammatory language suggests the model is learning superficial patterns. The upcoming DistilBERT transformer model should provide insights into whether semantic understanding improves factuality detection beyond these lexical patterns.
