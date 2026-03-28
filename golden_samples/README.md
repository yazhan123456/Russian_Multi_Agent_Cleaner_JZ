# Minimal Golden Sample Set

This directory is the first lightweight quality baseline for the Russian document cleaning pipeline.

## Recommended directory structure

```text
golden_samples/
  README.md
  samples/
    sample_001_body_clean/
      metadata.json
      raw.txt
      expected.txt
      actual.txt
    sample_002_heading_list/
      metadata.json
      raw.txt
      expected.txt
      actual.txt
```

## First 20-30 pages to cover

Pick pages from at least these classes:

1. Normal body page
2. Heading/list-heavy page
3. Notes/reference-polluted page
4. OCR error-prone page
5. Structure-recovery page

## `metadata.json` fields

Suggested fields:

```json
{
  "expected_page_type": "body_only",
  "predicted_page_type": "body_with_notes",
  "keywords": ["корпорация", "участник", "право"],
  "protected_spans": ["Глава IX. Права и обязанности участников"]
}
```

## Implemented first-version metrics

- `normalized_edit_distance`
- `keyword_retention_rate`
- `structure_overreach_rate`
- `russian_anomalies`
  - mixed-script token count
  - spaced hyphen count
  - compact suspicious hyphen count
- `page_type_misclassification`

## Command

```bash
python3 /Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent/scripts/evaluate_golden_samples.py \
  --samples-dir /Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent/golden_samples/samples
```
