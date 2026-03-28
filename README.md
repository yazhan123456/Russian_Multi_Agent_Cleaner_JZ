# Russian Data Cleaning Agent Demo

This folder is a lightweight interview/demo package extracted from the main project:

- project repo: `/Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent`
- purpose: show the problem, the pipeline, and a small before/after example without opening the full workspace first

## Start Here

1. Read [docs/PROJECT_README.md](/Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent_Demo/docs/PROJECT_README.md)
2. Read [docs/INTERVIEW_DEMO.md](/Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent_Demo/docs/INTERVIEW_DEMO.md)
3. Open the sample input PDF in [sample_input](/Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent_Demo/sample_input)
4. Compare it with the sanitized page and JSON in [sample_output](/Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent_Demo/sample_output)

## What's Included

- `docs/PROJECT_README.md`
  - high-level project introduction and architecture
- `docs/INTERVIEW_DEMO.md`
  - short demo script and interview talking points
- `sample_input/penitentiary_smoke_p118_121.pdf`
  - small Russian PDF sample
- `sample_output/penitentiary_smoke_p118_121.layout_ocr.json`
  - Paddle layout output with keep/mask regions
- `sample_output/page_0001_sanitized.png`
  - one sanitized page after layout masking
- `sample_output/Жизнь_и_смерть_...txt`
  - a real cleaned output example from a larger book
- `sample_output/Международное_право_...txt`
  - another cleaned output example from a legal text

## Recommended Demo Order

1. Show the sample PDF.
2. Show the sanitized page image and explain that `title/body` are kept while `note/picture/table` are masked.
3. Show the cleaned TXT output.
4. Then explain the pipeline and the engineering decisions.

## Main Story

This project is not just OCR.

It is a recoverable, page-state-driven Russian document cleaning pipeline for long academic, legal, and historical PDFs. The key value is:

- layout filtering before OCR/extraction
- page routing to avoid unnecessary heavy processing
- deterministic post-cleaning for recurring OCR errors
- checkpoint/resume for long-running book jobs

## If You Need The Full Repo

Open:

- `/Users/zhanjuanyi/Desktop/Russian_Data_Cleaning_Agent`
