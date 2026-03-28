# Demo Walkthrough

This document explains the demo package as a project overview, not as a speaking script.

## What This Demo Shows

This repository is a lightweight presentation package for a Russian document cleaning system.

The demo focuses on one core idea:

- noisy document pages are filtered before text extraction
- non-body regions are suppressed
- the downstream text is cleaner and more usable for research workflows

## Problem

Russian academic, legal, and historical PDFs are often not usable after plain OCR.

Common failure modes include:

- running headers and page numbers leaking into body text
- figure captions and notes polluting paragraphs
- picture and table regions being treated as body text
- A3 double-page scans confusing layout analysis
- OCR introducing hyphenation breaks and glued words

This project addresses document cleaning as a long-document systems problem, not only as an OCR problem.

## Pipeline Summary

The full system is organized as a staged pipeline:

1. optional document preprocessing
2. Paddle-based layout sanitization
3. OCR / text-extraction routing
4. rule cleaning
5. model cleaning
6. review
7. repair
8. structure restore
9. export + post-clean

Key design choices:

- keep `title/body`
- mask `note/picture/table`
- prefer `extract` when the PDF text layer is good
- send only riskier pages through heavier processing
- support checkpoint/resume for long book jobs

## Suggested Viewing Order

1. Open the original input page  
   [sample_input/page_0001_original.png](../sample_input/page_0001_original.png)

2. Open the sanitized page  
   [sample_output/page_0001_sanitized.png](../sample_output/page_0001_sanitized.png)

3. Open the side-by-side comparison  
   [sample_output/page_compare.png](../sample_output/page_compare.png)

4. Inspect the layout output  
   [sample_output/penitentiary_smoke_p118_121.layout_ocr.json](../sample_output/penitentiary_smoke_p118_121.layout_ocr.json)

5. Inspect the cleaned text outputs  
   - [Жизнь_и_смерть_...txt](../sample_output/Жизнь_и_смерть_в_России_скои__империи__Новые_открытия_в_области_археологии_и_истории_России_XVIII_XIX_вв____Life_and_Dea.txt)
   - [Международное_право_...txt](../sample_output/Международное_право_и_правовая_система_Российской_Федерации.txt)

## Why The Sample Page Matters

The selected sample page contains:

- a section heading
- multiple image regions
- multiple caption blocks
- two-column body text

This makes the before/after difference easy to see:

- image-heavy regions are suppressed
- non-body regions no longer dominate the page
- body text remains available for downstream extraction and cleaning

## Engineering Value

The main interest of the project is not prompt engineering alone.

The engineering value comes from:

- staged responsibility separation
- page-level state tracking
- recoverable long-running jobs
- layout masking before OCR/extraction
- deterministic post-cleaning for recurring OCR artifacts

## Limitations

This is a practical research-text cleaning system, not a perfect publishing-grade engine.

Known limitations include:

- some caption leakage may remain on difficult pages
- glued words and OCR artifacts still need post-clean rules
- quality varies across document types
- the demo package is intentionally smaller than the full codebase
