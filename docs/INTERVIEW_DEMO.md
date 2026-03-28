# Interview Demo Guide

This guide is for presenting the project as an internship portfolio piece.

## What To Show

Do not try to explain every module first.

Start with:

1. a messy Russian PDF page
2. one command that runs the pipeline
3. the cleaned TXT output

Then explain the architecture.

## Recommended Demo Assets

Prepare one small sample book or a short extracted sample with:

- one clean body page
- one page with figure captions or notes
- one page with difficult formatting

Good demo artifacts:

- original PDF page
- `layout_sanitize/*.sanitized_pages/page_XXXX.png`
- `ocr.json`
- final TXT snippet

## 30-Second Pitch

Use this if the interviewer asks “what is this project?”

> I built a Russian document cleaning pipeline for long academic and legal PDFs. It uses layout masking, OCR/extract routing, LLM-based cleaning and repair, and export-time post-processing to turn noisy PDFs into cleaner research text. The system is stateful, supports checkpoint/resume, and is designed for long-running book-scale jobs rather than one-off OCR.

## 2-Minute Demo Script

1. Show a noisy input page.
   - point out figure captions, page headers, notes, or broken line wraps
2. Show the command:

```bash
python3 scripts/process_books.py \
  --profile balanced_cost \
  --book '5/YourBook.pdf' \
  --run-root outputs/full_book_runs/demo_run \
  --final-txt-dir outputs/final_txt \
  --resume \
  --prevent-sleep
```

3. Explain the stages:
   - Paddle masks note/picture/table regions
   - OCR/extract picks the cheapest usable text path
   - rules + DeepSeek clean and repair text
   - export-time cleanup fixes figure captions and hyphenation
4. Show the final TXT output.
5. Point out one concrete bug you fixed during development:
   - mid-book bibliography truncation
   - page header leakage into `extract`
   - A3 split before layout sanitize

## 5-Minute Technical Walkthrough

If they want details, cover these:

### 1. Problem framing

- Russian research PDFs are not solved by simple OCR.
- The challenge is not just recognition but filtering and cleanup.

### 2. System design

- `process_books.py` orchestrates the full pipeline
- `PageState` and the state machine make long jobs resumable
- Paddle does local layout masking
- OCR/extract chooses the cheapest usable path
- DeepSeek handles cleaning, repair, and structure restoration
- post-clean rules catch recurring deterministic errors

### 3. Engineering tradeoffs

- keep body recall high, even if some note noise remains
- prefer `extract` over OCR when text layer is good
- use page-level checkpoints so book jobs can resume safely
- keep review as the formal diagnosis stage

### 4. Results

Mention things like:

- full-book processing instead of page toy demos
- reduced manual cropping for pictures and notes
- automatic recovery from failures
- cleaner TXT output for downstream RAG

## Questions You Should Expect

### “Why not just use one multimodal model?”

Because the problem is not only OCR quality. It also needs:

- layout filtering
- page routing
- deterministic cleanup
- checkpoint/resume
- cost control on long books

### “Why use multiple stages instead of one model?”

Because diagnosis, repair, and structure restoration have different failure modes. Separating them makes debugging and recovery much easier.

### “What was the hardest engineering problem?”

Good answers:

- keeping state consistent across long runs
- preventing page headers/figure captions from leaking back through extraction
- balancing body recall against aggressive cleanup

## Resume Bullets

### Chinese

- 构建面向俄语学术/法律 PDF 的长文档清洗系统，支持布局分流、OCR/extract 路由、页级状态恢复与断点续跑。
- 集成 Paddle 版面检测、Qwen OCR、DeepSeek 清洗/修复/结构恢复，降低图注、页眉、尾注对正文抽取的污染。
- 设计导出后规则收尾模块，修复断词、混合同形字、图注残留和 backmatter 截断等长文档清洗问题。

### English

- Built a long-document Russian text cleaning pipeline with layout masking, OCR/extract routing, page-level checkpointing, and resume support.
- Integrated Paddle layout detection, Qwen OCR, and DeepSeek cleaning/repair/structure stages to reduce figure-caption, header, and note pollution in extracted text.
- Implemented export-time rule cleanup for hyphenation, mixed-script OCR artifacts, caption leakage, and backmatter truncation issues.

## What Not To Do In An Interview

- do not start by listing every file in the repo
- do not describe it as “just an OCR bot”
- do not overclaim perfect accuracy
- do not focus only on prompt engineering

Instead, show:

- the document problem
- the pipeline design
- the engineering constraints
- the concrete before/after result

