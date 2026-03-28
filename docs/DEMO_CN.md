# 中文项目说明

## 项目概览

这是一个面向俄语学术、法律、历史 PDF 的文档清洗系统。

它的目标不是只做一次 OCR，而是把包含页眉页码、图注、尾注、图片、表格、A3 双页扫描、断词和词语粘连等问题的长文档，转换成更适合研究、检索和 RAG 使用的纯文本。

## 这个项目解决的问题

普通 OCR 在俄语长文档上经常会遇到这些问题：

- 页眉页码混进正文
- 图注、尾注、参考文献污染正文
- 图片和表格区域被当成正文
- A3 双页扫描导致版面分析混乱
- OCR 带来断词、连字符断裂和词语粘连
- 长文档处理中断后难以恢复

这个系统的重点不是“识别文字本身”，而是对整本书进行可恢复、可追踪、可批量运行的清洗。

## 处理流程

当前主链是：

1. 文档预处理
2. Paddle 版面分流与局部遮罩
3. OCR / extract 路由
4. Rule cleaning
5. Primary cleaning
6. Review
7. Repair
8. Structure restore
9. Export + post-clean

核心思路是：

- `title/body` 保留
- `note/picture/table` 遮罩
- 文本层好的页优先 `extract`
- 风险页再走 OCR
- 后续再做清洗、修复和结构恢复

## 这个 Demo 查看顺序：


1. 原始页面  
   [sample_input/page_0001_original.png](../sample_input/page_0001_original.png)

2. 遮罩后的页面  
   [sample_output/page_0001_sanitized.png](../sample_output/page_0001_sanitized.png)

3. 并排对比图  
   [sample_output/page_compare.png](../sample_output/page_compare.png)

4. Paddle 的版面输出  
   [sample_output/penitentiary_smoke_p118_121.layout_ocr.json](../sample_output/penitentiary_smoke_p118_121.layout_ocr.json)

5. 最终文本输出  
   - [Жизнь_и_смерть_...txt](../sample_output/Жизнь_и_смерть_в_России_скои__империи__Новые_открытия_в_области_археологии_и_истории_России_XVIII_XIX_вв____Life_and_Dea.txt)
   - [Международное_право_...txt](../sample_output/Международное_право_и_правовая_система_Российской_Федерации.txt)

## 示例页面说明

这个示例页包含：

- 分节标题
- 多个图片区域
- 多个图注块
- 两栏正文

遮罩后的页面可以直接看到：

- 图片主体被遮掉
- 非正文区域被压制
- 正文主体仍然保留

这类页面比纯文字页更能体现这套系统的价值，因为它展示的不是“能不能识别文字”，而是“能不能在复杂版面中先清理噪声，再提取正文”。

## 工程特点

这不是一个单模型自由发挥的 OCR 工具，而是一个工程化 pipeline：

- 支持长文档整本处理
- 页级 checkpoint 和 resume
- A3 双页先拆分再分析
- Paddle 先做版面遮罩
- OCR 与 extract 路由分流
- DeepSeek 负责清洗、修复和结构恢复
- 导出后规则继续处理图注、断词、粘连词和 backmatter

## 为什么不是“一个大模型一次做完”

因为这里的问题不只是 OCR 识别率，而是：

- 版面过滤
- 成本和速度控制
- 长任务恢复
- 不同阶段的职责分离
- 后处理的可解释性

对长文档而言，单模型一次完成通常不够稳定，也不利于排查问题。

## 当前局限

这仍然是一个工程中的系统，而不是出版级完美文本引擎。

当前仍可能存在：

- 少量图注残留
- 断词和词语粘连
- 个别结构页恢复不够理想
- 不同书型之间质量差异较大

但对于研究型语料清洗和 RAG 预处理，这套系统已经能显著减少人工处理量。
