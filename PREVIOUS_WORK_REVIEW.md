# Review of Previous Work — Formatting Issues & Notes
**Files reviewed:** Part 1 (Appendix A — code), Part 2 (Assignment form), Part 3 (Main report, 32 pages)  
**Compared against:** BACHELOR_THESIS_RULES.md (Lviv Ivan Franko University guidelines)

---

## IMPORTANT: Single document this time
Previous work was split across 3 PDFs. **The new report must be one single PDF/DOCX.**  
Order: Title → Assignment → Annotation → Contents → Abbreviations → Introduction → Chapters → Conclusions → References → Appendices.

---

## 1. Title Page — GOOD, minor fixes needed

✅ Correct structure: Ministry → University → Faculty → Department → "Допустити до захисту" → Title → Author/Supervisor/Reviewer → City/Year  
✅ Title is ALL CAPS centered: "СТВОРЕННЯ СИСТЕМИ АЕРОСПОСТЕРЕЖЕННЯ З РОЗПІЗНАВАННЯМ ОБ'ЄКТІВ"  
✅ "Кваліфікаційна робота / Бакалавр" with correct formatting

⚠️ **Fix for new work:** The assignment (Part 2) has supervisor as "проф. Кушнір О. С." (professor) but title page lists the same person as "доц. Кушнір О. С." (associate professor). Verify the correct title before submitting.

---

## 2. Chapter Headings — WRONG, must fix

**Rule:** Chapter headings must be **centered, ALL CAPS, bold, no period.**

❌ Previous work uses title case:
```
1   Огляд використаних технологій
2   Основні принципи
3   Покращення YOLOv7
4   Аналіз та результати експериментів
5   Висновки
```

✅ Must be:
```
1   ОГЛЯД ВИКОРИСТАНИХ ТЕХНОЛОГІЙ
2   ОСНОВНІ ПРИНЦИПИ
3   ПОКРАЩЕННЯ YOLOV7
4   АНАЛІЗ ТА РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТІВ
5   ВИСНОВКИ
```

Same applies to structural headings: **ВСТУП, ВИСНОВКИ, СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ** — all must be ALL CAPS.

---

## 3. Subsection Headings — ONE issue

✅ Format is mostly correct: from paragraph indent, sentence case, no period  
❌ **"1.1 Мова програмування Python:"** — has a colon at the end, which is not allowed. Remove the colon.

---

## 4. Figure Captions — WRONG FORMAT, must fix everywhere

**Rule:** `Рисунок X.Y – Description` (full word, em-dash separator, placed below figure)

❌ Previous work uses: `Рис. 1.1 Вигляд головного вікна програми`  
❌ Problems: abbreviated "Рис.", missing em-dash "–"

✅ Must be: `Рисунок 1.1 – Вигляд головного вікна програми`

**Every single figure caption in the new work must follow this format.** There are ~13 figures in the previous work — all had the wrong caption format.

---

## 5. Table Captions — WRONG FORMAT, must fix everywhere

**Rule:** `Таблиця X.Y – Description` (full word, em-dash, placed **above** table, left-aligned)

❌ Previous work uses: `Табл. 1.1 Технічні характеристики T-MOTOR P1604 KV2850`  
❌ Problems: abbreviated "Табл.", missing em-dash "–"

✅ Must be: `Таблиця 1.1 – Технічні характеристики T-MOTOR P1604 KV2850`

---

## 6. Formula Numbering — mostly OK, one error

✅ Most formulas: properly centered with `(3.1)` at right margin — correct  
✅ Symbol explanations use "де" without colon — correct  
❌ **Formula (3.6):** The number `(3.6)` appeared at the start of the next line instead of at the right margin of the formula line. Always place formula number in parentheses flush right on the same line as the formula.

---

## 7. References / Bibliography — MULTIPLE ISSUES

### 7a. Section title is wrong
❌ Previous work: "Список джерел" (lowercase, mixed case)  
✅ Must be: **"СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ"** (ALL CAPS, centered — it's a structural element)

### 7b. References use IEEE format, not ДСТУ 8302:2015
❌ Previous work uses semicolons between authors and adds "– URL:" at the end:
```
[1] Hastie, T.; Rosset, S.; Zhu, J.; Zou, H. Multi-class adaboost...
```

✅ ДСТУ 8302:2015 format for conference paper:
```
1. Hastie T., Rosset S., Zhu J., Zou H. Multi-class adaboost. Stat. Its Interface. 2009. Vol. 2. P. 349–360. URL: https://...
```

Key differences:
- No square brackets `[1]` — use `1.` numbering with period
- Commas between authors, not semicolons
- No `– URL:` separator — just `URL:` 
- Year, volume, pages format per ДСТУ

### 7c. "Електронні ресурси:" subheading is unnecessary
The previous work groups all references under "Електронні ресурси:". ДСТУ doesn't require this grouping — just list all sources in one numbered list.

### 7d. In-text citation format
✅ The body text correctly uses `[1]`, `[2]`, etc. in square brackets — this is correct per the guidelines.  
⚠️ Note: In-text brackets format is `[1]` but the bibliography itself should NOT use brackets — use `1.` instead.

---

## 8. Introduction — INCOMPLETE

**Rule:** Introduction (2–3 pages) must explicitly cover 7 elements.

❌ The previous introduction (3 paragraphs) does NOT explicitly state:
- Relevance *(partially present)*
- **Goal** — missing
- **Object of research** — missing
- **Subject of research** — missing
- **Research methods** — missing
- **Elements of novelty** — missing
- **Forecasted development directions** — missing

✅ For the new work, the introduction must explicitly cover all 7 items, ideally with labelled bullet points or clearly identifiable paragraphs.

---

## 9. Missing Section: List of Abbreviations

❌ The previous work has NO "ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ, СИМВОЛІВ, СКОРОЧЕНЬ І ТЕРМІНІВ" section.

This is a required structural element that must appear after the Table of Contents and before the Introduction. The new work uses many abbreviations (БПЛА, YOLOv7 / Mask R-CNN for the new work, mAP, IoU, FPV, DSC, EIOU, CNN, etc.) — all must be listed here.

---

## 10. Table of Contents — fix section titles

The TOC must list structural elements exactly as they appear in the document. Since we're fixing:
- Chapter headings to ALL CAPS → update TOC entries accordingly
- "Список джерел" → "СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ"
- Add "ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ, СИМВОЛІВ, СКОРОЧЕНЬ І ТЕРМІНІВ" entry

---

## 11. Appendix (Code Listing) — partially correct

✅ "Додаток А" centered with first capital — correct label  
✅ Starts on a new page  
❌ **No appendix title** — the label "Додаток А" must be followed by the appendix title on the next line (centered), e.g.:
```
Додаток А
Лістинг програмного коду
```
❌ **Code is in proportional font (Times New Roman)** — code listings should use a monospace font (Courier New) for readability, though the guidelines don't strictly prohibit TNR. Consider using a monospace font inside code blocks.

❌ **No reference to the appendix in the main text** — every appendix must be cited in the text: e.g., "...наведено у Додатку А."

---

## 12. Conclusions — content issue

✅ Has bullet-point structure listing results  
❌ Uses `–` (list dash) at the start of items without paragraph indent on wrapped lines (hanging indent issue in the PDF)  
❌ Missing: explicit recommendation for further development / forecasted directions  
❌ Missing: statement of the work's scientific/practical significance

---

## Summary: Priority fixes for new report

| Priority | Issue | Rule |
|---|---|---|
| 🔴 HIGH | Chapter headings must be ALL CAPS | §5 Headings |
| 🔴 HIGH | "СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ" not "Список джерел" | §7 References |
| 🔴 HIGH | Figure captions: use "Рисунок X.Y –" not "Рис. X.Y" | §8 Figures |
| 🔴 HIGH | Table captions: use "Таблиця X.Y –" not "Табл. X.Y" | §9 Tables |
| 🔴 HIGH | Add "ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ" section | §6 Structure |
| 🔴 HIGH | Bibliography: use ДСТУ 8302:2015 format (no brackets) | §11 References |
| 🟡 MED | Introduction must cover all 7 required elements | §7 Intro |
| 🟡 MED | Add appendix title below "Додаток А" label | §13 Appendices |
| 🟡 MED | Remove colon from subsection heading "1.1 Python:" | §5 Headings |
| 🟡 MED | Formula (3.6) number must be on the same line, flush right | §10 Formulas |
| 🟠 LOW | Cite appendix in main text body | §13 Appendices |
| 🟠 LOW | Verify supervisor title (prof. vs. doц.) on title page | §1 Title page |
| 🟠 LOW | Remove "Електронні ресурси:" subheading from bibliography | §11 References |
