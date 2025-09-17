"""
Direct TEX Code Generation Prompts (direct_tex_generator.py)
"""

# Generate LaTeX Beamer code directly from structured raw JSON content
DIRECT_TEX_GENERATION_PROMPT = r"""
You are a top-tier academic presentation design expert and LaTeX Beamer master. Your task is to analyze structured JSON content extracted from a PDF and directly transform it into a well-structured, content-refined, and visually appealing complete Beamer presentation. {language_prompt}.

Your goal is to create a directly compilable `.tex` file using the **{theme}** theme.

## Input Data Structure Analysis

You will receive a JSON object containing the following key information:
- `document_info`: Contains paper metadata such as title, authors, etc.
- `pages_content`: A list where each element represents a page, containing the plain text of that page (`text.plain`).
- `images`: A list containing information about all images extracted from the paper, such as path (`path`) and possible caption (`caption`).

## Core Instructions

1.  **Analysis & Planning**:
    -   Extract title, authors, and other information from `document_info` for creating the title page.
    -   Read through all page texts in `pages_content` to understand the overall structure and core content (background, methods, results, conclusions) of the paper.
    -   **Self-determine** the logical flow and content outline of the presentation. You are now playing both Planner and Generator roles.

2.  **Content Refinement & Organization**:
    -   Refine lengthy paragraphs into concise bullet points.
    -   Organize related content into appropriate slides (frames). Each slide should focus on one core idea.

3.  **Image-Text Matching**:
    -   When analyzing text, if content is clearly related to an image in the `images` list, you should include that image in the slide.
    -   **Strictly use** the `path` field provided in the `images` list to set the `\includegraphics` path, do not modify or simplify the path.
    -   If the image information contains a `caption`, use it as the image title.

4.  **Code Generation**:
    -   Generate **complete, independent, directly compilable** LaTeX Beamer code.
    -   Code must include complete document header (`\documentclass{{beamer}}`, etc.), necessary packages (especially `graphicx` and UTF-8 handling `ctex` or `inputenc`), `\titlepage`, multiple `frame`s, and `\end{{document}}`.
    -   **Mandatory requirement**: All `\includegraphics` commands **must** include `width=0.8\textwidth, height=0.6\textheight, keepaspectratio` parameters to ensure appropriate image sizing.
    -   **Image Path Requirements**:
        * Use figure environment and \\includegraphics command to insert images
        * **Strictly use** the complete `path` field provided in the `images` list as the image path, do not make any modifications
        * For example: If an image's path in JSON is `"output/images/1234567/_page_1_Figure_0.jpeg"`, then in TEX you must use `\includegraphics[width=0.8\textwidth, height=0.6\textheight, keepaspectratio]{{output/images/1234567/_page_1_Figure_0.jpeg}}`
        * **Absolutely do not** simplify paths to `images/_page_1_Figure_0.jpeg` or other forms

## Paper Structured Content (JSON):
```json
{raw_json}
```

Please start working now and directly output complete LaTeX Beamer code without any additional explanations or Markdown formatting.
"""
