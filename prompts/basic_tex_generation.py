"""
Basic TEX Code Generation Prompts (basic_tex_generator.py)
"""

# Generate LaTeX Beamer code directly from plain text (Basic LLM baseline)
BASIC_TEX_GENERATION_PROMPT = r"""
You are a professional LaTeX Beamer expert, skilled in creating high-quality presentations directly from plain text content of academic papers. {language_prompt}.

Your task is to analyze the provided paper plain text content and directly generate a complete, compilable LaTeX Beamer presentation.

## Input Content
The following is plain text content extracted from an academic paper PDF:

```
{text_content}
```

## Core Requirements

1. **Completeness**: Generate a complete LaTeX Beamer document including:
   - Document class declaration and theme settings
   - Necessary package imports
   - Title page
   - Multiple content slides
   - Document end marker

2. **Content Organization**: 
   - Infer the structure and key points of the paper from plain text
   - Create a logically clear presentation flow
   - Convert long paragraphs into concise bullet points
   - Identify and highlight key contributions and findings

3. **No Image Processing**:
   - **Strictly prohibit** using any image-related commands (such as `\includegraphics`)
   - Generate text-only slides
   - If figures are mentioned in the text, replace with textual descriptions

4. **Technical Specifications**:
   - Use `\documentclass{{beamer}}` and `\usetheme{{{theme}}}` theme
   - Include necessary packages: `graphicx`, `hyperref`, `amsmath`, etc.
   - Use `\begin{{frame}}` and `\end{{frame}}` to create slides
   - Use `itemize` and `enumerate` to organize bullet points

5. **Content Inference**:
   - Identify title, authors, abstract from the text
   - Infer research background, methods, results, conclusions sections
   - Create appropriate number of slides (typically 8-15 slides)
   - Ensure each slide has clear title and focused content

## Output Requirements

Directly output complete LaTeX code without any explanations or markdown formatting. The code must:
- Be directly compilable
- Have clear structure and logical flow
- Be concise with highlighted key points
- Be completely based on the provided text content

Please start generating LaTeX Beamer code now:
"""
