"""
Prompts Module - Unified Import Entry Point

This module imports prompts from individual prompt files,
maintaining compatibility with the original prompts.py
"""

# Import all prompts from prompt files
from .direct_tex_generation import DIRECT_TEX_GENERATION_PROMPT
from .key_content_extraction import KEY_CONTENT_EXTRACTION_PROMPT
from .slides_planning import SLIDES_PLANNING_PROMPT
from .interactive_refinement import INTERACTIVE_REFINEMENT_SYSTEM_MESSAGE
from .tex_generation import TEX_GENERATION_PROMPT
from .tex_revision import TEX_REVISION_SYSTEM_MESSAGE, TEX_REVISION_HUMAN_MESSAGE
from .basic_tex_generation import BASIC_TEX_GENERATION_PROMPT
from .tex_error_fix import TEX_ERROR_FIX_PROMPT

# New specialized prompts
from .extract_tables_and_equations import EXTRACT_TABLES_AND_EQUATIONS_PROMPT
from .summarize_text_for_presentation import SUMMARIZE_TEXT_FOR_PRESENTATION_PROMPT

# For backward compatibility, retain the original PRESENTATION_CONTENT_ENHANCEMENT_PROMPT
# Actually we will use two new prompts to replace it
PRESENTATION_CONTENT_ENHANCEMENT_PROMPT = SUMMARIZE_TEXT_FOR_PRESENTATION_PROMPT

# Export all prompts
__all__ = [
    'DIRECT_TEX_GENERATION_PROMPT',
    'KEY_CONTENT_EXTRACTION_PROMPT', 
    'SLIDES_PLANNING_PROMPT',
    'INTERACTIVE_REFINEMENT_SYSTEM_MESSAGE',
    'TEX_GENERATION_PROMPT',
    'TEX_REVISION_SYSTEM_MESSAGE',
    'TEX_REVISION_HUMAN_MESSAGE',
    'BASIC_TEX_GENERATION_PROMPT',
    'TEX_ERROR_FIX_PROMPT',
    'PRESENTATION_CONTENT_ENHANCEMENT_PROMPT',  # Backward compatibility
    'EXTRACT_TABLES_AND_EQUATIONS_PROMPT',     # New
    'SUMMARIZE_TEXT_FOR_PRESENTATION_PROMPT',  # New
]
