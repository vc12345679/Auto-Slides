"""
Reference Content Integration Prompts for Reference Agent
Prompts for intelligently integrating multiple literature sources into coherent expanded content
"""

# System prompt for content integration
CONTENT_INTEGRATION_SYSTEM_PROMPT = """
You are a professional academic literature synthesis expert, skilled at integrating content from multiple related papers into coherent and valuable expanded materials.

Your tasks are:
1. Analyze content related to target concepts from multiple papers
2. Extract the most important and relevant information points
3. Integrate this information into coherent, logical expanded content
4. Ensure content accuracy and objectivity, avoiding repetition and redundancy
5. Maintain academic writing rigor and readability

Integration principles:
- Prioritize the most relevant and reliable information
- Maintain clear logical structure
- Avoid simple listing; provide synthetic analysis
- Include source attribution (e.g., "research shows", "according to literature")
- Maintain objective and neutral tone
- ALWAYS write in English for international academic standards
"""


def create_content_integration_user_prompt(original_context: str,
                                         target_concept: str,
                                         literature_text: str,
                                         max_length: int) -> str:
    """
    Create user prompt for content integration
    
    Args:
        original_context: Original context where the concept appears
        target_concept: The concept to expand upon
        literature_text: Formatted literature information
        max_length: Maximum content length
        
    Returns:
        Formatted user prompt string
    """
    return f"""
Please help me integrate the following literature content to generate expanded material about "{target_concept}".

Original context:
{original_context[:300]}...

Target concept: {target_concept}

Related literature:
{literature_text}

Requirements:
1. Generate expanded content about "{target_concept}" ({max_length//2}-{max_length} characters)
2. Content should integrate viewpoints from multiple papers into coherent narrative
3. Highlight important findings relevant to the original context
4. Maintain academic rigor, avoid over-interpretation
5. Include 3-5 key points summary
6. Write entirely in English for international academic standards

Output format:
# Expanded Content
[Integrated expanded content in English]

# Key Points
1. [Point 1]
2. [Point 2]
3. [Point 3]
...

# Content Summary
[One sentence summary of the integrated content's value]
"""


# Template for simple rule-based integration (fallback)
SIMPLE_INTEGRATION_TEMPLATE = """
Based on relevant literature research, the following findings about {target_concept} emerge:

{key_points}

These studies provide important theoretical foundations and empirical support for understanding {target_concept}.
"""


