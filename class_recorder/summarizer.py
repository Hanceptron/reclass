"""LLM summarization using OpenRouter"""
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import prefilter_transcript

SUMMARY_PROMPT = """Analyze this lecture transcript and create a comprehensive Obsidian-formatted summary.

## Structure Requirements:

### 1. Overview (2-3 sentences)
Brief description of the lecture's main topic and objectives.

### 2. Key Concepts
Main theories, definitions, and principles covered. Use callouts:
> [!important] Core Concept
> Brief explanation

### 3. Important Examples & Case Studies
Concrete examples discussed with brief explanations.

### 4. Key Takeaways
3-5 most critical points students should remember.

### 5. Discussion Questions (if any mentioned)
Questions posed to students or for review.

### 6. Action Items
- Assignments mentioned
- Readings assigned  
- Exam topics referenced
- Practice problems suggested

## Formatting Rules:
- Use Obsidian callouts: `> [!note]`, `> [!important]`, `> [!tip]`, `> [!warning]`
- Include timestamps in [HH:MM:SS] format for topic transitions
- Bold key terms on first mention
- Use tables for comparisons or structured data
- Create Mermaid diagrams for complex relationships (if applicable)
- Use proper Markdown headers (##, ###)

## Transcript:
{transcript}

Create the summary now:"""

class LLMSummarizer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key
        )
        self.model = config.get('summarization.model', 'google/gemini-2.5-flash')
        self.max_tokens = config.get('summarization.max_tokens', 2000)
        self.temperature = config.get('summarization.temperature', 0.3)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def summarize(self, transcript_text, course_folder, base_name, metadata=None):
        """Generate summary from transcript"""
        print(f"\n🤖 Generating summary with {self.model}...")
        
        filtered_transcript = prefilter_transcript(transcript_text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(transcript=filtered_transcript)
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        summary = response.choices[0].message.content
        
        # Add frontmatter
        frontmatter = self._create_frontmatter(base_name, metadata)
        full_markdown = f"{frontmatter}\n\n{summary}"
        
        # Save summary
        summary_path = Path(course_folder) / f"{base_name}.md"
        with open(summary_path, 'w') as f:
            f.write(full_markdown)
        
        print(f"✅ Summary saved: {summary_path}")
        
        return {
            'summary_path': str(summary_path),
            'summary': summary
        }
    
    def _create_frontmatter(self, base_name, metadata):
        """Create Obsidian frontmatter"""
        date_parts = base_name.split('-')[0:3]
        date_str = '-'.join(date_parts) if len(date_parts) == 3 else base_name
        
        course = metadata.get('course', 'Unknown') if metadata else 'Unknown'
        duration = metadata.get('duration', 0) if metadata else 0
        
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return f"""---
date: {date_str}
course: {course}
duration: {duration_str}
tags: [lecture, {course.lower()}]
---"""
