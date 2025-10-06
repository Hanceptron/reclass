"""LLM summarization using OpenRouter"""
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import prefilter_transcript

NARRATIVE_PROMPT = """You are Emirhan's classroom scribe. Rewrite the transcript into a polished lesson narrative without inventing facts.

Output requirements:
- Title the document `# Classroom Narrative`.
- Start with a short situational overview (2-3 sentences) in a `> [!note]` callout.
- Organize the remaining content chronologically with `##` headings containing timestamps (e.g., `## [HH:MM:SS] Topic`).
- Highlight definitions or formulas with `> [!important]` callouts, and include short bullet lists where the instructor enumerated items.
- Preserve technical language, examples, and problem statements; paraphrase only for clarity.
- Finish with a `## Instructor Side Comments` section capturing offhand remarks, logistics, or humor if present. If none, write `- None mentioned.`

Transcript:
{transcript}
"""

COMPANION_PROMPT = """You are Emirhan's battle companion for this course. Using the classroom narrative and transcript, extract actionable guidance.

Produce Obsidian markdown with these sections (use them exactly):

## Mission Control
- One paragraph summarizing what this class session was really about.

## Key Concepts & Definitions
- Bullet list. Each bullet: **Term** — concise definition. Include timestamps when possible.

## Assignments, Projects, Exams
- Use task list `- [ ]` items. Specify deliverables, due dates, submission platforms, partner/individual requirements, grading weight, and any rubrics or hints mentioned. If none were stated, add `- [ ] Confirm: no assignments announced.`

## Study & Revision Checklist
- Grouped checklists for theory, practice, and admin (three subheadings). Under each, add `- [ ]` items that tell Emirhan exactly what to review, code, or submit. Reference lecture material, textbook sections, or practice problems when mentioned. Add estimated time if implied.

## Risk & Follow-ups
- List potential pitfalls, questions to clarify next class, or campus admin tasks (like office hours, lab sign-ups). Mark anything urgent with `> [!warning]` callouts.

## Next Moves
- 3 bullet points starting with action verbs (e.g., “Schedule…”, “Email…”, “Prototype…”).

Inputs you may reference:
- Classroom Narrative:
{narrative}
- Filtered Transcript:
{transcript}
- Raw Transcript (use for verification only):
{raw_transcript}
"""

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

        narrative = self._generate_text(
            prompt=NARRATIVE_PROMPT.format(transcript=filtered_transcript)
        )

        companion = self._generate_text(
            prompt=COMPANION_PROMPT.format(
                transcript=filtered_transcript,
                narrative=narrative,
                raw_transcript=transcript_text
            )
        )

        frontmatter = self._create_frontmatter(base_name, metadata)
        date_str = self._extract_date(base_name)

        narrative_path = Path(course_folder) / f"Classroom output {date_str}.md"
        companion_path = Path(course_folder) / f"Guide {date_str}.md"

        narrative_path.write_text(f"{frontmatter}\n\n{narrative}")
        companion_path.write_text(f"{frontmatter}\n\n{companion}")

        print(f"✅ Narrative saved: {narrative_path}")
        print(f"✅ Guide saved: {companion_path}")

        return {
            'narrative_path': str(narrative_path),
            'companion_path': str(companion_path),
            'narrative': narrative,
            'companion': companion
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
        tag_slug = course.lower().replace(' ', '-')

        return f"""---
date: {date_str}
course: {course}
duration: {duration_str}
tags: [lecture, {tag_slug}]
---"""

    def _extract_date(self, base_name):
        parts = base_name.split('-')[0:3]
        return '-'.join(parts) if len(parts) == 3 else base_name

    def _generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content
