"""LLM summarization using OpenRouter"""
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import build_frontmatter, extract_date_fragment

NARRATIVE_PROMPT = """You are Emirhan's private tutor. Rewrite the raw lecture into a polished lesson, keeping every factual detail.

Requirements:
- Title the document `# Classroom Lesson Narrative`.
- Start with a `> [!note] Context` callout summarizing the class setting and goals in 2-3 sentences.
- Present the session chronologically using sections like `## [HH:MM:SS] Topic`. If the topic is unclear, craft a concise descriptive title.
- Within each section, paraphrase the instructor clearly without omitting examples, equations, or problem statements. Use bullet lists when the speaker enumerates items.
- Use `> [!important]` callouts for definitions, formulas, or rules exactly as stated.
- Preserve student questions, instructor answers, and anecdotes (rewrite only for grammar and flow).
- Conclude with `## Logistics & Side Remarks` capturing reminders, humor, or admin notes. If none exist, write `- None mentioned.`

Transcript:
{transcript}
"""

COMPANION_PROMPT = """You are Emirhan's nerdy study buddy. Using the transcript and tutor narrative, build a battle plan for exams and projects.

Output format (Obsidian markdown):

## Mission Control
Friendly paragraph explaining what the class focused on and why it matters for Emirhan.

## Key Concepts & Definitions
- Bullet list. Format each bullet as `- **Term** — explanation (timestamp)`. Include precise timestamps whenever available. No filler.

## Assignments, Projects, Exams
- Task list using `- [ ]`. For each task mention deliverable, due date/time, submission platform, partner rules, grading weight, and any hints the instructor gave. If nothing was announced, add `- [ ] Confirm: no assignments announced this session.`

## Study & Revision Checklist
### Theory
- [ ] Items that describe what concepts/sections to review, with timestamps or resource pointers.
### Practice
- [ ] Coding problems, worksheets, or exercises to work through. Mention estimated effort if implied.
### Admin
- [ ] Logistics (sign-ups, emails, materials to download, office hours to attend).

## Exam Intel
- Bulleted list calling out likely exam or quiz question angles. Explain why each topic is risky based on instructor emphasis. Use `> [!warning]` callouts for high-stakes items.

## Risk & Follow-ups
- Bullet list of open questions, unclear instructions, or things to confirm with the professor/TA. Include relevant deadlines or office hours if mentioned.

## Next Moves
- Exactly three bullet points starting with action verbs (e.g., "Schedule…", "Email…", "Prototype…") that Emirhan should complete within the next 48 hours.

Inputs for context:
- Tutor Narrative:
{narrative}
- Filtered Transcript:
{transcript}
- Raw Transcript (for verification only):
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

        narrative = self._generate_text(
            prompt=NARRATIVE_PROMPT.format(transcript=transcript_text)
        )

        companion = self._generate_text(
            prompt=COMPANION_PROMPT.format(
                transcript=transcript_text,
                narrative=narrative,
                raw_transcript=transcript_text
            )
        )

        course_name = metadata.get('course', 'Unknown') if metadata else 'Unknown'
        duration = metadata.get('duration', 0) if metadata else 0
        frontmatter = build_frontmatter(base_name, course_name, duration)
        date_str = extract_date_fragment(base_name)

        structured_path = Path(course_folder) / f"{date_str}-structured.md"
        companion_path = Path(course_folder) / f"Guide {date_str}.md"

        structured_path.write_text(f"{frontmatter}\n\n{narrative}")
        companion_path.write_text(f"{frontmatter}\n\n{companion}")

        print(f"✅ Structured transcript saved: {structured_path}")
        print(f"✅ Guide saved: {companion_path}")

        return {
            'structured_path': str(structured_path),
            'companion_path': str(companion_path),
            'structured': narrative,
            'companion': companion
        }

    def _generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content
