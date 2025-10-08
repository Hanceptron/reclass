"""LLM summarization using OpenRouter with cleaned + raw transcript context."""
import json
from collections import OrderedDict
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import (
    build_frontmatter,
    chunk_text,
    clean_transcript_text,
    extract_date_fragment,
)

NARRATIVE_CHUNK_PROMPT = '''You are Emirhan's private tutor rewriting lecture chunk {chunk_index}/{total_chunks}.

Use the CLEANED chunk for wording, but cross-check with the RAW chunk to preserve every fact.

Cleaned transcript chunk:
"""{cleaned_chunk}"""

Raw transcript chunk (reference only):
"""{raw_chunk}"""

Previous section headings: {prior_topics}
{context_instruction}

Rules:
- Preserve chronological order within this chunk.
- Use `## [HH:MM:SS] Topic` style headings when timestamps or cues are present; otherwise craft descriptive headings.
- Highlight definitions/formulas with `> [!important]` callouts.
- Keep examples, problem statements, and Q&A intact.
- If audio is unclear, write `[unclear audio]` rather than guessing.
- Do not repeat material already covered in earlier chunks.

Return only Markdown for this chunk.'''

GUIDE_CHUNK_PROMPT = '''You are Emirhan's nerdy study buddy. Build actionable notes from chunk {chunk_index}/{total_chunks}.

Inputs:
- Structured narrative chunk:
"""{structured_chunk}"""
- Cleaned transcript chunk:
"""{cleaned_chunk}"""
- Raw transcript chunk (reference):
"""{raw_chunk}"""

Respond with JSON (no markdown fencing, no extra text):
{{
  "mission_control": [],
  "key_concepts": [],
  "assignments": [],
  "study_theory": [],
  "study_practice": [],
  "study_admin": [],
  "exam_intel": [],
  "risk_followups": [],
  "next_moves": []
}}

Rules:
- Only include facts explicitly stated.
- Use `term — explanation (timestamp)` for key concepts when timestamps exist.
- Use `- [ ] ...` for assignments/study items, and `- ...` for next moves.
- Return empty arrays when a category has no content.
'''

PROFESSOR_CHUNK_PROMPT = '''You are Emirhan's friendly professor delivering a one-on-one recap for chunk {chunk_index}/{total_chunks}.

Use the structured notes plus the cleaned/raw transcript for accuracy.

Structured chunk:
"""{structured_chunk}"""

Cleaned transcript chunk:
"""{cleaned_chunk}"""

Raw transcript chunk (reference):
"""{raw_chunk}"""

Previously covered topics: {prior_topics}

Explain the material conversationally, highlight intuition and transitions, and end with a reflective question or quick self-check when appropriate. Output Markdown with a heading `## Segment {chunk_index}`.'''


class LLMSummarizer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key
        )
        self.model = config.get('summarization.model', 'anthropic/claude-3.5-haiku')
        self.max_tokens = config.get('summarization.max_tokens', 4000)
        self.temperature = config.get('summarization.temperature', 0.1)
        self.chunk_chars = int(config.get('summarization.chunk_chars', 8000))
        self.chunk_overlap = int(config.get('summarization.chunk_overlap_paragraphs', 2))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def summarize(self, raw_transcript_text, course_folder, base_name, metadata=None, cleaned_transcript_text=None):
        """Generate structured notes, action guide, and professor recap."""
        print(f"\n🤖 Generating summary with {self.model}...")

        raw_transcript_text = raw_transcript_text or ""
        cleaned_transcript_text = cleaned_transcript_text or clean_transcript_text(raw_transcript_text)

        structured_chunks, raw_chunks, cleaned_chunks = self._generate_structured_chunks(
            raw_transcript_text,
            cleaned_transcript_text
        )
        structured_body = "\n\n".join(chunk for chunk in structured_chunks if chunk).strip()

        guide_body = self._generate_guide(structured_chunks, cleaned_chunks, raw_chunks)
        professor_body = self._generate_professor(structured_chunks, cleaned_chunks, raw_chunks)

        course_name = metadata.get('course', 'Unknown') if metadata else 'Unknown'
        duration = metadata.get('duration', 0) if metadata else 0
        frontmatter = build_frontmatter(base_name, course_name, duration)
        date_str = extract_date_fragment(base_name)

        structured_path = Path(course_folder) / f"{date_str}-structured.md"
        guide_path = Path(course_folder) / f"Guide {date_str}.md"
        professor_path = Path(course_folder) / f"Professor {date_str}.md"

        structured_path.write_text(self._compose_document(frontmatter, structured_body))
        guide_path.write_text(self._compose_document(frontmatter, guide_body))
        professor_path.write_text(self._compose_document(frontmatter, professor_body))

        print(f"✅ Structured transcript saved: {structured_path}")
        print(f"✅ Guide saved: {guide_path}")
        print(f"✅ Professor recap saved: {professor_path}")

        return {
            'structured_path': str(structured_path),
            'guide_path': str(guide_path),
            'professor_path': str(professor_path),
            'structured': structured_body,
            'guide': guide_body,
            'professor': professor_body
        }

    def _compose_document(self, frontmatter: str, body: str) -> str:
        body = body.strip()
        if not body:
            body = "*(No content generated.)*"
        return f"{frontmatter}\n\n{body}\n"

    def _generate_structured_chunks(self, raw_transcript_text, cleaned_transcript_text):
        raw_chunks = chunk_text(raw_transcript_text, self.chunk_chars, self.chunk_overlap)
        
        # Debug: print chunk sizes
        print(f"🔍 DEBUG: Created {len(raw_chunks)} chunks from {len(raw_transcript_text)} chars")
        for i, chunk in enumerate(raw_chunks):
            print(f"   Chunk {i+1}: {len(chunk)} chars")
        # End Debug
        
        cleaned_chunks = []
        structured_chunks = []
        recent_headings = []

        total = len(raw_chunks)
        if total == 0:
            return structured_chunks, raw_chunks, cleaned_chunks

        for idx, raw_chunk in enumerate(raw_chunks, start=1):
            cleaned_chunk = clean_transcript_text(raw_chunk)
            cleaned_chunks.append(cleaned_chunk)

            prior_topics = "\n".join(recent_headings[-5:]) if recent_headings else "None yet."
            context_instruction = (
                "First chunk: start with `# Classroom Lesson Narrative` and a `> [!note] Context` callout summarizing the lecture goals before other sections."
                if idx == 1 else
                "Continue with new `##` sections only; do NOT repeat the top-level header or context callout."
            )
            prompt = NARRATIVE_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=total,
                prior_topics=prior_topics,
                context_instruction=context_instruction,
                cleaned_chunk=cleaned_chunk,
                raw_chunk=raw_chunk
            )
            response = self._generate_text(prompt)
            structured_chunk = response.strip()
            structured_chunks.append(structured_chunk)
            recent_headings.extend(self._extract_headings(structured_chunk))

        return structured_chunks, raw_chunks, cleaned_chunks

    def _generate_guide(self, structured_chunks, cleaned_chunks, raw_chunks):
        aggregate = OrderedDict(
            mission_control=[],
            key_concepts=[],
            assignments=[],
            study_theory=[],
            study_practice=[],
            study_admin=[],
            exam_intel=[],
            risk_followups=[],
            next_moves=[],
        )

        total = len(structured_chunks)
        for idx, (structured_chunk, cleaned_chunk, raw_chunk) in enumerate(
            zip(structured_chunks, cleaned_chunks, raw_chunks), start=1
        ):
            prompt = GUIDE_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=total,
                structured_chunk=structured_chunk,
                cleaned_chunk=cleaned_chunk,
                raw_chunk=raw_chunk
            )
            response = self._generate_text(prompt)
            chunk_data = self._parse_guide_json(response)
            for key in aggregate:
                self._extend_unique(aggregate[key], chunk_data.get(key, []))

        if not aggregate['assignments']:
            aggregate['assignments'].append('- [ ] Confirm: no assignments announced this session.')
        if not aggregate['mission_control']:
            aggregate['mission_control'].append('No additional summary beyond structured notes.')

        aggregate['next_moves'] = aggregate['next_moves'][:3]

        lines = []
        lines.append('## Mission Control')
        lines.append("\n".join(aggregate['mission_control']).strip())

        lines.append('\n## Key Concepts & Definitions')
        lines.extend(self._ensure_bullets(aggregate['key_concepts']))

        lines.append('\n## Assignments, Projects, Exams')
        lines.extend(self._ensure_checkboxes(aggregate['assignments']))

        lines.append('\n## Study & Revision Checklist')
        lines.append('### Theory')
        lines.extend(self._ensure_checkboxes(aggregate['study_theory']))
        lines.append('\n### Practice')
        lines.extend(self._ensure_checkboxes(aggregate['study_practice']))
        lines.append('\n### Admin')
        lines.extend(self._ensure_checkboxes(aggregate['study_admin']))

        lines.append('\n## Exam Intel')
        lines.extend(self._ensure_bullets(aggregate['exam_intel']))

        lines.append('\n## Risk & Follow-ups')
        lines.extend(self._ensure_bullets(aggregate['risk_followups']))

        lines.append('\n## Next Moves')
        lines.extend(self._ensure_bullets(aggregate['next_moves']))

        return "\n".join(line for line in lines if line is not None)

    def _generate_professor(self, structured_chunks, cleaned_chunks, raw_chunks):
        outputs = []
        recent_headings = []
        total = len(structured_chunks)

        for idx, (structured_chunk, cleaned_chunk, raw_chunk) in enumerate(
            zip(structured_chunks, cleaned_chunks, raw_chunks), start=1
        ):
            prior_topics = "; ".join(recent_headings[-5:]) if recent_headings else "None yet."
            prompt = PROFESSOR_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=total,
                prior_topics=prior_topics,
                structured_chunk=structured_chunk,
                cleaned_chunk=cleaned_chunk,
                raw_chunk=raw_chunk
            )
            response = self._generate_text(prompt)
            recap = response.strip()
            outputs.append(recap)
            recent_headings.extend(self._extract_headings(recap))

        return "\n\n".join(output for output in outputs if output)

    def _generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def _extract_headings(self, markdown: str):
        headings = []
        if not markdown:
            return headings
        for line in markdown.splitlines():
            clean = line.strip()
            if clean.startswith('##'):
                headings.append(clean.lstrip('#').strip())
        return headings

    def _parse_guide_json(self, response: str):
        if not response:
            return {}
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                start = response.index('{')
                end = response.rindex('}') + 1
                return json.loads(response[start:end])
            except (ValueError, json.JSONDecodeError):
                return {}

    def _extend_unique(self, target_list, items):
        if not items:
            return
        for item in items:
            cleaned = item.strip()
            if cleaned and cleaned not in target_list:
                target_list.append(cleaned)

    def _ensure_bullets(self, items):
        if not items:
            return ['- None recorded.']
        bullets = []
        for item in items:
            line = item.strip()
            if not line:
                continue
            if not line.startswith('-') and not line.startswith('>'):
                line = f"- {line}"
            bullets.append(line)
        return bullets or ['- None recorded.']

    def _ensure_checkboxes(self, items):
        if not items:
            return ['- [ ] None recorded.']
        boxes = []
        for item in items:
            line = item.strip()
            if not line:
                continue
            if not line.startswith('- ['):
                line = f"- [ ] {line.lstrip('-').strip()}"
            boxes.append(line)
        return boxes or ['- [ ] None recorded.']
