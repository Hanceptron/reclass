"""LLM summarization using OpenRouter"""
import json
from collections import OrderedDict
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import build_frontmatter, chunk_text, extract_date_fragment

NARRATIVE_CHUNK_PROMPT = '''You are Emirhan's private tutor capturing a lecture chunk-by-chunk.
Chunk {chunk_index}/{total_chunks}.
Previously covered headings:
{prior_topics}

{context_instruction}

Rules:
- Preserve chronological order within this chunk only.
- Prefer headings like `## [HH:MM:SS] Topic` when timestamps exist; otherwise craft a clear descriptive heading.
- Use `> [!important]` callouts for definitions, formulas, or rules that matter.
- Reproduce examples, problem statements, and student questions from this chunk.
- Use bullet lists wherever the instructor enumerated items.
- Do NOT repeat material already covered in earlier chunks.

Chunk transcript:
"""{chunk_text}"""

Return Markdown only.'''

GUIDE_CHUNK_PROMPT = '''You are Emirhan's nerdy study buddy. Extract actionable intel chunk-by-chunk.
Chunk {chunk_index}/{total_chunks}.

Respond ONLY with valid JSON using this schema (arrays of strings):
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

Keep entries concise, unique, and grounded in the chunk—no speculation.
Use `term — explanation (timestamp)` format for key concepts when possible.
Use `- [ ] ...` for assignments and study items, and `- ...` for next moves.

Structured narrative chunk:
"""{structured_chunk}"""

Raw transcript chunk for verification:
"""{raw_chunk}"""
'''

PROFESSOR_CHUNK_PROMPT = '''You are Emirhan's friendly professor giving a one-on-one recap.
Chunk {chunk_index}/{total_chunks}.
Previously addressed topics:
{prior_topics}

Explain this chunk conversationally, highlighting intuition, transitions, and why each idea matters. Reference examples or anecdotes from the transcript. End the chunk with a reflective question or quick self-check tied to the material if appropriate.

Structured narrative chunk:
"""{structured_chunk}"""

Raw transcript chunk for verification:
"""{raw_chunk}"""

Return Markdown paragraphs (include a `## Segment {chunk_index}` heading).'''


class LLMSummarizer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key
        )
        self.model = config.get('summarization.model', 'google/gemini-2.5-flash')
        self.max_tokens = config.get('summarization.max_tokens', 2000)
        self.temperature = config.get('summarization.temperature', 0.3)
        self.chunk_chars = int(config.get('summarization.chunk_chars', 4000))
        self.chunk_overlap = int(config.get('summarization.chunk_overlap_paragraphs', 1))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def summarize(self, transcript_text, course_folder, base_name, metadata=None):
        """Generate structured notes, guide, and professor recap from transcript."""
        print(f"\n🤖 Generating summary with {self.model}...")

        structured_chunks, transcript_chunks = self._generate_structured_chunks(transcript_text)
        structured_body = "\n\n".join(chunk for chunk in structured_chunks if chunk).strip()

        guide_body = self._generate_guide(structured_chunks, transcript_chunks)
        professor_body = self._generate_professor(structured_chunks, transcript_chunks)

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

    def _generate_structured_chunks(self, transcript_text):
        transcript_chunks = chunk_text(transcript_text, self.chunk_chars, self.chunk_overlap)
        structured_chunks = []
        recent_headings = []

        total = len(transcript_chunks)
        if total == 0:
            return structured_chunks, transcript_chunks

        for idx, chunk in enumerate(transcript_chunks, start=1):
            prior_topics = "\n".join(recent_headings[-5:]) if recent_headings else "None yet."
            context_instruction = (
                "Include the heading `# Classroom Lesson Narrative` followed by a `> [!note] Context` callout summarizing the lecture goals before other sections."
                if idx == 1 else
                "Continue with new `##` headings only; do NOT repeat the top-level header or the context callout."
            )
            prompt = NARRATIVE_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=total,
                prior_topics=prior_topics,
                context_instruction=context_instruction,
                chunk_text=chunk
            )
            response = self._generate_text(prompt)
            structured_chunks.append(response.strip())
            recent_headings.extend(self._extract_headings(response))

        return structured_chunks, transcript_chunks

    def _generate_guide(self, structured_chunks, transcript_chunks):
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
        for idx, (structured_chunk, raw_chunk) in enumerate(zip(structured_chunks, transcript_chunks), start=1):
            prompt = GUIDE_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=total,
                structured_chunk=structured_chunk,
                raw_chunk=raw_chunk
            )
            response = self._generate_text(prompt)
            chunk_data = self._parse_guide_json(response)
            for key in aggregate:
                items = chunk_data.get(key, []) if chunk_data else []
                self._extend_unique(aggregate[key], items)

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

    def _generate_professor(self, structured_chunks, transcript_chunks):
        outputs = []
        recent_headings = []
        total = len(structured_chunks)

        for idx, (structured_chunk, raw_chunk) in enumerate(zip(structured_chunks, transcript_chunks), start=1):
            prior_topics = "; ".join(recent_headings[-5:]) if recent_headings else "None yet."
            prompt = PROFESSOR_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=total,
                prior_topics=prior_topics,
                structured_chunk=structured_chunk,
                raw_chunk=raw_chunk
            )
            response = self._generate_text(prompt)
            outputs.append(response.strip())
            recent_headings.extend(self._extract_headings(response))

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
            if not cleaned:
                continue
            if cleaned not in target_list:
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
        return bullets if bullets else ['- None recorded.']

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
        return boxes if boxes else ['- [ ] None recorded.']
