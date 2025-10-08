"""Russian LLM summarization with translation to English."""
import json
from collections import OrderedDict
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .utils import (
    build_frontmatter,
    chunk_text,
    clean_transcript_text,
    extract_date_fragment,
)

# ==================== RUSSIAN PROMPTS ====================

NARRATIVE_CHUNK_PROMPT_RU = '''Ты личный репетитор Эмирхана, переписывающий фрагмент лекции {chunk_index}/{total_chunks}.

Используй ОЧИЩЕННЫЙ фрагмент для формулировок, но сверяйся с СЫРЫМ фрагментом, чтобы сохранить все факты.

Очищенный фрагмент транскрипции:
"""{cleaned_chunk}"""

Сырой фрагмент транскрипции (только для справки):
"""{raw_chunk}"""

Предыдущие заголовки разделов: {prior_topics}
{context_instruction}

Правила:
- Сохраняй хронологический порядок внутри этого фрагмента.
- Используй заголовки в стиле `## [HH:MM:SS] Тема`, когда есть временные метки или подсказки; иначе создавай описательные заголовки.
- Выделяй определения/формулы с помощью выносок `> [!important]`.
- Сохраняй примеры, постановки задач и вопросы-ответы без изменений.
- Если аудио нечеткое, пиши `[неразборчивое аудио]` вместо догадок.
- Не повторяй материал, уже охваченный в предыдущих фрагментах.

Верни только Markdown для этого фрагмента.'''

GUIDE_CHUNK_PROMPT_RU = '''Ты помощник по учебе Эмирхана. Создай практические заметки из фрагмента {chunk_index}/{total_chunks}.

Входные данные:
- Структурированный нарративный фрагмент:
"""{structured_chunk}"""
- Очищенный фрагмент транскрипции:
"""{cleaned_chunk}"""
- Сырой фрагмент транскрипции (справка):
"""{raw_chunk}"""

Ответь в формате JSON (без markdown ограждений, без дополнительного текста):
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

Правила:
- Включай только явно указанные факты.
- Используй формат `термин — объяснение (временная метка)` для ключевых концепций, когда есть временные метки.
- Используй `- [ ] ...` для заданий/учебных пунктов, и `- ...` для следующих шагов.
- Возвращай пустые массивы, когда в категории нет контента.
'''

PROFESSOR_CHUNK_PROMPT_RU = '''Ты дружелюбный профессор Эмирхана, дающий персональное резюме для фрагмента {chunk_index}/{total_chunks}.

Используй структурированные заметки плюс очищенную/сырую транскрипцию для точности.

Структурированный фрагмент:
"""{structured_chunk}"""

Очищенный фрагмент транскрипции:
"""{cleaned_chunk}"""

Сырой фрагмент транскрипции (справка):
"""{raw_chunk}"""

Ранее охваченные темы: {prior_topics}

Объясни материал в разговорном стиле, подчеркни интуицию и переходы, и заканчивай рефлексивным вопросом или быстрой самопроверкой, когда уместно. Выводи Markdown с заголовком `## Сегмент {chunk_index}`.'''

TRANSLATION_PROMPT = '''Переведи следующую транскрипцию лекции с русского на английский.

Сохрани:
- Все технические термины точно
- Структуру предложений естественной для английского
- Все имена, названия компаний и специфические термины

Сырая русская транскрипция:
"""{russian_text}"""

Верни только английский перевод, без дополнительных комментариев.'''

# ==================== ENGLISH PROMPTS (for translated transcript) ====================

NARRATIVE_CHUNK_PROMPT_EN = '''You are Emirhan's private tutor rewriting lecture chunk {chunk_index}/{total_chunks}.

This is a TRANSLATED transcript from Russian. Use the CLEANED chunk for wording, but cross-check with the RAW chunk to preserve every fact.

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

GUIDE_CHUNK_PROMPT_EN = '''You are Emirhan's nerdy study buddy. Build actionable notes from chunk {chunk_index}/{total_chunks}.

This is from a TRANSLATED Russian lecture transcript.

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

PROFESSOR_CHUNK_PROMPT_EN = '''You are Emirhan's friendly professor delivering a one-on-one recap for chunk {chunk_index}/{total_chunks}.

This is from a TRANSLATED Russian lecture transcript. Use the structured notes plus the cleaned/raw transcript for accuracy.

Structured chunk:
"""{structured_chunk}"""

Cleaned transcript chunk:
"""{cleaned_chunk}"""

Raw transcript chunk (reference):
"""{raw_chunk}"""

Previously covered topics: {prior_topics}

Explain the material conversationally, highlight intuition and transitions, and end with a reflective question or quick self-check when appropriate. Output Markdown with a heading `## Segment {chunk_index}`.'''


class RussianLLMSummarizer:
    """Handles Russian transcription processing with English translation."""
    
    def __init__(self, config):
        """
        Initialize with Russian-specific config.
        
        Args:
            config: Config object with russian_model and english_model settings
        """
        self.config = config
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key
        )
        
        # Separate models for Russian and English processing
        self.russian_model = config.get('summarization.russian_model', 'anthropic/claude-3.5-sonnet')
        self.english_model = config.get('summarization.english_model', 'anthropic/claude-3.5-haiku')
        
        self.max_tokens = config.get('summarization.max_tokens', 4000)
        self.temperature = config.get('summarization.temperature', 0.1)
        self.chunk_chars = int(config.get('summarization.chunk_chars', 6000))
        self.chunk_overlap = int(config.get('summarization.chunk_overlap_paragraphs', 5))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def summarize(self, raw_transcript_text, course_folder, base_name, metadata=None, cleaned_transcript_text=None):
        """
        Generate both Russian and English summaries.
        
        Workflow:
        1. Process Russian transcript → 3 Russian files
        2. Translate to English
        3. Process English transcript → 3 English files
        
        Returns dict with paths to all 6 files.
        """
        print(f"\n🇷🇺 Generating Russian summaries with {self.russian_model}...")
        
        raw_transcript_text = raw_transcript_text or ""
        cleaned_transcript_text = cleaned_transcript_text or clean_transcript_text(raw_transcript_text)
        
        # ===== STEP 1: Generate Russian summaries =====
        russian_results = self._generate_language_summaries(
            raw_transcript_text,
            cleaned_transcript_text,
            language='ru',
            prompts={
                'narrative': NARRATIVE_CHUNK_PROMPT_RU,
                'guide': GUIDE_CHUNK_PROMPT_RU,
                'professor': PROFESSOR_CHUNK_PROMPT_RU
            },
            model=self.russian_model
        )
        
        # ===== STEP 2: Translate to English =====
        print(f"\n🔄 Translating transcript to English with {self.russian_model}...")
        english_transcript = self._translate_to_english(cleaned_transcript_text)
        
        # Save translated transcript
        course_folder_path = Path(course_folder)
        date_str = extract_date_fragment(base_name)
        translated_path = course_folder_path / f"{date_str}-translated-en.txt"
        translated_path.write_text(english_transcript)
        print(f"✅ English translation saved: {translated_path}")
        
        # ===== STEP 3: Generate English summaries =====
        print(f"\n🇬🇧 Generating English summaries with {self.english_model}...")
        english_results = self._generate_language_summaries(
            english_transcript,
            english_transcript,  # Already cleaned by translation
            language='en',
            prompts={
                'narrative': NARRATIVE_CHUNK_PROMPT_EN,
                'guide': GUIDE_CHUNK_PROMPT_EN,
                'professor': PROFESSOR_CHUNK_PROMPT_EN
            },
            model=self.english_model
        )
        
        # ===== STEP 4: Save all files =====
        course_name = metadata.get('course', 'Unknown') if metadata else 'Unknown'
        duration = metadata.get('duration', 0) if metadata else 0
        frontmatter = build_frontmatter(base_name, course_name, duration)
        
        # Russian files
        structured_ru = course_folder_path / f"{date_str}-structured-ru.md"
        guide_ru = course_folder_path / f"Guide {date_str}-ru.md"
        professor_ru = course_folder_path / f"Professor {date_str}-ru.md"
        
        structured_ru.write_text(self._compose_document(frontmatter, russian_results['structured']))
        guide_ru.write_text(self._compose_document(frontmatter, russian_results['guide']))
        professor_ru.write_text(self._compose_document(frontmatter, russian_results['professor']))
        
        # English files
        structured_en = course_folder_path / f"{date_str}-structured-en.md"
        guide_en = course_folder_path / f"Guide {date_str}-en.md"
        professor_en = course_folder_path / f"Professor {date_str}-en.md"
        
        structured_en.write_text(self._compose_document(frontmatter, english_results['structured']))
        guide_en.write_text(self._compose_document(frontmatter, english_results['guide']))
        professor_en.write_text(self._compose_document(frontmatter, english_results['professor']))
        
        print(f"\n✅ Russian summaries saved:")
        print(f"   - {structured_ru}")
        print(f"   - {guide_ru}")
        print(f"   - {professor_ru}")
        print(f"\n✅ English summaries saved:")
        print(f"   - {structured_en}")
        print(f"   - {guide_en}")
        print(f"   - {professor_en}")
        
        return {
            'russian': {
                'structured_path': str(structured_ru),
                'guide_path': str(guide_ru),
                'professor_path': str(professor_ru)
            },
            'english': {
                'structured_path': str(structured_en),
                'guide_path': str(guide_en),
                'professor_path': str(professor_en),
                'translated_transcript_path': str(translated_path)
            }
        }

    def _translate_to_english(self, russian_text: str) -> str:
        """Translate Russian transcript to English."""
        # Split into chunks if text is too long
        chunks = chunk_text(russian_text, self.chunk_chars * 2, 0)  # Larger chunks for translation
        
        translated_chunks = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"   Translating chunk {idx}/{len(chunks)}...")
            prompt = TRANSLATION_PROMPT.format(russian_text=chunk)
            translation = self._generate_text(prompt, self.russian_model)
            translated_chunks.append(translation.strip())
        
        return "\n\n".join(translated_chunks)

    def _generate_language_summaries(self, raw_text, cleaned_text, language, prompts, model):
        """Generate structured, guide, and professor summaries for a specific language."""
        # Generate structured chunks
        structured_chunks, raw_chunks, cleaned_chunks = self._generate_structured_chunks(
            raw_text, cleaned_text, prompts['narrative'], model
        )
        structured_body = "\n\n".join(chunk for chunk in structured_chunks if chunk).strip()
        
        # Generate guide
        guide_body = self._generate_guide(
            structured_chunks, cleaned_chunks, raw_chunks, prompts['guide'], model
        )
        
        # Generate professor recap
        professor_body = self._generate_professor(
            structured_chunks, cleaned_chunks, raw_chunks, prompts['professor'], model
        )
        
        return {
            'structured': structured_body,
            'guide': guide_body,
            'professor': professor_body
        }

    def _generate_structured_chunks(self, raw_text, cleaned_text, prompt_template, model):
        """Generate structured narrative chunks."""
        raw_chunks = chunk_text(raw_text, self.chunk_chars, self.chunk_overlap)
        
        print(f"🔍 Created {len(raw_chunks)} chunks from {len(raw_text):,} characters")
        for idx, chunk in enumerate(raw_chunks, 1):
            print(f"   Chunk {idx}: {len(chunk):,} chars")
        
        cleaned_chunks = []
        structured_chunks = []
        recent_headings = []
        
        total = len(raw_chunks)
        if total == 0:
            return structured_chunks, raw_chunks, cleaned_chunks
        
        for idx, raw_chunk in enumerate(raw_chunks, start=1):
            print(f"🔄 Processing structured chunk {idx}/{total}...")
            
            cleaned_chunk = clean_transcript_text(raw_chunk)
            cleaned_chunks.append(cleaned_chunk)
            
            prior_topics = "\n".join(recent_headings[-5:]) if recent_headings else "None yet."
            context_instruction = (
                "First chunk: start with `# Classroom Lesson Narrative` and a `> [!note] Context` callout summarizing the lecture goals before other sections."
                if idx == 1 else
                "Continue with new `##` sections only; do NOT repeat the top-level header or context callout."
            )
            
            prompt = prompt_template.format(
                chunk_index=idx,
                total_chunks=total,
                prior_topics=prior_topics,
                context_instruction=context_instruction,
                cleaned_chunk=cleaned_chunk,
                raw_chunk=raw_chunk
            )
            
            response = self._generate_text(prompt, model)
            structured_chunk = response.strip()
            structured_chunks.append(structured_chunk)
            recent_headings.extend(self._extract_headings(structured_chunk))
        
        return structured_chunks, raw_chunks, cleaned_chunks

    def _generate_guide(self, structured_chunks, cleaned_chunks, raw_chunks, prompt_template, model):
        """Generate action guide."""
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
            print(f"🔄 Processing guide chunk {idx}/{total}...")
            
            prompt = prompt_template.format(
                chunk_index=idx,
                total_chunks=total,
                structured_chunk=structured_chunk,
                cleaned_chunk=cleaned_chunk,
                raw_chunk=raw_chunk
            )
            
            response = self._generate_text(prompt, model)
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

    def _generate_professor(self, structured_chunks, cleaned_chunks, raw_chunks, prompt_template, model):
        """Generate professor recap."""
        outputs = []
        recent_headings = []
        total = len(structured_chunks)
        
        for idx, (structured_chunk, cleaned_chunk, raw_chunk) in enumerate(
            zip(structured_chunks, cleaned_chunks, raw_chunks), start=1
        ):
            print(f"🔄 Processing professor chunk {idx}/{total}...")
            
            prior_topics = "; ".join(recent_headings[-5:]) if recent_headings else "None yet."
            prompt = prompt_template.format(
                chunk_index=idx,
                total_chunks=total,
                prior_topics=prior_topics,
                structured_chunk=structured_chunk,
                cleaned_chunk=cleaned_chunk,
                raw_chunk=raw_chunk
            )
            
            response = self._generate_text(prompt, model)
            recap = response.strip()
            outputs.append(recap)
            recent_headings.extend(self._extract_headings(recap))
        
        return "\n\n".join(output for output in outputs if output)

    def _generate_text(self, prompt: str, model: str) -> str:
        """Generate text using specified model."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def _compose_document(self, frontmatter: str, body: str) -> str:
        """Compose final document with frontmatter."""
        body = body.strip()
        if not body:
            body = "*(No content generated.)*"
        return f"{frontmatter}\n\n{body}\n"

    def _extract_headings(self, markdown: str):
        """Extract markdown headings."""
        headings = []
        if not markdown:
            return headings
        for line in markdown.splitlines():
            clean = line.strip()
            if clean.startswith('##'):
                headings.append(clean.lstrip('#').strip())
        return headings

    def _parse_guide_json(self, response: str):
        """Parse JSON response from guide generation."""
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
        """Extend list with unique items."""
        if not items:
            return
        for item in items:
            cleaned = item.strip()
            if cleaned and cleaned not in target_list:
                target_list.append(cleaned)

    def _ensure_bullets(self, items):
        """Ensure items are formatted as bullets."""
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
        """Ensure items are formatted as checkboxes."""
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