"""LLM summarization with improved accuracy and minimal hallucination"""
import json
from collections import OrderedDict
from pathlib import Path
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .utils import build_frontmatter, chunk_text, extract_date_fragment

# Improved prompts with strict anti-hallucination instructions
NARRATIVE_CHUNK_PROMPT = '''You are transcribing lecture chunk {chunk_index}/{total_chunks}.

CRITICAL RULES:
1. ONLY use information explicitly stated in the transcript
2. NEVER add information not present in the transcript
3. If something is unclear, mark it as [unclear] rather than guessing
4. Preserve ALL technical terms, formulas, and numbers exactly as stated
5. Include timestamp markers when mentioned

Previous topics covered: {prior_topics}

{context_instruction}

Format requirements:
- Use ## headings for major topics
- Use > [!important] for key definitions/formulas
- Use bullet points for lists
- Include ALL examples and problems mentioned
- Mark unclear audio as [unclear] or [inaudible]

TRANSCRIPT CHUNK:
"""{chunk_text}"""

Output only the markdown notes for this chunk.'''

GUIDE_CHUNK_PROMPT = '''Extract ONLY factual information from this transcript chunk {chunk_index}/{total_chunks}.

STRICT RULES:
1. Extract ONLY what is explicitly stated
2. NO speculation or inference beyond what's said
3. Use empty array [] if nothing found for a category
4. Use exact terminology from the transcript

YOU MUST RESPOND WITH ONLY THIS JSON STRUCTURE - NO OTHER TEXT:
{{
  "topics_covered": [],
  "key_concepts": [],
  "assignments": [],
  "formulas": [],
  "examples": [],
  "important_dates": [],
  "questions_asked": [],
  "next_class": []
}}

DO NOT include any text before or after the JSON.
DO NOT wrap the JSON in markdown code blocks.
DO NOT add comments or explanations.
ONLY output the JSON object.

TRANSCRIPT:
"""{chunk_text}"""'''

VERIFICATION_PROMPT = '''Review this summary for accuracy against the original transcript.

ORIGINAL TRANSCRIPT:
"""{original}"""

GENERATED SUMMARY:
"""{summary}"""

Check for:
1. Any information in the summary NOT present in the transcript (hallucination)
2. Any important information from transcript missing in summary
3. Any misrepresented facts or numbers

Return JSON:
{{
  "has_hallucination": true/false,
  "missing_important_info": [],
  "corrections_needed": [],
  "accuracy_score": 0-100
}}'''


class LLMSummarizer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key
        )
        self.model = config.get('summarization.model', 'google/gemini-2.0-flash-exp')
        self.max_tokens = config.get('summarization.max_tokens', 4000)
        self.temperature = config.get('summarization.temperature', 0.1)
        self.chunk_chars = int(config.get('summarization.chunk_chars', 8000))
        self.chunk_overlap = int(config.get('summarization.chunk_overlap_paragraphs', 2))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def summarize(self, transcript_text, course_folder, base_name, metadata=None):
        """Generate accurate summaries with verification."""
        print(f"\n🤖 Generating summary with {self.model}...")
        print(f"📊 Transcript length: {len(transcript_text)} characters")
        
        # Pre-process transcript to clean it up
        transcript_text = self._preprocess_transcript(transcript_text)
        
        # Generate chunks with better overlap
        chunks = self._create_smart_chunks(transcript_text)
        print(f"📦 Split into {len(chunks)} chunks for processing")
        
        # Process chunks
        structured_notes = self._generate_structured_notes(chunks)
        study_guide = self._generate_study_guide(chunks)
        
        # Verify for hallucinations
        print("🔍 Verifying accuracy...")
        structured_notes = self._verify_and_correct(transcript_text, structured_notes)
        
        # Prepare metadata
        course_name = metadata.get('course', 'Unknown') if metadata else 'Unknown'
        duration = metadata.get('duration', 0) if metadata else 0
        frontmatter = build_frontmatter(base_name, course_name, duration)
        date_str = extract_date_fragment(base_name)
        
        # Save files
        structured_path = Path(course_folder) / f"{date_str}-notes.md"
        guide_path = Path(course_folder) / f"{date_str}-guide.md"
        
        # Add verification stats to the notes
        stats = self._calculate_coverage_stats(transcript_text, structured_notes)
        
        structured_content = f"{frontmatter}\n\n"
        structured_content += f"*Coverage: {stats['coverage_percent']:.1f}% of transcript processed*\n\n"
        structured_content += structured_notes
        
        guide_content = f"{frontmatter}\n\n{study_guide}"
        
        structured_path.write_text(structured_content)
        guide_path.write_text(guide_content)
        
        print(f"✅ Structured notes saved: {structured_path}")
        print(f"✅ Study guide saved: {guide_path}")
        print(f"📈 Coverage: {stats['coverage_percent']:.1f}% of content captured")
        
        return {
            'structured_path': str(structured_path),
            'guide_path': str(guide_path),
            'coverage_stats': stats
        }
    
    def _preprocess_transcript(self, text):
        """Clean up transcript before processing."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common transcription errors
        replacements = {
            ' gonna ': ' going to ',
            ' wanna ': ' want to ',
            ' gotta ': ' got to ',
            ' kinda ': ' kind of ',
            ' sorta ': ' sort of ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Mark unclear sections
        text = re.sub(r'\[inaudible\]', '[UNCLEAR]', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _create_smart_chunks(self, text):
        """Create chunks that preserve sentence boundaries."""
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds limit, save current chunk
            if current_size + sentence_size > self.chunk_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for context
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else []
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_structured_notes(self, chunks):
        """Generate structured notes from chunks."""
        all_notes = []
        prior_topics = []
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"  Processing chunk {idx}/{len(chunks)}...")
            
            context = "Start with # Lecture Notes" if idx == 1 else "Continue with new sections"
            prior = "\n".join(prior_topics[-5:]) if prior_topics else "None"
            
            prompt = NARRATIVE_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=len(chunks),
                prior_topics=prior,
                context_instruction=context,
                chunk_text=chunk
            )
            
            response = self._generate_text(prompt)
            all_notes.append(response)
            
            # Extract topics for context
            topics = re.findall(r'^##\s+(.+)$', response, re.MULTILINE)
            prior_topics.extend(topics)
        
        return "\n\n".join(all_notes)
    
    def _generate_study_guide(self, chunks):
        """Generate study guide from chunks."""
        all_data = {
            "topics_covered": [],
            "key_concepts": [],
            "assignments": [],
            "formulas": [],
            "examples": [],
            "important_dates": [],
            "questions_asked": [],
            "next_class": []
        }
        
        for idx, chunk in enumerate(chunks, 1):
            prompt = GUIDE_CHUNK_PROMPT.format(
                chunk_index=idx,
                total_chunks=len(chunks),
                chunk_text=chunk
            )
            
            response = self._generate_text(prompt)
            
            # More robust JSON parsing
            chunk_data = self._parse_json_response(response)
            
            if chunk_data:
                for key in all_data:
                    items = chunk_data.get(key, [])
                    # Filter out empty or "NOT MENTIONED" items
                    if items and isinstance(items, list):
                        valid_items = [item for item in items 
                                     if item and item != "NOT MENTIONED" and item != ""]
                        all_data[key].extend(valid_items)
            else:
                print(f"  ⚠️ Failed to parse JSON for chunk {idx}")
                # Try to extract at least basic info from the text
                if "assignment" in chunk.lower() or "homework" in chunk.lower():
                    all_data["assignments"].append(f"[Check chunk {idx} for assignment details]")
        
        # Remove duplicates while preserving order
        for key in all_data:
            all_data[key] = list(dict.fromkeys(all_data[key]))
        
        # Format as markdown
        guide = "# Study Guide\n\n"
        
        if all_data["topics_covered"]:
            guide += "## Topics Covered\n"
            for topic in all_data["topics_covered"]:
                guide += f"- {topic}\n"
            guide += "\n"
        
        if all_data["key_concepts"]:
            guide += "## Key Concepts\n"
            for concept in all_data["key_concepts"]:
                guide += f"- {concept}\n"
            guide += "\n"
        
        if all_data["formulas"]:
            guide += "## Formulas\n"
            for formula in all_data["formulas"]:
                guide += f"- `{formula}`\n"
            guide += "\n"
        
        if all_data["assignments"]:
            guide += "## Assignments & Deadlines\n"
            for assignment in all_data["assignments"]:
                guide += f"- [ ] {assignment}\n"
            guide += "\n"
        
        if all_data["important_dates"]:
            guide += "## Important Dates\n"
            for date in all_data["important_dates"]:
                guide += f"- 📅 {date}\n"
            guide += "\n"
        
        # Add a note if parsing had issues
        if not any(all_data.values()):
            guide += "*Note: Study guide extraction had issues. Please review the structured notes for complete information.*\n"
        
        return guide
    
    def _parse_json_response(self, response):
        """Robust JSON parsing that handles various LLM response formats."""
        if not response:
            return None
        
        # Clean the response
        response = response.strip()
        
        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, response)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the text
        try:
            # Find the first { and last }
            start = response.index('{')
            # Find matching closing brace
            count = 0
            end = start
            for i in range(start, len(response)):
                if response[i] == '{':
                    count += 1
                elif response[i] == '}':
                    count -= 1
                    if count == 0:
                        end = i
                        break
            
            if end > start:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass
        
        # Last resort: try to construct from text patterns
        try:
            result = {
                "topics_covered": [],
                "key_concepts": [],
                "assignments": [],
                "formulas": [],
                "examples": [],
                "important_dates": [],
                "questions_asked": [],
                "next_class": []
            }
            
            # Look for common patterns in the response
            lines = response.split('\n')
            current_category = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line matches a category
                for key in result.keys():
                    if key.replace('_', ' ') in line.lower():
                        current_category = key
                        break
                
                # If we have a category and this looks like an item
                if current_category and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                    item = line.lstrip('-*• ').strip()
                    if item and item not in ['[]', 'NOT MENTIONED', '']:
                        result[current_category].append(item)
            
            # Only return if we found something
            if any(result.values()):
                return result
                
        except Exception:
            pass
        
        return None
    
    def _verify_and_correct(self, original, summary):
        """Verify summary against original to prevent hallucination."""
        # For now, simple verification - can be enhanced
        # Check if all numbers in summary exist in original
        summary_numbers = re.findall(r'\b\d+\b', summary)
        original_numbers = set(re.findall(r'\b\d+\b', original))
        
        for num in summary_numbers:
            if num not in original_numbers and len(num) > 2:
                print(f"  ⚠️ Potential hallucination: number {num} not in original")
                summary = summary.replace(num, "[NUMBER]")
        
        return summary
    
    def _calculate_coverage_stats(self, original, summary):
        """Calculate how much of the original was captured."""
        # Extract key terms from both
        original_words = set(re.findall(r'\b[A-Z][a-z]+\b|\b\w{6,}\b', original))
        summary_words = set(re.findall(r'\b[A-Z][a-z]+\b|\b\w{6,}\b', summary))
        
        if not original_words:
            coverage = 0
        else:
            coverage = len(original_words & summary_words) / len(original_words) * 100
        
        return {
            'coverage_percent': coverage,
            'original_terms': len(original_words),
            'captured_terms': len(original_words & summary_words)
        }
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text with the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content