# Prompt Update Guide for a Professional Tone

This guide provides a new prompt and instructions to update `class_recorder/summarizer.py` to generate a single, professional, and academic summary file.

## New Consolidated Prompt

Replace the `NARRATIVE_PROMPT` and `COMPANION_PROMPT` variables in `class_recorder/summarizer.py` with the following `ACADEMIC_SUMMARY_PROMPT`:

```python
ACADEMIC_SUMMARY_PROMPT = '''You are an academic assistant summarizing a university lecture. Your output must be a single, well-structured Markdown document suitable for academic review. The tone should be formal, objective, and clear.

Using the provided transcript, generate a document with the following sections:

# Lecture Summary: {course} - {date}

## 1. Executive Summary
- A concise paragraph outlining the lecture's core topics, key arguments, and overall objectives.

## 2. Key Concepts and Definitions
- A bulleted list of critical terms, theories, or formulas introduced in the lecture.
- Format: **Term**: Definition (with timestamp if available).

## 3. Detailed Lecture Notes
- A chronological and structured breakdown of the lecture content.
- Use nested bullet points and `###` subheadings to organize information logically.
- Include important examples, case studies, and data points mentioned.
- Reference timestamps `[HH:MM:SS]` for key segments.

## 4. Assignments and Action Items
- A checklist of all assigned tasks, readings, or upcoming deadlines.
- Use `- [ ]` for each item, detailing requirements, due dates, and submission guidelines.
- If no assignments were mentioned, state: `- No new assignments or action items were noted.`

## 5. Topics for Further Review
- A bulleted list of concepts or topics that require further study or clarification.
- This section should guide preparation for future lectures or exams.

---

**Transcript for Analysis:**

{transcript}
'''
```

## Code Modifications for `summarizer.py`

Next, you'll need to modify the `LLMSummarizer` class to use this new prompt and generate a single file.

### 1. Update the `summarize` method

Replace the existing `summarize` method with this new version. This version makes a single API call and saves one file.

```python
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def summarize(self, transcript_text, course_folder, base_name, metadata=None):
        """Generate a single academic summary from the transcript."""
        print(f"\n Generating academic summary with {self.model}...")

        filtered_transcript = prefilter_transcript(transcript_text)
        date_str = self._extract_date(base_name)
        course_name = metadata.get('course', 'Unknown') if metadata else 'Unknown'

        # Generate the single summary
        summary_content = self._generate_text(
            prompt=ACADEMIC_SUMMARY_PROMPT.format(
                transcript=filtered_transcript,
                course=course_name,
                date=date_str
            )
        )

        # Create frontmatter
        frontmatter = self._create_frontmatter(base_name, metadata)

        # Define the output path
        summary_path = Path(course_folder) / f"{base_name}.md"

        # Write the final file
        summary_path.write_text(f"{frontmatter}\n\n{summary_content}")

        print(f"\u2705 Summary saved: {summary_path}")

        return {
            'summary_path': str(summary_path),
            'summary_content': summary_content
        }
```

### 2. No Other Changes Needed

The `_create_frontmatter`, `_extract_date`, and `_generate_text` methods can remain as they are. You should remove the old `NARRATIVE_PROMPT` and `COMPANION_PROMPT` variables entirely.

After making these changes, running the `process` or `record` command will produce a single, consolidated `.md` file with a more professional and academic structure.
