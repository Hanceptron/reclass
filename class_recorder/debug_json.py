#!/usr/bin/env python3
"""Debug tool to test and fix JSON parsing issues"""

import json
import sys
from pathlib import Path
from openai import OpenAI
from config import config

def test_json_extraction():
    """Test JSON extraction from a sample transcript."""
    
    # Sample transcript text
    sample_transcript = """
    Today we're going to discuss machine learning fundamentals. 
    First, let's define supervised learning - it's when we have labeled data.
    For homework, please complete exercises 3.1 through 3.5 by next Monday.
    The midterm exam will be on March 15th.
    Remember the formula: accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.openrouter_api_key
    )
    
    # Test with the strict JSON prompt
    prompt = f'''Extract information from this transcript.

YOU MUST RESPOND WITH ONLY THIS JSON STRUCTURE - NO OTHER TEXT:
{{
  "topics_covered": [],
  "key_concepts": [],
  "assignments": [],
  "formulas": [],
  "important_dates": []
}}

DO NOT include any text before or after the JSON.
DO NOT wrap the JSON in markdown code blocks.
ONLY output the JSON object.

TRANSCRIPT:
"""{sample_transcript}"""'''
    
    print("🧪 Testing JSON extraction...")
    print("=" * 50)
    print("Sending prompt to LLM...")
    
    response = client.chat.completions.create(
        model=config.get('summarization.model', 'google/gemini-2.5-flash'),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1
    )
    
    raw_response = response.choices[0].message.content
    
    print("\n📝 Raw LLM Response:")
    print("-" * 40)
    print(raw_response)
    print("-" * 40)
    
    # Try parsing
    print("\n🔍 Attempting to parse JSON...")
    
    try:
        parsed = json.loads(raw_response)
        print("✅ Direct parsing successful!")
        print("\nParsed data:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print(f"❌ Direct parsing failed: {e}")
        
        # Try alternative parsing methods
        print("\n🔧 Trying alternative parsing methods...")
        
        # Method 1: Remove markdown code blocks
        import re
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, raw_response)
        if matches:
            print("Found JSON in markdown code block...")
            try:
                parsed = json.loads(matches[0])
                print("✅ Markdown extraction successful!")
                print(json.dumps(parsed, indent=2))
            except:
                print("❌ Markdown extraction failed")
        
        # Method 2: Find JSON object
        try:
            start = raw_response.index('{')
            end = raw_response.rindex('}') + 1
            json_str = raw_response[start:end]
            parsed = json.loads(json_str)
            print("✅ JSON object extraction successful!")
            print(json.dumps(parsed, indent=2))
        except:
            print("❌ JSON object extraction failed")

def test_with_actual_transcript(transcript_file):
    """Test with an actual transcript file."""
    
    if not Path(transcript_file).exists():
        print(f"❌ File not found: {transcript_file}")
        return
    
    with open(transcript_file, 'r') as f:
        transcript = f.read()
    
    # Take just the first 1000 chars for testing
    sample = transcript[:1000]
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.openrouter_api_key
    )
    
    print(f"\n📄 Testing with actual transcript: {transcript_file}")
    print(f"Sample length: {len(sample)} characters")
    print("=" * 50)
    
    # Test simplified extraction
    simplified_prompt = f'''List the main topics from this transcript.
Format your response as:
TOPICS: topic1 | topic2 | topic3
CONCEPTS: concept1 | concept2 | concept3
ASSIGNMENTS: assignment1 | assignment2

Transcript: {sample}'''
    
    print("Testing simplified extraction...")
    
    response = client.chat.completions.create(
        model=config.get('summarization.model', 'google/gemini-2.5-flash'),
        messages=[{"role": "user", "content": simplified_prompt}],
        max_tokens=500,
        temperature=0.1
    )
    
    raw = response.choices[0].message.content
    print("\nSimplified response:")
    print(raw)
    
    # Parse the simplified format
    data = {"topics": [], "concepts": [], "assignments": []}
    for line in raw.split('\n'):
        if line.startswith('TOPICS:'):
            data['topics'] = [t.strip() for t in line[7:].split('|')]
        elif line.startswith('CONCEPTS:'):
            data['concepts'] = [c.strip() for c in line[9:].split('|')]
        elif line.startswith('ASSIGNMENTS:'):
            data['assignments'] = [a.strip() for a in line[12:].split('|')]
    
    print("\nParsed simplified data:")
    print(json.dumps(data, indent=2))

def suggest_fix():
    """Suggest configuration changes to fix the issue."""
    print("\n💡 SUGGESTED FIXES:")
    print("=" * 50)
    print("""
1. Try a different model that follows instructions better:
   - anthropic/claude-3-haiku (fast and accurate)
   - openai/gpt-4o-mini (good at structured output)
   
2. Use simplified extraction format instead of JSON:
   Replace complex JSON with simple key-value format
   
3. Add a fallback when JSON parsing fails:
   Extract basic info using regex patterns
   
4. Consider using OpenAI's function calling API:
   More reliable for structured output
   
To implement fix #2 (recommended), update your config.yaml:
   
summarization:
  use_simplified_extraction: true  # Add this line
  model: 'openai/gpt-4o-mini'  # Or try this model
""")

def main():
    """Run the debug tool."""
    if len(sys.argv) < 2:
        print("🔍 JSON Parsing Debug Tool\n")
        print("Usage:")
        print("  python debug_json.py test        - Test with sample")
        print("  python debug_json.py <file>      - Test with transcript file")
        print("  python debug_json.py fix         - Show fix suggestions")
        return
    
    command = sys.argv[1]
    
    if command == 'test':
        test_json_extraction()
        suggest_fix()
    elif command == 'fix':
        suggest_fix()
    else:
        # Assume it's a file path
        test_with_actual_transcript(command)
        suggest_fix()

if __name__ == '__main__':
    main()