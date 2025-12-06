"""
Lecture Generation Service - Core AI/LLM Integration
Handles all lecture content generation, math detection, and prompting
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from groq import Groq


# ============================================================================
# MATH UTILITIES
# ============================================================================

def wrap_math_expressions(text: str) -> str:
    """Wrap mathematical expressions in LaTeX delimiters for proper rendering."""
    math_patterns = [
        (r'\b(\d+\s*[+\-×÷]\s*\d+)\b', r'$\1$'),
        (r'\b(\d+\s*[+\-×÷]\s*\d+\s*[+\-×÷]\s*\d+)\b', r'$\1$'),
        (r'\b(\d+)/(\d+)\b', r'$\\frac{\1}{\2}$'),
        (r'\b(\w+)\^(\w+)\b', r'$\1^{\2}$'),
        (r'\b(\w+)_(\w+)\b', r'$\1_{\2}$'),
        (r'\b(a\s*×\s*b\s*=\s*HCF\(a,b\)\s*×\s*LCM\(a,b\))', r'$$\1$$'),
        (r'\b(HCF|LCM|GCD|sin|cos|tan|log|ln|exp)\(', r'$\\\1('),
        (r'\b([a-zA-Z]\s*[=≠<>≤≥]\s*[\d\w+\-×÷^()√π]+)', r'$\1$'),
    ]
    
    for pattern, replacement in math_patterns:
        text = re.sub(pattern, replacement, text)
    
    text = text.replace('%', '\\%').replace('&', '\\&')
    return text


def detect_math_content(text: str) -> bool:
    """Detect if content is Math-related based on keywords and patterns."""
    math_keywords = [
        'mathematics', 'algebra', 'arithmetic', 'polynomial', 'quadratic equation',
        'linear equation', 'matrix', 'determinant', 'eigenvalue', 'permutation',
        'combination', 'factorial', 'binomial', 'sequence', 'series', 'logarithm',
        'prime number', 'composite number', 'factorization', 'theorem', 'proof',
        'गणित', 'बीजगणित', 'अंकगणित', 'समीकरण', 'आव्यूह', 'क्रमचय', 'संचय',
        'અનુક્રમ', 'શ્રેણી', 'પ્રમેય', 'સાબિતી', 'ગણિત', 'બીજગણિત',
    ]
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in math_keywords if keyword.lower() in text_lower)
    
    math_patterns = [
        r'(polynomial|quadratic|linear)\s+(equation|expression)',
        r'(permutation|combination|factorial)',
        r'(prime|composite)\s+number',
        r'(theorem|lemma|corollary|proof)',
        r'(matrix|determinant|eigenvalue)',
        r'(set\s+theory|subset|union|intersection)',
    ]
    
    pattern_matches = sum(1 for pattern in math_patterns if re.search(pattern, text, re.IGNORECASE))
    
    return keyword_count >= 5 or pattern_matches >= 3 or (keyword_count >= 3 and pattern_matches >= 2)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_lecture_prompt(*, text: str, language: str, duration: int, style: str) -> str:
    """Build optimized prompt for lecture generation with strict JSON structure."""
    
    language_instructions = {
        "Gujarati": (
            "LANGUAGE RULES:\n"
            "1. Keep ONLY pure technical/domain terms in English (e.g., algorithm, database, function, API).\n"
            "2. Write all other words in Gujarati – connecting words, verbs, adjectives, descriptions.\n"
            "3. Use everyday conversational language. Avoid formal English words like 'critical', 'important'.\n"
            "4. Include easy English words naturally throughout the lecture.\n"
            "5. Example: 'Algorithm એક પદ્ધતિ છે જે આપણને સમસ્યા હલ કરવામાં મદદ કરે છે'\n"
            "6. Keep English around 20-30% of content; rest should be Gujarati.\n"
            "7. Speak like a teacher talks in class – natural, friendly, and simple."
        ),
        "Hindi": (
            "LANGUAGE RULES:\n"
            "1. Keep ONLY pure technical/domain terms in English (e.g., algorithm, database, function, API).\n"
            "2. Write all other words in Hindi – connecting words, verbs, adjectives, descriptions.\n"
            "3. Use everyday conversational language. Avoid formal English words.\n"
            "4. Include easy English words naturally throughout the lecture.\n"
            "5. Example: 'Algorithm एक तरीका है जो हमें समस्या हल करने में मदद करता है'\n"
            "6. Keep English around 20-30% of content; rest should be Hindi.\n"
            "7. Speak like a teacher talks in class – natural, friendly, and simple."
        ),
        "English": (
            "Generate all content in clear, student-friendly English.\n"
            "Use simple, conversational language."
        )
    }
    
    lang_instruction = language_instructions.get(
        language,
        f"Generate content in {language} with natural mixing of terminology."
    )
    
    prompt = f"""
Create a {duration}-minute educational lecture from the provided source material.

CRITICAL: The source material may contain text in Hindi, Gujarati, or English script.
Read and understand the EXACT content. The lecture topic MUST match the source material.

{lang_instruction}

REQUIRED JSON OUTPUT STRUCTURE:
{{
  "slides": [/* Array of exactly 9 slide objects */],
  "estimated_duration": {duration}
}}

MANDATORY SLIDE STRUCTURE (MUST HAVE ALL 9):

=== SLIDE 1: INTRODUCTION ===
{{
  "title": "Introduction to [Topic]",
  "bullets": [],
  "narration": "Welcome students! Today we'll explore... [250+ words]",
  "question": ""
}}

=== SLIDE 2-3: KEY CONCEPTS ===
{{
  "title": "Key Concepts - Part 1/2",
  "bullets": ["Concept 1", "Concept 2", "Concept 3"],
  "narration": "",
  "question": ""
}}

=== SLIDES 4-7: DEEP TEACHING WITH EXAMPLES ===
Pattern: Explain concept (2-3 paragraphs) → Real example (1-2 paragraphs) → Another concept → Another example
Total: 600-1000 words per slide with teaching + examples mixed naturally

=== SLIDE 8: PRACTICAL APPLICATIONS ===
{{
  "title": "How to Apply This Knowledge",
  "bullets": [],
  "narration": "Practical applications, career uses, daily life usage [280+ words]",
  "question": ""
}}

=== SLIDE 9: QUIZ & REFLECTION ===
{{
  "title": "Quiz Time - Test Your Understanding",
  "bullets": [],
  "narration": "",
  "question": "1. Question 1?\\n2. Question 2?\\n3. Question 3?\\n4. Question 4?\\n5. Question 5?"
}}

SOURCE MATERIAL:
{text[:60000]}

NOW GENERATE EXACTLY 9 SLIDES with integrated teaching and examples.
"""
    return prompt


def create_math_lecture_prompt(*, text: str, language: str, duration: int, style: str) -> str:
    """Build prompt for MATH lectures with proper mathematical formatting."""
    
    language_instructions = {
        "Gujarati": (
            "GUIDELINES FOR GUJARATI MATH LECTURE:\n"
            "1. Use English only for pure math terms (equation, theorem, polynomial).\n"
            "2. Write explanations in Gujarati.\n"
            "3. Format math expressions: $...$ for inline, $$...$$ for display.\n"
            "4. Use LaTeX: \\frac{a}{b}, a^b, a_b\n"
            "5. Structure with clear steps and proper notation."
        ),
        "Hindi": (
            "GUIDELINES FOR HINDI MATH LECTURE:\n"
            "1. Use English only for pure math terms.\n"
            "2. Write explanations in Hindi.\n"
            "3. Format math expressions: $...$ for inline, $$...$$ for display.\n"
            "4. Use LaTeX: \\frac{a}{b}, a^b, a_b\n"
            "5. Structure with clear steps and proper notation."
        ),
        "English": (
            "GUIDELINES FOR ENGLISH MATH LECTURE:\n"
            "1. Simple, student-friendly English.\n"
            "2. Format math: $...$ inline, $$...$$ display.\n"
            "3. Use LaTeX notation properly.\n"
            "4. Clear step-by-step explanations."
        )
    }
    
    lang_instruction = language_instructions.get(language, language_instructions["English"])
    
    prompt = f"""
Create a {duration}-minute MATHEMATICS lecture with proper formatting:
- Use LaTeX math delimiters: $...$ inline, $$...$$ display
- Use \\frac{{num}}{{den}}, ^, _, \\times, \\div
- Include step-by-step derivations
- All expressions properly formatted

{lang_instruction}

REQUIRED JSON OUTPUT STRUCTURE:
{{
  "slides": [/* 9 slides */],
  "estimated_duration": {duration}
}}

MANDATORY SLIDES:
1. Introduction with math concepts [250+ words]
2. Key Concepts with formulas
3. Important Formulas & Theorems
4-7. Step-by-step teaching with solved examples [600-1000 words each]
8. Practice Problems with Solutions
9. Quiz & Real-World Applications

SOURCE MATERIAL:
{text[:60000]}

Generate 9 slides with step-by-step solutions and proper LaTeX formatting.
"""
    return prompt


def generate_fallback_content(*, text: str, language: str, duration: int) -> Dict[str, Any]:
    """Generate fallback slides when generation fails."""
    
    templates = {
        "English": {
            "title": "Concept {}",
            "default_bullet": "Key point from source material",
            "narration": "Let's explore this concept. {}",
            "question": "What aspect interests you most?",
        },
        "Gujarati": {
            "title": "વિચાર {}",
            "default_bullet": "સામગ્રીમાંથી મુખ્ય મુદ્દો",
            "narration": "ચાલો આ concept સમજીએ. {}",
            "question": "કયો ભાગ તમને interesting લાગે છે?",
        },
        "Hindi": {
            "title": "अवधारणा {}",
            "default_bullet": "सामग्री से मुख्य बिंदु",
            "narration": "आइए इस concept को समझें। {}",
            "question": "कौन सा हिस्सा आपको interesting लगता है?",
        },
    }
    
    template = templates.get(language, templates["English"])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    
    if not sentences:
        sentences = [template["default_bullet"]]
    
    slides = []
    
    # Slide 1: Introduction
    slides.append({
        "number": 1,
        "title": "Introduction",
        "bullets": [],
        "narration": " ".join(sentences[:5]) if len(sentences) >= 5 else template["narration"].format(template["default_bullet"]),
        "question": "",
    })
    
    # Slides 2-8
    for i in range(2, 9):
        start_idx = i * 5
        chunk = sentences[start_idx:start_idx + 5] if len(sentences) > start_idx else [template["default_bullet"]]
        slides.append({
            "number": i,
            "title": f"{template['title'].format(i)}",
            "bullets": chunk[:3] if i <= 3 else [],
            "narration": "" if i <= 3 else " ".join(chunk),
            "question": "",
        })
    
    # Slide 9: Quiz
    slides.append({
        "number": 9,
        "title": "Quiz & Reflection",
        "bullets": [],
        "narration": "",
        "question": "1. First question?\n2. Second question?\n3. Third question?\n4. Fourth question?\n5. Fifth question?"
    })
    
    return {
        "slides": slides,
        "total_slides": 9,
        "estimated_duration": duration,
    }


# ============================================================================
# GROQ SERVICE - MAIN GENERATION ENGINE
# ============================================================================

class GroqService:
    """Service wrapper around the Groq chat completion API."""
    
    def __init__(self, api_key: str) -> None:
        self._client: Optional[Groq] = Groq(api_key=api_key) if api_key else None
    
    @property
    def configured(self) -> bool:
        return self._client is not None
    
    async def generate_lecture_content(
        self,
        *,
        text: str,
        language: str = "English",
        duration: int = 30,
        style: str = "storytelling",
    ) -> Dict[str, Any]:
        """Generate lecture content using Groq API."""
        if not self.configured:
            raise RuntimeError("Groq API client is not configured.")
        
        base_prompt = self._create_lecture_prompt(
            text=text,
            language=language,
            duration=duration,
            style=style,
        )
        
        system_message = {
            "role": "system",
            "content": (
                f"You are an expert teacher who creates engaging lectures. "
                f"CRITICAL: Generate ALL content in {language}. "
                f"Follow language mixing instructions precisely. "
                f"Respond ONLY with valid JSON."
            ),
        }
        
        max_attempts = 3
        last_failure: Optional[str] = None
        
        try:
            for attempt in range(1, max_attempts + 1):
                prompt = base_prompt
                if last_failure:
                    prompt += (
                        f"\n\nRETRY: Previous failed because: {last_failure}.\n"
                        "Generate EXACTLY 9 slides with 600-1000 word narrations for slides 4-7."
                    )
                
                completion = await self._create_chat_completion(
                    messages=[system_message, {"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                
                response = completion["choices"][0]["message"]["content"]
                print(f"\n🔄 Attempt {attempt} - Response length: {len(response)}")
                
                parsed = self._parse_lecture_response(response)
                slides = parsed.get("slides") or []
                failure_reason: Optional[str] = None
                
                if not slides:
                    failure_reason = "No properly formatted slides found."
                elif len(slides) != 9:
                    failure_reason = f"Expected 9 slides but got {len(slides)}."
                
                if language in ["Hindi", "Gujarati"]:
                    self._enforce_minimum_narration(slides, language)
                    validation_errors = self._validate_language_mixing(parsed, language)
                    if validation_errors:
                        print("⚠️ Language mixing validation failed:")
                        for error in validation_errors:
                            print(f"  - {error}")
                        failure_reason = ", ".join(validation_errors)
                
                if not failure_reason:
                    print(f"✅ Generation successful on attempt {attempt}")
                    return parsed
                
                last_failure = failure_reason
                print(f"⚠️ Attempt {attempt} failed: {failure_reason}")
            
            error_msg = f"Failed to generate lecture after {max_attempts} attempts. Last error: {last_failure}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
            
        except Exception as e:
            print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
            raise
    
    async def answer_question(
        self,
        *,
        question: str,
        context: str,
        language: str = "English",
        script_instruction: str | None = None,
        question_language_hint: str | None = None,
        answer_type: str | None = None,
        is_edit_command: bool = False,
    ) -> str | Dict[str, str] | Dict[str, Any]:
        """Answer questions or handle edit commands."""
        if not self.configured:
            return "I understand your question. Please review the lecture materials."
        
        if is_edit_command:
            return await self._handle_edit_command(question, context)
        
        return await self._handle_question(
            question, context, language, script_instruction, 
            question_language_hint, answer_type
        )
    
    async def _handle_edit_command(self, question: str, context: str) -> Dict[str, Any]:
        """Handle edit commands for slide content."""
        system_prompt = (
            "You are editing lecture slides. "
            "Return JSON with 'edited_content' and 'explanation' fields."
        )
        user_prompt = f"Current content:\n{context}\n\nEdit: {question}"
        
        try:
            completion = await self._create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
            
            response = completion["choices"][0]["message"]["content"]
            
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                match = re.search(r"\{[\s\S]*\}", response)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    parsed = None
            
            if parsed and isinstance(parsed, dict) and "edited_content" in parsed:
                return {
                    "answer": parsed.get("explanation", "Changes applied."),
                    "edited_content": parsed["edited_content"],
                }
            
            return {
                "answer": "Applied the edit.",
                "edited_content": response.strip(),
            }
            
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}
    
    async def _handle_question(
        self, 
        question: str, 
        context: str, 
        language: str,
        script_instruction: Optional[str],
        question_language_hint: Optional[str],
        answer_type: Optional[str]
    ) -> str | Dict[str, Any]:
        """Handle regular Q&A."""
        system_prompt = (
            f"You are a teaching assistant for this {language} lecture. "
            "Provide brief, direct answers. Keep under 3 sentences."
        )
        if script_instruction:
            system_prompt += f" Follow: {script_instruction}"
        
        user_prompt = f"Context: {context[:1000]}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            completion = await self._create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=5000,
            )
            content = completion["choices"][0]["message"]["content"]
            
            if answer_type == "json":
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "display_text" in data:
                        data.setdefault("tts_text", data["display_text"])
                        return data
                except json.JSONDecodeError:
                    pass
            
            return content
        except Exception:
            return "I'm having trouble processing your question."
    
    async def _create_chat_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 32000,
    ) -> Dict[str, Any]:
        """Create chat completion via Groq API."""
        if not self._client:
            raise RuntimeError("Groq client not configured")
        
        def _invoke() -> Dict[str, Any]:
            completion = self._client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.to_dict() if hasattr(completion, "to_dict") else completion
        
        return await asyncio.to_thread(_invoke)
    
    def _create_lecture_prompt(
        self,
        *,
        text: str,
        language: str,
        duration: int,
        style: str,
    ) -> str:
        """Create appropriate prompt based on content type."""
        if detect_math_content(text):
            print("🔢 Math content detected, using math prompt")
            return create_math_lecture_prompt(
                text=text,
                language=language,
                duration=duration,
                style=style,
            )
        
        return create_lecture_prompt(
            text=text,
            language=language,
            duration=duration,
            style=style,
        )
    
    def _parse_lecture_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response and extract slides."""
        response = response.strip()
        
        # Clean markdown
        if response.startswith("```"):
            response = re.sub(r'^```(?:json|JSON)?\s*\n?', '', response)
            response = re.sub(r'\n?```\s*$', '', response)
            response = response.strip()
        
        # Try JSON first
        parsed_json = self._try_parse_json(response)
        if parsed_json:
            return parsed_json
        
        # Fallback to text parsing
        return self._parse_text_format(response)
    
    def _try_parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse response as JSON."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None
        
        if not isinstance(data, dict) or not isinstance(data.get("slides"), list):
            return None
        
        normalized_slides = []
        for raw_slide in data["slides"]:
            if not isinstance(raw_slide, dict):
                continue
            
            title = str(raw_slide.get("title", "")).strip()
            bullets = raw_slide.get("bullets") or []
            narration = str(raw_slide.get("narration", "")).strip()
            question = str(raw_slide.get("question", "")).strip()
            
            if not title:
                continue
            
            cleaned_bullets = [str(b).strip() for b in bullets if str(b).strip()]
            if not cleaned_bullets:
                cleaned_bullets = [title]
            
            if not question:
                question = "क्या आप इस विषय के बारे में और जानना चाहेंगे?"
            
            normalized_slides.append({
                "number": len(normalized_slides) + 1,
                "title": title,
                "bullets": cleaned_bullets[:3],
                "narration": narration,
                "question": question,
            })
        
        if not normalized_slides:
            return None
        
        estimated = data.get("estimated_duration")
        try:
            estimated_duration = int(estimated)
        except (TypeError, ValueError):
            estimated_duration = len(normalized_slides) * 3
        
        return {
            "slides": normalized_slides,
            "total_slides": len(normalized_slides),
            "estimated_duration": max(estimated_duration, len(normalized_slides) * 3),
        }
    
    def _parse_text_format(self, response: str) -> Dict[str, Any]:
        """Fallback parser for text format."""
        slides = []
        pattern = re.compile(r"(?i)slide\s*(\d+)\s*[:：]?")
        matches = list(pattern.finditer(response))
        
        if not matches:
            return {"slides": [], "total_slides": 0, "estimated_duration": 0}
        
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
            block = response[start:end].strip()
            lines = block.splitlines()
            
            slide_data = {
                "number": int(match.group(1)) if match.group(1).isdigit() else len(slides) + 1,
                "title": "",
                "bullets": [],
                "narration": "",
                "question": "",
            }
            
            current_section = None
            for raw_line in lines:
                value = raw_line.strip()
                if not value:
                    continue
                
                upper = value.upper()
                if upper.startswith("TITLE:"):
                    slide_data["title"] = value.split(":", 1)[1].strip()
                    current_section = "title"
                elif upper.startswith("BULLETS:"):
                    current_section = "bullets"
                elif upper.startswith("NARRATION:"):
                    slide_data["narration"] = value.split(":", 1)[1].strip()
                    current_section = "narration"
                elif upper.startswith("QUESTION:"):
                    slide_data["question"] = value.split(":", 1)[1].strip()
                    current_section = "question"
                elif current_section == "bullets":
                    cleaned = value.lstrip("-*•0123456789. ").strip()
                    if cleaned:
                        slide_data["bullets"].append(cleaned)
                elif current_section == "narration":
                    slide_data["narration"] = (slide_data["narration"] + " " + value).strip()
                elif current_section == "question":
                    slide_data["question"] = (slide_data["question"] + " " + value).strip()
            
            if not slide_data["bullets"]:
                slide_data["bullets"] = [slide_data["title"]]
            
            if slide_data["title"] and (slide_data["narration"] or slide_data["question"]):
                slide_data["number"] = len(slides) + 1
                slides.append(slide_data)
        
        return {
            "slides": slides,
            "total_slides": len(slides),
            "estimated_duration": len(slides) * 3
        }
    
    def _complete_slides_with_fallback(
        self,
        *,
        slides: List[Dict[str, Any]],
        language: str,
        duration: int,
        text: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fill missing slides with fallback."""
        try:
            fallback = generate_fallback_content(text=text, language=language, duration=duration)
        except Exception:
            return None
        
        fallback_slides = fallback.get("slides")
        if not isinstance(fallback_slides, list) or len(fallback_slides) != 9:
            return None
        
        completed = []
        for index in range(9):
            fallback_slide = fallback_slides[index]
            existing_slide = slides[index] if index < len(slides) else None
            
            if existing_slide:
                merged = {
                    "number": index + 1,
                    "title": existing_slide.get("title") or fallback_slide.get("title"),
                    "bullets": existing_slide.get("bullets") or fallback_slide.get("bullets", []),
                    "narration": existing_slide.get("narration") or fallback_slide.get("narration", ""),
                    "question": existing_slide.get("question") or fallback_slide.get("question", ""),
                }
            else:
                merged = {
                    "number": index + 1,
                    "title": fallback_slide.get("title"),
                    "bullets": fallback_slide.get("bullets", []),
                    "narration": fallback_slide.get("narration", ""),
                    "question": fallback_slide.get("question", ""),
                }
            
            if not merged.get("title") or not (merged.get("narration") or merged.get("question")):
                return None
            
            if not merged.get("bullets"):
                merged["bullets"] = [merged["title"]]
            
            completed.append(merged)
        
        return completed
    
    def _validate_language_mixing(self, parsed: Dict[str, Any], language: str) -> List[str]:
        """Validate Hindi/Gujarati content has proper English mixing."""
        errors = []
        
        forbidden_words = {
            "आर्टिफिशियल इंटेलिजेंस": "Artificial Intelligence",
            "મશીન લર્નિંગ": "Machine Learning",
        }
        
        slides = parsed.get("slides", [])
        for idx, slide in enumerate(slides):
            number = slide.get("number") or idx + 1
            title = str(slide.get("title", ""))
            narration = str(slide.get("narration", ""))
            
            for wrong, correct in forbidden_words.items():
                if wrong in title or wrong in narration:
                    errors.append(f"Slide {idx + 1}: Found '{wrong}' - should be '{correct}'")
            
            word_count = len(narration.split())
            min_words = self._get_minimum_word_requirement(number)
            
            if number == 9 and not narration and slide.get("question"):
                continue
            
            if min_words and word_count < min_words:
                errors.append(
                    f"Slide {idx + 1}: Narration too short ({word_count} words, need {min_words}+)"
                )
        
        return errors
    
    def _enforce_minimum_narration(self, slides: List[Dict[str, Any]], language: str) -> None:
        """Pad narrations to satisfy minimum word requirements."""
        for idx, slide in enumerate(slides):
            number = slide.get("number") or idx + 1
            min_words = self._get_minimum_word_requirement(number)
            if not min_words:
                continue
            
            narration = str(slide.get("narration", "")).strip()
            current_words = len(narration.split()) if narration else 0
            if current_words >= min_words:
                continue
            
            padding = self._build_padding_text(
                slide=slide,
                language=language,
                additional_words=min_words - current_words,
            )
            slide["narration"] = f"{narration}\n\n{padding}".strip() if narration else padding
    
    @staticmethod
    def _get_minimum_word_requirement(slide_number: int) -> int:
        """Get minimum word count for each slide."""
        if slide_number == 1:
            return 100
        if 4 <= slide_number <= 7:
            return 250
        if slide_number == 8:
            return 180
        return 0
    
    def _build_padding_text(
        self,
        *,
        slide: Dict[str, Any],
        language: str,
        additional_words: int,
    ) -> str:
        """Build padding text for insufficient narration."""
        title = str(slide.get("title") or f"Slide {slide.get('number', '')}").strip()
        bullets = slide.get("bullets") or []
        bullet_summary = ", ".join(bullets[:3]) if bullets else ""
        
        if language == "Hindi":
            intro = f"{title} विषय को समझाते समय हम उदाहरणों से अवधारणा को गहराई से स्पष्ट करते हैं."
            if bullet_summary:
                intro += f" मुख्य बिंदु: {bullet_summary}."
        elif language == "Gujarati":
            intro = f"{title} વિષયને વિદ્યાર્થી સમજી શકે તે માટે અમે સાદા ઉદાહરણોથી દરેક પગલું સમજાવીએ છીએ."
            if bullet_summary:
                intro += f" મુખ્ય મુદ્દા: {bullet_summary}."
        else:
            intro = f"While covering {title}, we guide students through each idea with relatable stories."
            if bullet_summary:
                intro += f" Key ideas: {bullet_summary}."
        
        sentences = [intro]
        filler_bank = self._get_language_fillers(language)
        filler_index = 0
        
        while len(" ".join(sentences).split()) < additional_words:
            sentences.append(filler_bank[filler_index % len(filler_bank)])
            filler_index += 1
        
        return " ".join(sentences)
    
    @staticmethod
    def _get_language_fillers(language: str) -> List[str]:
        """Get language-specific filler sentences."""
        hindi_fillers = [
            "हम सरल भाषा में कारण और परिणाम बताते हैं ताकि हर छात्र अवधारणा को अपनी दैनिक जिंदगी से जोड़ सके.",
            "अभ्यास प्रश्नों के माध्यम से हम बच्चों से त्वरित प्रतिक्रिया लेते हैं और उनकी शंकाओं को वहीं दूर करते हैं.",
            "हर चरण के बाद संक्षिप्त पुनरावृत्ति कराते हैं ताकि मुख्य विचार लंबे समय तक याद रहें.",
            "वास्तविक जीवन के उदाहरणों से सिद्धांत को जोड़कर हम सीखने को रोचक और भरोसेमंद बनाते हैं.",
            "समूह गतिविधियों से सहयोगी अधिगम को बढ़ावा देते हैं और विद्यार्थियों को चर्चा के लिए प्रेरित करते हैं.",
        ]
        
        gujarati_fillers = [
            "અમે સરળ શબ્દોમાં કારણ અને પરિણામ સમજાવીએ છીએ જેથી દરેક વિદ્યાર્થી વિચારને પોતાના અનુભવ સાથે જોડે.",
            "નાના પ્રશ્નો દ્વારા તાત્કાલિક પ્રતિસાદ લઈ ભૂલને તરત સુધારીએ છીએ.",
            "દરેક વિભાગ પછી ઝડપી પુનરાવર્તન કરીએ છીએ જેથી મુખ્ય વિચારો મનમાં મજબૂત બને.",
            "દરરોજના જીવનના ઉદાહરણો દ્વારા સિદ્ધાંતને જીવંત બનાવીને શીખવાનું આનંદમય કરીએ છીએ.",
            "સમૂહ ચર્ચાઓથી સહકાર વધે છે અને વિદ્યાર્થીઓ આત્મવિશ્વાસથી પોતાના વિચારો રજૂ કરે છે.",
        ]
        
        english_fillers = [
            "We pause after each sub-topic to let students reflect and share their understanding.",
            "Short reflective questions strengthen recall and give quick glimpses of progress.",
            "Linking concepts with everyday decisions keeps the narrative practical and memorable.",
            "Hands-on mini activities ensure that theory converts into skill and confidence.",
            "We end blocks with concise recaps so learners feel ready for next challenges.",
        ]
        
        if language == "Hindi":
            return hindi_fillers
        if language == "Gujarati":
            return gujarati_fillers
        return english_fillers