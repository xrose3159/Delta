"""OpenAI-compatible generator for producing harder math problems."""

from __future__ import annotations

import json
import logging
import ast
import re
import os
from typing import Any, Dict, List, Optional

try:  
    from openai import OpenAI  
except ImportError:  
    OpenAI = None 

logger = logging.getLogger(__name__)

REASONING_DIFFICULTY_SCALE_TEXT = """
**Reasoning Difficulty Scale (1.0 - 10.0, ordered levels)** — increments are not uniform; always move along the list below:
1: Beginner problems (MOEMS, AMC 8 1-10, AMC 10 1-10, easy AMC 12 1-5)
1.5: Strong beginner problems (AMC 8 11-20, harder AMC 10 1-10, AMC 12 1-5)
2: Motivated beginner, harder AMC 8 21-25, MATHCOUNTS Chapter/State, AMC 10 11-15
2.5: Advanced beginner, hardest AMC 8, harder MATHCOUNTS States, AMC 10 16-20, AMC 12 11-15
3: Early intermediate (harder MATHCOUNTS Nationals, AMC 10 21-25, AMC 12 15-20, easier AIME 1-3)
4: Intermediate (AMC 12 21-25, medium AIME 4-6, easy AIME 7-10)
5: Harder AIME 11-13, simple proof-style (JBMO, easiest USAJMO 1/4)
6: High AIME 14-15, introductory Olympiad (hard USAJMO 1/4, easy USAJMO 2/5, easy USAMO/IMO 1/4)
7: Tough Olympiad (harder USAJMO 2/5, most USAJMO 3/6, very hard USAMO, IMO 1/4, easy IMO 2/5)
8: High Olympiad (medium-hard USAMO, IMO 2/5, easiest IMO 3/6)
9: Expert Olympiad (average USAMO, IMO 3/6)
9.5: Hardest solvable Olympiad (hard USAMO, IMO 3/6)
10: Extreme research-level style problems (beyond IMO, very long/tedious)
""".strip()

REASONING_DIFFICULTY_LEVELS: List[float] = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
]

VISUAL_DIFFICULTY_SCALE_TEXT = """
**Visual Difficulty Scale (Level 1 - 7):**
1: Explicit Directness – plain visuals with fully labeled values and no distractions.
2: Symbolic Implication – relies on geometric/physical symbols (e.g., right-angle marks, circuit symbols).
3: Implicit Measurement – values encoded via grids, rulers, axes, or chart scales that must be read.
4: Noise Filtering – cluttered scenes with distractors requiring selective attention.
5: Spatial Perspective – 3D-to-2D projections, hidden edges, need for spatial reconstruction.
6: Topological Structure Analysis – dense networks (circuits/graphs) requiring path tracing and connectivity reasoning.
7: Geometric Transformation & Folding – requires mental rotation, folding, or transformation of the visual input.
""".strip()

class GeminiHardProblemGenerator:
    """Generate hard problems via an OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        base_url: str | None = None,
        max_tokens: int = 4096,
        max_output_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model_name = model.strip() or "gemini-2.5-pro"
        self.base_url = base_url.strip() if base_url else ""
        self.max_tokens = max_tokens if max_tokens and max_tokens > 0 else 4096
        self.max_output_tokens = max_output_tokens if max_output_tokens and max_output_tokens > 0 else None

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is empty. Hard problem generation will be skipped.")
            self.client = None
            return

        if OpenAI is None:
            logger.warning(
                "openai package not installed. Run `pip install openai` to enable hard problem generation."
            )
            self.client = None
            return

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        try: 
            self.client = OpenAI(**client_kwargs)
            logger.info(
                "Initialised OpenAI-compatible client (model='%s', base_url='%s')",
                self.model_name,
                self.base_url or "default",
            )
        except Exception as exc:  
            logger.error("Failed to initialise OpenAI-compatible client: %s", exc)
            self.client = None

        self.last_raw_payload: Optional[str] = None

    def get_last_payload(self) -> Optional[str]:
        return self.last_raw_payload

    def _run_generation(self, prompt: str, quota: int, category_name: str, image_path: Optional[str] = None) -> List[Dict[str, str]]:
        if quota <= 0:
            return []

        if self.client is None:
            logger.info("Gemini client unavailable. Skipping generation for category %s.", category_name)
            return []

        self.last_raw_payload = None
        
        if image_path:
            logger.info("🤖 Calling Gemini API with IMAGE for category '%s'... (this may take 10-60 seconds)", category_name)
        else:
            logger.info("🤖 Calling Gemini API to generate %d problems for category '%s'... (this may take 10-60 seconds)", quota, category_name)

        try:  
            user_content = prompt
            if image_path and os.path.exists(image_path):
                import base64
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                image_ext = os.path.splitext(image_path)[1].lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }.get(image_ext, 'image/jpeg')
                user_content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    }
                ]
                logger.debug("Added image to request: %s (type: %s)", image_path, mime_type)
            
            request_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert math problem author who creates problems in English. "
                            "ALL content must be in English only - questions, answers, and plot labels. "
                            "Output must be a single JSON object with field 'problems'. "
                            "Do not include markdown fences, commentary, or extra text. "
                            "When generating matplotlib code, use only standard syntax and English text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "temperature": 0.8,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            if self.max_output_tokens:
                request_kwargs.setdefault("extra_body", {})["max_output_tokens"] = self.max_output_tokens

            response = self.client.chat.completions.create(**request_kwargs)
            
            logger.info(" Gemini API call completed for category '%s'", category_name)
            
        except Exception as exc:  
            logger.error(
                "Gemini generation failed for category %s (quota=%s): %s",
                category_name,
                quota,
                exc,
            )
            return []

        payload = self._extract_text_response(response)
        self.last_raw_payload = payload
        if not payload:
            logger.warning("Gemini response contained no text for category %s.", category_name)
            return []

        normalised = self._normalise_payload(payload)
        if normalised is None:
            logger.warning(
                "Gemini response for category %s could not be normalised to JSON. Raw payload preview: %s",
                category_name,
                payload[:200].replace("\n", " "),
            )
            return []

        data = self._parse_json_payload(normalised, category_name)
        if data is None:
            return []

        problems = data.get("problems") if isinstance(data, dict) else None
        if not isinstance(problems, list):
            logger.warning("Gemini response missing 'problems' list for category %s.", category_name)
            return []

        results: List[Dict[str, str]] = []
        for item in problems:
            if not isinstance(item, dict):
                continue
            question = item.get("question")
            answer = item.get("answer")
            image_code = item.get("image_code")
            difficulty_type = item.get("difficulty_type")  
            
            if difficulty_type:
                logger.debug("Gemini returned difficulty_type: '%s'", difficulty_type)
            else:
                logger.warning("Gemini did NOT return difficulty_type field for a problem in category %s. Item keys: %s", 
                             category_name, list(item.keys()))
            
            if isinstance(question, str) and isinstance(answer, str):
                question = question.strip()
                answer = answer.strip()
                if isinstance(image_code, str):
                    image_code = image_code.strip()
                else:
                    image_code = ""
                if question and answer:
                    entry: Dict[str, str] = {"question": question, "answer": answer}
                    if image_code:
                        entry["image_code"] = image_code
                    if isinstance(difficulty_type, str) and difficulty_type.strip():
                        entry["difficulty_type"] = difficulty_type.strip()
                    results.append(entry)
            if len(results) >= quota:
                break

        if not results:
            logger.warning("Gemini returned no usable problems for category %s.", category_name)
        else:
            logger.info("📝 Successfully generated %d problems for category '%s' (requested: %d)", len(results), category_name, quota)
        
        return results

    @staticmethod
    def _extract_text_response(response: Any) -> Optional[str]:
        if response is None:
            return None

        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                        elif hasattr(part, "text"):
                            text = getattr(part, "text")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "".join(parts)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        return None

    @staticmethod
    def _close_json_fragment(fragment: str) -> str:
        """Balance braces/brackets/quotes in a JSON fragment as best as possible."""

        fragment = (fragment or "").strip()
        if not fragment:
            return fragment

        escaped = False
        quote_open = False
        for ch in fragment:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == '"':
                quote_open = not quote_open
        if quote_open:
            fragment += '"'

        brace_delta = fragment.count("{") - fragment.count("}")
        bracket_delta = fragment.count("[") - fragment.count("]")
        if brace_delta > 0:
            fragment += "}" * brace_delta
        if bracket_delta > 0:
            fragment += "]" * bracket_delta

        return fragment

    @staticmethod
    def _sanitize_candidate(candidate: str) -> str:
        candidate = GeminiHardProblemGenerator._close_json_fragment(candidate)
        candidate = re.sub(r'"question":\s*"([^"\\]*(?:\\.[^"\\]*)*)$', r'"question": "\1"', candidate)
        candidate = re.sub(r'"answer":\s*"([^"\\]*(?:\\.[^"\\]*)*)$', r'"answer": "\1"', candidate)
        candidate = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', candidate)
        return candidate

    def _parse_json_payload(self, text: str, category_name: str) -> Optional[Dict[str, Any]]:
        """Attempt to decode Gemini payload using several fallbacks."""

        if not text:
            return None

        base = self._sanitize_candidate(text)
        candidates = [base]

        cleaned = re.sub(r",\s*([}\]])", r"\1", base)
        if cleaned != base:
            candidates.append(cleaned)
            candidates.append(self._sanitize_candidate(cleaned))

        if candidates:
            last_candidate = candidates[-1]
            stripped = last_candidate.rstrip()
            if stripped and not stripped.endswith('"'):
                candidates.append(stripped + '"')

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(candidate, strict=False)
                except json.JSONDecodeError:
                    try:
                        value = ast.literal_eval(candidate)
                        if isinstance(value, dict):
                            return value
                    except (ValueError, SyntaxError):
                        continue

        logger.warning(
            "Failed to decode Gemini JSON for category %s after fallbacks. "
            "This usually means the response was truncated due to max_tokens limit. "
            "Current max_tokens=%d, max_output_tokens=%s. "
            "Consider increasing GEMINI_MAX_TOKENS or reducing quota per category. "
            "Payload preview: %s",
            category_name,
            self.max_tokens,
            self.max_output_tokens or "None",
            base[:400].replace("\n", " "),
        )
        return None

    @staticmethod
    def _normalise_payload(payload: str) -> Optional[str]:

        if not payload:
            return None

        text = payload.strip()

        if text.startswith("```"):
            parts = text.splitlines()
            if parts:
                parts = parts[1:]
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            text = "\n".join(parts).strip()

        start = text.find("{")
        if start != -1:
            fragment = GeminiHardProblemGenerator._extract_balanced_json(text, start)
            if fragment:
                return fragment

        if text.startswith("{") or text.startswith("["):
            return GeminiHardProblemGenerator._close_json_fragment(text)

        return None

    @staticmethod
    def _extract_balanced_json(text: str, start: int) -> Optional[str]:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def fix_image_code(
        self,
        question: str,
        answer: str,
        original_code: str,
        error_message: str,
    ) -> Optional[str]:
        if not self.client:
            return None
        
        prompt = self._build_fix_code_prompt(question, answer, original_code, error_message)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens or self.max_tokens,
                temperature=0.7,
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.warning("Gemini returned empty response for code fix")
                return None
            
            fixed_code = self._extract_code_from_response(content)
            if fixed_code:
                logger.info("Successfully got fixed code from Gemini")
                return fixed_code
            else:
                logger.warning("Could not extract fixed code from Gemini response")
                return None
                
        except Exception as exc:
            logger.error("Failed to get code fix from Gemini: %s", exc)
            return None
    
    def _build_fix_code_prompt(
        self,
        question: str,
        answer: str,
        original_code: str,
        error_message: str,
    ) -> str:
        return f"""You are a Python expert specializing in matplotlib and numpy. A code snippet has failed with an error.

**Question**: {question}
**Answer**: {answer}

**Original Code** (that failed):
```python
{original_code}
```

**Error Message**:
```
{error_message}
```

**Task**: Fix the code so it runs successfully without errors.

**CRITICAL Requirements:**
1. Use ONLY matplotlib.pyplot (as plt) and numpy (as np)
2. Code must be self-contained and executable with exec()
3. End with: plt.savefig('problem.png')
4. ALL text in the plot MUST be in English (no Chinese or other languages)
5. Fix the specific error mentioned above
6. **EXTREMELY IMPORTANT**: Carefully check ALL variable names - every variable MUST be defined before use!

**Common Fixes:**
- If error mentions 'rgba': Change 'rgba(255,0,0,0.5)' to (1.0, 0.0, 0.0, 0.5) or '#FF0000'
- If error mentions LaTeX/ParseFatalException: Use raw strings r'$...$' for math expressions
- **If error mentions undefined variable (NameError)**: 
  * Carefully read through ALL the code
  * Find where the variable is used
  * Make sure it is defined BEFORE that line
  * Check for typos in variable names
  * If a value is needed from the problem, calculate or define it explicitly
- If error mentions Chinese fonts: Replace Chinese text with English

**Output Format:**
- Output ONLY the fixed Python code
- Do NOT include markdown code fences (```)
- Do NOT include explanations or comments
- The code must be valid Python that can be executed directly with exec()

Fixed code:"""

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        if "import" in response and "plt.savefig" in response:
            lines = response.strip().split("\n")
            code_lines = []
            started = False
            for line in lines:
                if "import" in line:
                    started = True
                if started:
                    code_lines.append(line)
            if code_lines:
                return "\n".join(code_lines).strip()
        
        return None

    def upgrade_problem_difficulty(
        self,
        problem: str,
        answer: str,
        image_path: str,
        category_id: int,
        category_name: str,
        difficulty_aspect: str,
        current_reasoning_level: Optional[float] = None,
        current_visual_level: Optional[float] = None,
        target_reasoning_level: Optional[float] = None,
        target_visual_level: Optional[float] = None,
    ) -> Optional[Dict[str, str]]:
        logger.info("Upgrading problem difficulty (aspect: %s)", difficulty_aspect)
        
        if difficulty_aspect == "reasoning":
            difficulty_instruction = """
**Upgrade Type**: REASONING DIFFICULTY
- Increase the mathematical/logical complexity
- Add more steps or calculations
- Introduce multi-step reasoning
- Add constraints or conditions
- Make relationships more complex
- **IMPORTANT**: Keep the visualization similar to the original (same visual complexity)
- Update the image only to reflect new values/labels in the upgraded question
- Do NOT leave image_code empty - always generate the visualization code
Example upgrades:
  - Simple calculation → Multi-step calculation with intermediate variables
  - Single condition → Multiple conditions that must be satisfied
  - Direct formula → Requires deriving or combining formulas
"""
        elif difficulty_aspect == "visual":
            difficulty_instruction = """
**Upgrade Type**: VISUAL DIFFICULTY
- Increase visual complexity in the diagram
- Add more geometric elements or shapes
- Make spatial relationships more complex
- Add overlapping or hidden elements
- Increase the visual information to process
Example upgrades:
  - Simple shape → Composite shape with multiple parts
  - Single object → Multiple overlapping objects
  - 2D → More complex 2D or pseudo-3D representation
  - Clear labels → Implied or partially labeled elements
"""
        elif difficulty_aspect == "similar":
            difficulty_instruction = """
**Upgrade Type**: SIMILAR DIFFICULTY (Different Content for Diversity)
- Keep the SAME difficulty level as the original problem
- Change the specific numbers, objects, or scenario
- Use different mathematical approach or context
- Ensure the problem tests similar skills but with different content
- The goal is DIVERSITY, not difficulty increase
Example variations:
  - Triangle area → Rectangle perimeter (same difficulty, different shape)
  - Speed problem with car → Speed problem with train (same logic, different context)
  - Algebra with x,y → Algebra with a,b (same structure, different variables)
"""
        else:
            difficulty_instruction = """
**Upgrade Type**: GENERAL DIFFICULTY
- Increase the complexity of the problem
- Add more steps or complexity
"""
        
        def _format_level(value: Optional[float | int]) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return format(value, "g")
            return str(value)
        
        level_guidance_lines = []
        formatted_current_reasoning = _format_level(current_reasoning_level)
        formatted_target_reasoning = _format_level(target_reasoning_level)
        formatted_current_visual = _format_level(current_visual_level)
        formatted_target_visual = _format_level(target_visual_level)

        reasoning_upgrade_possible = (
            formatted_target_reasoning
            and formatted_current_reasoning
            and formatted_target_reasoning != formatted_current_reasoning
        )
        visual_upgrade_possible = (
            formatted_target_visual
            and formatted_current_visual
            and formatted_target_visual != formatted_current_visual
        )
        
        if formatted_current_reasoning and reasoning_upgrade_possible:
            level_guidance_lines.append(f"- Current reasoning level: {formatted_current_reasoning}")
        if reasoning_upgrade_possible and formatted_target_reasoning:
            level_guidance_lines.append(f"- Target reasoning level: {formatted_target_reasoning}")
        if formatted_current_visual and visual_upgrade_possible:
            level_guidance_lines.append(f"- Current visual level: {formatted_current_visual}")
        if visual_upgrade_possible and formatted_target_visual:
            level_guidance_lines.append(f"- Target visual level: {formatted_target_visual}")
        if difficulty_aspect == "reasoning" and not reasoning_upgrade_possible:
            level_guidance_lines.append(
                "- No higher labeled reasoning level exists; still create a meaningfully harder reasoning challenge (ignore numeric tags)."
            )
        if difficulty_aspect == "visual" and not visual_upgrade_possible:
            level_guidance_lines.append(
                "- Visual difficulty is already at the top label; nevertheless, increase the visual complexity beyond the current depiction."
            )
        
        if level_guidance_lines:
            level_guidance_lines.append(
                "- Match the target levels precisely. Only change the aspect being upgraded."
            )
            level_guidance = "\n".join(["**Difficulty Level Guidance:**", *level_guidance_lines])
        else:
            level_guidance = ""
        
        prompt = f"""You are an expert at creating challenging math problems.

**Original Problem:**
{problem}

**Original Answer:**
{answer}

**Category**: {category_name} (ID: {category_id})

{difficulty_instruction}

{level_guidance}

{REASONING_DIFFICULTY_SCALE_TEXT}

{VISUAL_DIFFICULTY_SCALE_TEXT}

**Task**: Create an UPGRADED version of this problem with increased difficulty according to the upgrade type above.

**CRITICAL Requirements:**
1. Keep the same category and general topic
2. Align the upgraded problem with the specified target reasoning/visual levels (roughly one step per upgrade request)
3. Maintain mathematical correctness and ensure the problem is solvable
4. Provide a clear, correct answer
5. **MUST ALWAYS generate Python code for visualization** - the image_code field is REQUIRED and cannot be empty

**Output Format** (JSON):
{{
  "question": "The upgraded problem text",
  "answer": "The correct answer (numerical or short text)",
  "image_code": "Python code using ONLY matplotlib and numpy to generate the visualization"
}}

**Python Code Requirements:**
- Use ONLY: import matplotlib.pyplot as plt, import numpy as np
- Self-contained, executable with exec()
- End with: plt.savefig('problem.png')
- ALL text in plot must be in English
- Use colors: (R/255, G/255, B/255) format or '#RRGGBB'
- Avoid Chinese characters
- **IMPORTANT**: The image_code field must NEVER be empty. Always provide valid visualization code.
  * For REASONING upgrades: Keep the visualization similar to the original
  * For VISUAL upgrades: Make the visualization significantly more complex
  * For SIMILAR upgrades: Create a similar visualization for the new scenario

Output ONLY the JSON (no explanations):"""

        try:
            if self.client is None:
                logger.warning("Gemini client unavailable. Cannot upgrade problem.")
                return None
            
            request_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert math problem author who creates challenging problems in English. "
                            "ALL content must be in English only. "
                            "Output must be a single JSON object with fields 'question', 'answer', and 'image_code'. "
                            "The 'image_code' field is REQUIRED and must contain valid Python visualization code - it cannot be empty or omitted. "
                            "Do not include markdown fences or extra text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.8,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            if self.max_output_tokens:
                request_kwargs.setdefault("extra_body", {})["max_output_tokens"] = self.max_output_tokens
            
            response = self.client.chat.completions.create(**request_kwargs)
            
            response_text = self._extract_text_response(response)
            if not response_text:
                logger.warning("Empty response from Gemini for upgrade")
                return None
            
            logger.debug("Gemini response (first 200 chars): %s", response_text[:200])
            
            normalized = self._normalise_payload(response_text)
            if not normalized:
                logger.warning("Failed to normalize Gemini response for upgrade")
                return None
            
            result = self._parse_json_payload(normalized, f"upgrade_{difficulty_aspect}")
            if not result or not isinstance(result, dict):
                logger.warning("Failed to parse JSON from upgrade response")
                return None
            
            if "question" not in result or "answer" not in result:
                logger.warning("Missing required fields in upgrade response: %s", list(result.keys()))
                return None
            
            result["question"] = str(result.get("question", "")).strip()
            result["answer"] = str(result.get("answer", "")).strip()
            if "image_code" in result:
                result["image_code"] = str(result.get("image_code", "")).strip()
            
            logger.info("✅ Successfully upgraded problem (difficulty: %s)", difficulty_aspect)
            return result
            
        except Exception as exc:
            logger.error("Failed to upgrade problem difficulty: %s", exc)
            import traceback
            logger.debug("Full traceback: %s", traceback.format_exc())
            return None


