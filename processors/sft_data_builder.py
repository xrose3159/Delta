import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass


logger = logging.getLogger(__name__)

SFT_SYSTEM_PROMPT = """You are an expert in science and visual reasoning with advanced capabilities in multimodal analysis. Your goal is to create a **perfect, highly detailed training example** for a new AI model. Do not summarize or abbreviate. Your reasoning must be **expansive, verbose, and pedagogical**.
        ### Core Principles
        1.  **Extreme Detail.** Prioritize depth over brevity. Explain the "why" and "how" behind every step, even simple ones.
        2.  **Visual Dependency.** Explicitly quote visual features (coordinates, colors, relative positions) in your reasoning to prove you are looking at the image.
        3.  **Atomic Reasoning.** Break down complex logic into small, atomic steps. Do not leap from premise to conclusion; build the bridge.
        4.  **Self-Correction.** If a step involves a common pitfall, explicitly mention it and explain how to avoid it.

        ### Response Structure & Detailed Instructions

        You **must** respond using **only** the three tags below in the exact order specified.

        #### 1. `<caption>` (Exhaustive Information Extraction)
        **Goal:** Create a text-based "digital twin" of the provided images and text.
        * **Visual Analysis:** Do not just list objects. Describe their relationships, exact values on axes, trend directions, specific colors used for specific data, and geometry properties.
        * **Text Analysis:** Copy out specific numbers and constraints verbatim before interpreting them.
        * **Synthesis:** Explicitly map text variables to image labels (e.g., "The variable $x$ in the text corresponds to the horizontal axis in Figure 1").
        * *Instruction:* Be verbose. Capture details that might seem minor, as they may be crucial later.

        #### 2. `<reasoning>` (Deep-Dive Solution Execution)
        **Goal:** A long-form, step-by-step derivation that reads like a detailed lecture.
        * **Step 1: Strategic Planning**
            * Restate the objective clearly.
            * List the specific formulas or scientific principles required.
            * Explain *why* these principles were chosen over others.
        * **Step 2: Execution (The "Thinking" Engine)**
            * **Expand Every Calculation:** Do not just show `$a + b = c$`. Instead, write: "Substitute $a=5$ and $b=3$ into the equation. This yields $5 + 3$, which equals $8$."
            * **Inner Monologue:** Explain the physical or logical meaning of intermediate results.
            * **Visual Check:** Continuously refer back to the `<caption>`. (e.g., "As seen in the graph, the curve peaks at $t=5$, which aligns with our calculated critical point.").
            * **Handling Complexity:** If a problem has multiple cases, analyze each one strictly and separately.
        * **Step 3: Verification**
            * Perform a sanity check on the magnitude and units of the result.
            * Does the answer physically make sense given the visual context?

        *Constraint:* **Do not omit algebra.** Show the manipulation of terms. If you are solving a system of equations, show the substitution or elimination steps explicitly.

        #### 3. `<answer>` (Final Conclusion)
        **Goal:** Provide the definitive result.
        * Format: `<answer>\\boxed{YOUR_ANSWER}</answer>`
        * For multiple choice, include the letter and the value.
        * Strictly no reasoning text inside this tag, only the final result.

        Analyze all provided materials carefully. **Write a lengthy, comprehensive, and meticulous response following the strictly defined format above.**
    """



def normalize_image_paths(image_path_raw: Union[str, List[str], None]) -> List[str]:
    if isinstance(image_path_raw, list):
        return image_path_raw
    elif isinstance(image_path_raw, str) and image_path_raw:
        return [image_path_raw]
    else:
        return []


def get_problem_id(problem: Dict[str, Any]) -> str:
    problem_id_val = problem.get('id')
    if problem_id_val is None or problem_id_val == 0:
        return str(hash(problem.get('problem', '')) % 100000)
    else:
        return str(problem_id_val)


@dataclass
class SFTDataPoint:
    instruction: str  
    response: str
    images: List[str] = None  
    system: str = None 
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "conversations": [
                {
                    "from": "user",
                    "value": self.instruction  
                },
                {
                    "from": "assistant",
                    "value": self.response 
                }
            ]
        }

        if self.system:
            data["system"] = self.system

        if self.images:
            data["images"] = self.images
        
        return data


class SFTDataBuilder:
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_sft_dataset(
        self,
        wrong_problems: List[Dict[str, Any]],
        output_file: Path,
        corrected_cots: Dict[str, str] = None
    ) -> tuple[int, int]:
        if not wrong_problems:
            self.logger.warning("No wrong problems provided")
            return 0, 0
        
        self.logger.info(f"Building SFT dataset from {len(wrong_problems)} wrong problems...")
        
        sft_data_points = []
        skipped_count = 0
        
        for problem in wrong_problems:
            problem_id = get_problem_id(problem)
            
            problem_text = problem.get('problem', '')
            if not problem_text:
                self.logger.warning(f"Problem {problem_id}: missing problem text, skipping")
                skipped_count += 1
                continue
            
            if corrected_cots and problem_id in corrected_cots:
                correct_answer = corrected_cots[problem_id]
            else:
                self.logger.warning(f"Problem {problem_id}: no corrected CoT available, skipping")
                skipped_count += 1
                continue
            
            image_paths = normalize_image_paths(problem.get('image_path'))
            
            sanitized_answer = correct_answer.replace("<image>", "the image")

            sft_point = SFTDataPoint(
                instruction=problem_text,
                response=sanitized_answer,
                images=image_paths,
                system=SFT_SYSTEM_PROMPT
            )
            
            sft_data_points.append(sft_point.to_dict())
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sft_data_points, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"SFT dataset saved to: {output_file}")
        self.logger.info(f"  Total SFT data points: {len(sft_data_points)}")
        self.logger.info(f"  Skipped problems: {skipped_count}")
        self.logger.info(f"  Success rate: {len(sft_data_points)}/{len(wrong_problems)} "
                        f"({len(sft_data_points)/len(wrong_problems)*100:.1f}%)")
        
        return len(sft_data_points), len(wrong_problems)

