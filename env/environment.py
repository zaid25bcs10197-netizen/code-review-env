# Environment configuration module
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Observation(BaseModel):
    code: str
    step: int
    previous_action: Optional[Dict[str, Any]] = None


class Action(BaseModel):
    bug_present: str
    issue_type: str
    action: str
    line_numbers: List[int]


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: Dict[str, Any]


class CodeReviewEnv:
    def __init__(self, dataset: List[Dict]):
        self.dataset = dataset
        self.index = 0
        self.current_task = None
        self.step_count = 0
        self.max_steps = 2
        self.last_action = None

    def reset(self) -> Observation:
        self.current_task = self.dataset[self.index]
        self.step_count = 0
        self.last_action = None

        return Observation(
            code=self.current_task["code"],
            step=self.step_count,
            previous_action=None
        )

    def state(self) -> Observation:
        return Observation(
            code=self.current_task["code"],
            step=self.step_count,
            previous_action=self.last_action
        )

    def step(self, action_dict: Dict) -> StepResult:
        self.step_count += 1

        valid, error = self._validate_action(action_dict)
        if not valid:
            return StepResult(
                observation=None,
                reward=-0.1,  # Penalize invalid actions
                done=True,
                info={"error": error}
            )

        self.last_action = action_dict

        reward, breakdown = self._compute_reward(action_dict, self.current_task)

        difficulty = self.current_task.get("difficulty", "easy")
        scale = {"easy": 1.0, "medium": 1.2, "hard": 1.5}
        reward *= scale.get(difficulty, 1.0)

        reward = max(-1.0, min(1.0, reward))

        done = False
        if self.step_count >= self.max_steps:
            done = True
            self.index = (self.index + 1) % len(self.dataset)

        next_obs = None if done else Observation(
            code=self.current_task["code"],
            step=self.step_count,
            previous_action=self.last_action
        )

        return StepResult(
            observation=next_obs,
            reward=round(reward, 4),
            done=done,
            info={"breakdown": breakdown, "difficulty": difficulty}
        )

    def _validate_action(self, action: Dict):
        required_keys = ["bug_present", "issue_type", "action", "line_numbers"]

        for key in required_keys:
            if key not in action:
                return False, f"missing_{key}"

        if action["bug_present"] not in ["yes", "no"]:
            return False, "invalid_bug_present"

        if action["issue_type"] not in [
            "syntax_error",
            "runtime_error",
            "logic_error",
            "style_issue",
            "none"
        ]:
            return False, "invalid_issue_type"

        if action["action"] not in [
            "approve",
            "request_changes",
            "reject"
        ]:
            return False, "invalid_action"

        if not isinstance(action["line_numbers"], list):
            return False, "invalid_line_numbers"

        if not all(isinstance(x, int) and x >= 1 for x in action["line_numbers"]):
            return False, "invalid_line_values"

        return True, None

    def _compute_reward(self, pred: Dict, gt: Dict):
        score = 0.0
        breakdown = {}

        if pred["bug_present"] == gt["bug_present"]:
            score += 0.25
            breakdown["bug_present"] = 0.25
        else:
            breakdown["bug_present"] = 0.0
            if pred["bug_present"] == "yes" and gt["bug_present"] == "no":
                score -= 0.2
                breakdown["false_positive_penalty"] = -0.2

        if pred["issue_type"] == gt["issue_type"]:
            score += 0.25
            breakdown["issue_type"] = 0.25
        else:
            breakdown["issue_type"] = 0.0

        if pred["action"] == gt["action"]:
            score += 0.15
            breakdown["action"] = 0.15
        else:
            breakdown["action"] = 0.0

        pred_lines = set(pred["line_numbers"])
        gt_lines = set(gt["line_numbers"])

        if len(pred_lines) == 0 and len(gt_lines) == 0:
            line_score = 1.0
        else:
            intersection = len(pred_lines & gt_lines)
            precision = intersection / len(pred_lines) if pred_lines else 0
            recall = intersection / len(gt_lines) if gt_lines else 0

            if precision + recall == 0:
                line_score = 0
            else:
                line_score = 2 * precision * recall / (precision + recall)

        line_reward = 0.15 * line_score
        score += line_reward
        breakdown["line_numbers"] = line_reward

        if len(pred_lines) > len(gt_lines) * 2:
            score -= 0.1
            breakdown["line_spam_penalty"] = -0.1

        return score, breakdown