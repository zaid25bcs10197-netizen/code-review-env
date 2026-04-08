# Inference module
import os
import json
from openai import OpenAI
from env.environment import CodeReviewEnv

DEBUG = False

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN environment variable must be set")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

PROMPT = """
You are a code review assistant.

Analyze the given code and respond EXACTLY in the following format:

bug_present: yes/no
issue_type: syntax_error/runtime_error/logic_error/style_issue/none
action: approve/request_changes/reject
line_numbers: [list of integers]
"""


def parse_response(text):
    try:
        lines = text.strip().split("\n")
        result = {}

        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "line_numbers":
                result[key] = json.loads(value)
            else:
                result[key] = value

        # Validate required keys
        required = ["bug_present", "issue_type", "action", "line_numbers"]
        if not all(k in result for k in required):
            return None, "missing required fields"

        return result, None
    except Exception as e:
        return None, str(e)


def load_dataset():
    with open("dataset.json") as f:
        return json.load(f)


def run_task(task_name, dataset):
    env = CodeReviewEnv(dataset)

    done = False
    step = 0
    rewards = []
    last_error = None

    print(f"[START] task={task_name} env=code-review model={MODEL_NAME}")

    try:
        obs = env.reset()

        while not done:
            step += 1
            if DEBUG:
                print(f"DEBUG: Step {step}, Code: {obs.code[:50]}...")

            user_input = f"{PROMPT}\n\nCode:\n{obs.code}"

            fallback_action = {
                "bug_present": "yes",
                "issue_type": "logic_error",
                "action": "request_changes",
                "line_numbers": [1]
            }

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": user_input}],
                    temperature=0,
                    timeout=30
                )

                output = response.choices[0].message.content.strip()
                if DEBUG:
                    print(f"DEBUG: Model output: {output}")
                action, err = parse_response(output)

                if err:
                    action = fallback_action
                    last_error = f"parse_error_{err}"
                    if DEBUG:
                        print(f"DEBUG: Parse error: {err}")
                else:
                    last_error = None

            except Exception as e:
                action = fallback_action
                last_error = f"api_error_{str(e)}"
                if DEBUG:
                    print(f"DEBUG: API error: {e}")

            result = env.step(action)

            reward = result.reward
            done = result.done

            rewards.append(reward)

            action_str = json.dumps(action, separators=(',', ':')) if action else "null"
            error_str = last_error.replace(" ", "_") if last_error else "null"

            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}")

            obs = result.observation

    except Exception as e:
        if DEBUG:
            print(f"DEBUG: Execution error: {e}")
    finally:
        score = round(sum(rewards) / len(rewards), 2) if rewards else 0.0
        success = score >= 0.6
        steps = len(rewards)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")


def main():
    dataset = load_dataset()

    easy = [d for d in dataset if d["difficulty"] == "easy"]
    medium = [d for d in dataset if d["difficulty"] == "medium"]
    hard = [d for d in dataset if d["difficulty"] == "hard"]

    if easy:
        run_task("easy", easy)
    if medium:
        run_task("medium", medium)
    if hard:
        run_task("hard", hard)


if __name__ == "__main__":
    main()