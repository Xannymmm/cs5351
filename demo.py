#!/usr/bin/env python3
"""
demo.py — Teaching skeleton for HumanEvalFix agent (s01+s02+s06+s11+s15 according to learning claude code github repo)

Teaching patterns implemented:
  s01: Single agent loop — while True, stop when no tool_calls
  s02: TOOL_HANDLERS dispatch map (generate_fix, run_test, finalize)
  s06: Sub-agent fixer — separate LLM call for code repair
  s11: Retry with exponential backoff on API failures
  s15: Parallel entry workers via ThreadPoolExecutor

Core patterns:
  - FixerAgent class with per-entry state isolation
  - ThreadPoolExecutor concurrency with configurable max workers
  - Retry with exponential backoff (3 attempts)
  - Per-thread OpenAI client isolation (thread-safe)
  - Print/results locks to prevent interleaved output
  - Fixer→router communication via tool result context

Student Tasks to complete for the assignment:
  - Prompt engineering: improve FIXER_SYSTEM_PROMPT and SYSTEM_PROMPT
  - Thinking effort calibration: tune reasoning_effort per retry tier
  - Concurrency tuning: --parallel N to find optimal throughput
  - Token optimization: compare token budgets across prompt/effort configs
  - Tool dispatch: how tool schemas affect LLM behavior

Baseline performance (164 entries, 164 workers):
  | Metric             | demo (baseline)   | teacher's code    | Student target |
  |--------------------|-------------------|-------------------|----------------|
  | Pass rate          | 100%              | 100%              |  97%           |
  | Total tokens       | 389,524           | 280,638           | <600,000       |
  | Avg tok/entry      | 2,375             | 1,711             | <4,000         |
  | Wall clock         | 133.2s            | 51.4s             | <180s          |
  | Entry CPU sum      | 2,076.6s          | 779.3s            | <1,200s        |
  | Avg time/entry     | 12.7s             | 4.8s              | <8s            |
  | Retry entries      | 71                | 15                | no target      |

NOTE: Prompts and reasoning strategy are intentionally basic — students improve them in the assignment.
NOTE: agent.env is a file to contain the environmental variables for API endpoint, key, and model name. It is not included in the repo for security reasons.

agent.env example:
ACTIVE_AGENT_TIER="flash"
AZURE_FOUNDRY_BASE_URL="${CITYUCS_LLM_BASE_URL}"
AZURE_INFERENCE_CREDENTIAL="${CITYUCS_LLM_API_KEY}"
DEEPSEEK_V4_FLASH_DEPLOYMENT="CS5351/DeepSeek-V4-Flash"  


Main:
  python3 demo.py 10                        # first 10 entries
  python3 demo.py 10 --parallel 8           # 8 concurrent workers
  python3 demo.py Python/0 Python/1         # specific entries
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import random
from openai import OpenAI
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, env_path: str = "agent.env"):
        if importlib.util.find_spec("dotenv"):
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            self._load_dotenv_manual(env_path)
        self.base_url = os.environ.get("AZURE_FOUNDRY_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("AZURE_INFERENCE_CREDENTIAL", "")
        self.model_name = os.environ.get("DEEPSEEK_V4_FLASH_DEPLOYMENT", "")
        if not self.base_url or not self.api_key:
            raise RuntimeError(f"Missing env vars. Check {env_path}.")

    @staticmethod
    def _load_dotenv_manual(env_path: str) -> None:
        p = Path(env_path)
        if not p.exists():
            return
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("\"'")
                if key.startswith("export "):
                    key = key[7:]
                if val.startswith("${") and val.endswith("}"):
                    val = os.environ.get(val[2:-1], "")
                os.environ[key] = val


# ---------------------------------------------------------------------------
# HumanEvalFix Dataset
# ---------------------------------------------------------------------------

class HumanEvalFix:
    BENCHMARK_DIR = Path(__file__).parent / "HumanEvalFix"

    def __init__(self):
        if not self.BENCHMARK_DIR.is_dir():
            raise FileNotFoundError(f"Benchmark dir not found: {self.BENCHMARK_DIR}")

    def get_by_ids(self, entry_ids: list[str]) -> list[dict]:
        results, missing = [], []
        for eid in entry_ids:
            fname = eid.replace("/", "_") + ".json"
            fpath = self.BENCHMARK_DIR / fname
            if fpath.exists():
                results.append(json.loads(fpath.read_text()))
            else:
                missing.append(eid)
        if missing:
            print(f"  [WARN] Entries not found: {missing}", file=sys.stderr)
        return results

    def get_first_n(self, n: int) -> list[dict]:
        files = sorted(
            self.BENCHMARK_DIR.glob("Python_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        return [json.loads(f.read_text()) for f in files[:n]]

    @staticmethod
    def entry_key(entry: dict) -> str:
        return entry.get("task_id", "unknown")


# ---------------------------------------------------------------------------
# API Call Logger
# ---------------------------------------------------------------------------

class ApiCallLog:
    def __init__(self):
        self.calls: list[dict] = []

    def record(self, round_num: int, messages_sent: list, response, latency_s: float):
        if response is None:
            self.calls.append({
                "round": round_num,
                "latency_s": round(latency_s, 2),
                "messages_sent_summary": [{"note": "no response — max steps"}],
                "response_summary": {"content": "(no response)", "content_len": 0, "thinking_len": 0, "thinking": ""},
                "usage": None,
            })
            return

        choice = response.choices[0]
        msg = choice.message

        msg_summary = []
        for m in messages_sent:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else (getattr(m, "content", None) or "")
            tool_calls = m.get("tool_calls") if isinstance(m, dict) else (getattr(m, "tool_calls", None) or [])
            entry = {
                "role": role,
                "content_len": len(content or ""),
                "content_preview": (content or "")[:300],
            }
            if tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "?"),
                     "args_len": len(tc.function.arguments if hasattr(tc, "function") else json.dumps(tc.get("function", {})))}
                    for tc in (tool_calls if isinstance(tool_calls, list) else [])
                ]
            msg_summary.append(entry)

        reasoning_content = getattr(msg, "reasoning_content", None) or ""

        resp_summary = {
            "finish_reason": choice.finish_reason,
            "content_len": len(msg.content or ""),
            "content": (msg.content or "")[:2000],
            "reasoning_len": len(reasoning_content),
            "reasoning": reasoning_content[:5000],
            "tool_calls": [],
        }
        if msg.tool_calls:
            for tc in msg.tool_calls:
                resp_summary["tool_calls"].append({
                    "name": tc.function.name,
                    "args": tc.function.arguments,
                })

        usage = response.usage
        usage_dict = {}
        if usage:
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                reasoning_tok = getattr(details, "reasoning_tokens", None)
                if reasoning_tok is not None:
                    usage_dict["reasoning_tokens"] = reasoning_tok

        self.calls.append({
            "round": round_num,
            "latency_s": round(latency_s, 2),
            "messages_sent_summary": msg_summary,
            "response_summary": resp_summary,
            "usage": usage_dict if usage_dict else None,
        })

    def record_sub(self, label: str, response, latency_s: float):
        """Record a sub-agent LLM call (e.g. generate_fix)."""
        choice = response.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None) or ""
        usage = response.usage
        usage_dict = {}
        if usage:
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                rt = getattr(usage.completion_tokens_details, "reasoning_tokens", None)
                if rt is not None:
                    usage_dict["reasoning_tokens"] = rt

        self.calls.append({
            "round": label,
            "latency_s": round(latency_s, 2),
            "response_summary": {
                "finish_reason": choice.finish_reason,
                "content_len": len(msg.content or ""),
                "content": (msg.content or "")[:2000],
                "reasoning_len": len(reasoning),
                "reasoning": reasoning[:5000],
            },
            "usage": usage_dict if usage_dict else None,
        })

    def save(self, task_id: str, results_dir: Path):
        total_reasoning = sum(
            (c.get("usage", {}) or {}).get("reasoning_tokens", 0) or 0
            for c in self.calls
        )
        log = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "total_calls": len(self.calls),
            "total_tokens": sum((c.get("usage") or {}).get("total_tokens", 0) or 0 for c in self.calls),
            "total_reasoning_tokens": total_reasoning,
            "calls": self.calls,
        }
        path = results_dir / f"api_log_{task_id.replace('/', '_')}.json"
        path.write_text(json.dumps(log, indent=2))
        return path


# ---------------------------------------------------------------------------
# Tool: run_test
# ---------------------------------------------------------------------------

# Helper functions referenced by test harnesses but not part of the buggy code.
# Injected silently into the test script — the fixer LLM never sees them.
HUMANEVAL_HELPERS = {
    "encode_cyclic": """
def encode_cyclic(s: str):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
""",
    "encode_shift": """
def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])
""",
    "poly": """
def poly(xs: list, x: float):
    return sum([coeff * x ** i for i, coeff in enumerate(xs)])
""",
}

# STUDENT WORK: add descriptions and parameters to guide the router
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_fix",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_test",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Per-thread entry cache: keys are task_ids, no overlap between threads
_entry_cache: dict[str, dict] = {}
_entry_cache_lock = threading.Lock()


def _run_test(task_id: str, fixed_code: str) -> dict:
    """Execute the test harness, return {'passed': bool, 'output': str}."""
    with _entry_cache_lock:
        entry = _entry_cache.get(task_id)
    if not entry:
        return {"passed": False, "output": f"Unknown task_id: {task_id}"}

    test_code = entry["test"]
    entry_point = entry["entry_point"]

    code = fixed_code.strip()
    body = [l for l in code.splitlines() if not l.strip().startswith(("import ", "from "))]

    # Seek the first def/class so leading prose doesn't get wrapped into a SyntaxError.
    def_idx = next(
        (i for i, l in enumerate(body) if l.strip().startswith(("def ", "class "))),
        None,
    )
    if def_idx is not None:
        head = body[def_idx:]
        if head and head[0].strip().startswith("```"):
            head = head[1:]
        func_def = "\n".join(l for l in head if l.strip() != "```").strip()
    else:
        # Fallback: generic wrapper when no def is present.
        func_def = f"def {entry_point}(*args, **kwargs):\n"
        for line in body:
            if line.strip():
                indent = "    " if not line.startswith("    ") and not line.startswith("\t") else ""
                func_def += f"{indent}{line}\n"

    import_stmt = entry.get("import", "")
    extra_imports = []
    for line in fixed_code.strip().splitlines():
        ls = line.strip()
        if ls.startswith(("import ", "from ")):
            extra_imports.append(ls)

    lines = []
    if import_stmt:
        lines.append(import_stmt)
    elif extra_imports:
        for imp in extra_imports:
            lines.append(imp)
    lines.append(func_def)

    # Inject helper functions — skip comment-only occurrences
    for helper_name, helper_code in HUMANEVAL_HELPERS.items():
        code_only = "\n".join(l for l in test_code.splitlines() if not l.strip().startswith("#"))
        if helper_name in code_only:
            lines.append(helper_code)

    lines.append(test_code)
    full_source = "\n".join(lines)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"test_{entry_point}_", delete=False,
    )
    tmp_path = tmp.name
    try:
        tmp.write(full_source)
        tmp.close()
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10,
            env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")},
        )
        if result.returncode == 0:
            return {"passed": True, "output": result.stdout.strip()}
        err = result.stderr.strip() or result.stdout.strip()
        return {"passed": False, "output": err}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "TIMEOUT (>10s)"}
    except Exception as e:
        return {"passed": False, "output": f"EXCEPTION: {e}"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# STUDENT WORK: try different tool_choice values: "required", "auto", "none"
TOOL_CHOICE = os.environ.get("DEMO_TOOL_CHOICE", "required")

# --- STUDENT WORK AREA: prompts and reasoning strategy ---
# Modify these to improve agent performance.
SYSTEM_PROMPT = (
    "Start with generate_fix.\n"
    "After generate_fix: run_test.\n"
    "After test failure: generate_fix.\n"
    "After test pass: finalize."
)

FIXER_SYSTEM_PROMPT = (
    "You are a Python debugger. Fix the bug in the provided code.\n"
    "Return ONLY the complete function including its def signature, "
    "correctly indented, no prose, no markdown."
)


RETRY_MAX = 3
RETRY_BASE_DELAY = 1.0

def retry_with_backoff(fn, label="API call"):
    for attempt in range(RETRY_MAX):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRY_MAX - 1:
                raise
            delay = random.uniform(0.5, min(30.0, RETRY_BASE_DELAY * (2 ** attempt)))
            print(f"  [retry] {label} attempt {attempt+1}/{RETRY_MAX} failed: {e}. "
                  f"Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)


def agent_loop(messages: list, api_log: ApiCallLog, client, handlers: dict, model_name: str, max_steps: int = 10) -> tuple[int, int]:
    steps = 0

    while True:
        steps += 1
        if steps > max_steps:
            api_log.record(steps - 1, messages, None, 0)
            return steps, 0
        t0 = time.time()

        response = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice=TOOL_CHOICE,
                temperature=0.0,
                max_tokens=400,
                # STUDENT WORK: the router uses reasoning_effort "low" by default with
                # tool_choice="required". Try "auto" to see what happens with thinking.
                extra_body={"reasoning_effort": "low"},
            ),
            label=f"router step {steps}",
        )
        latency = time.time() - t0

        recorded = False
        try:
            choice = response.choices[0]
            msg = choice.message
            if msg is None:
                raise AttributeError("message is None")

            api_log.record(steps, messages, response, latency)
            recorded = True

            messages.append(msg)

            has_tc = msg.tool_calls is not None and len(msg.tool_calls) > 0
            if not has_tc or choice.finish_reason != "tool_calls":
                return steps, 0

            results = []
            should_finalize = False
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                handler = handlers.get(fn_name)
                if handler is None:
                    output = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    try:
                        args = json.loads(tc.function.arguments)
                        output = handler(**args)
                        if output == "__FINALIZE__":
                            should_finalize = True
                    except Exception as e:
                        output = json.dumps({"error": f"Tool {fn_name} failed: {e}"})
                results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })
            messages.extend(results)
            if should_finalize:
                return steps, 0
        except (IndexError, AttributeError) as e:
            print(f"  [router] step {steps} response processing failed: {e}", flush=True)
            if not recorded:
                api_log.record(steps, messages, response, 0)
            continue


class FixerAgent:
    """Per-entry state container for fix generation and test dispatch.

    State is isolated per instance — each entry gets its own FixerAgent.
    Handlers return JSON tool responses; agent_loop orchestrates the ReAct flow.

    STUDENT WORK: _handler_generate_fix contains the prompts and reasoning
    strategy. Modify FIXER_SYSTEM_PROMPT and reasoning_effort there.
    """

    AUTO_DEEP_AFTER = 2

    def __init__(self, entry: dict, client, model_name: str):
        self.task_id = HumanEvalFix.entry_key(entry)
        self.entry_data = entry
        self.client = client
        self.model_name = model_name
        self.raw_fix = ""
        self.failures: list[dict] = []
        self.test_fail_count = 0
        self._last_test_fix = ""

        # Populate the module-level entry cache (needed by _run_test)
        with _entry_cache_lock:
            _entry_cache[self.task_id] = entry

    def _handler_generate_fix(self, **kwargs) -> str:
        # STUDENT WORK: self.failures contains failure history from _handler_run_test.
        # You can access self.failures[-1]['error'] for the most recent failure.
        # Wire this into the fixer_message below to give the fixer context on retries.

        # STUDENT WORK: experiment with reasoning_effort values.
        # Valid: "low", "medium", "high". Use reasoning_effort "low" to minimize thinking.
        retries = len(self.failures)
        if retries == 0:
            extra = {"reasoning_effort": "low"}
        elif retries == 1:
            extra = {"reasoning_effort": "medium"}
        else:
            extra = {"reasoning_effort": "high"}

        # STUDENT WORK: compose the fixer message. Currently only includes buggy code.
        # Consider adding failure context from self.failures and thinking constraints.
        fixer_message = (
            f"### Buggy code\n```python\n{self.entry_data['buggy_solution']}\n```\n\n"
            f"### Signature\n{self.entry_data['signature']}\n\n"
            f"### Instruction\n{self.entry_data.get('instruction', '')}"
        )

        t0 = time.time()
        response = retry_with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": FIXER_SYSTEM_PROMPT},
                    {"role": "user", "content": fixer_message},
                ],
                temperature=0.0,
                **extra,
            ),
            label=f"fixer {self.task_id}",
        )
        latency = time.time() - t0
        self.api_log.record_sub(f"fix-{len(self.failures)+1}", response, latency)
        msg = response.choices[0].message
        # CityUCS gateway returns the answer in reasoning_content with content
        # empty; fall back to it when content is absent.
        raw = (msg.content or "").strip() or (getattr(msg, "reasoning_content", None) or "").strip()
        self.raw_fix = raw

        return json.dumps({"status": "fix generated"})

    def _handler_run_test(self, **kwargs) -> str:
        code = self.raw_fix
        if not code:
            return json.dumps({"passed": False, "output": "No fix generated yet. Call generate_fix first."})
        if code == self._last_test_fix:
            return json.dumps({"passed": False, "output": "duplicate — fix unchanged"})
        self._last_test_fix = code

        result = _run_test(self.task_id, code)
        if not result.get("passed"):
            self.test_fail_count += 1
            self.failures.append({
                "fix": (self.raw_fix or "")[:300],
                "error": result.get("output", "")[:500],
            })
            return json.dumps({"passed": False, "error": result.get("output", "")[:100]})
        self.test_fail_count = 0
        return json.dumps({"passed": True})

    def _handler_finalize(self) -> str:
        return "__FINALIZE__"

    def run(self) -> tuple:
        """Set up messages and run the agent loop. Returns (steps, tokens, messages, api_log)."""
        self.api_log = ApiCallLog()

        handlers = {
            "generate_fix": self._handler_generate_fix,
            "run_test": self._handler_run_test,
            "finalize": self._handler_finalize,
        }

        user_prompt = f"Task ID: {self.task_id}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        steps, _ = agent_loop(messages, self.api_log, self.client, handlers, model_name=self.model_name)
        # Token counting: agent_loop returns 0 for tokens (deferred).
        # Full tally (router + fixer calls) computed from api_log below.
        tokens = sum((c.get("usage") or {}).get("total_tokens", 0) or 0 for c in self.api_log.calls)

        return steps, tokens, messages, self.api_log


# ---------------------------------------------------------------------------
# Benchmark Results
# ---------------------------------------------------------------------------

class BenchmarkResults:
    RESULTS_DIR = Path(__file__).parent / "results"

    def __init__(self):
        self.RESULTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.RESULTS_DIR / f"benchmark_{ts}.md"
        self.json_path = self.RESULTS_DIR / f"benchmark_{ts}.json"

    def save(self, results: list[dict], wall_seconds: float = 0.0, metadata: dict = None):
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")
        errors = sum(1 for r in results if r["status"] == "error")
        total_tokens = sum(r.get("tokens", 0) for r in results)
        total_steps = sum(r.get("steps", 0) for r in results)
        total_time = sum(r.get("duration_s", 0) for r in results)

        lines = [
            f"# Benchmark Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total entries | {total} |",
            f"| Passed | {passed} |",
            f"| Failed | {failed} |",
            f"| Errors | {errors} |",
        ]
        if total > 0:
            lines.append(f"| Pass rate | {passed / total * 100:.1f}% |")
        lines += [
            f"| Total tokens | {total_tokens:,} |",
            f"| Total steps | {total_steps} |",
            f"| Wall clock | {wall_seconds:.1f}s |",
            f"| Entry sum | {total_time:.1f}s |",
        ]
        if total > 0:
            lines += [
                f"| Avg tokens/entry | {total_tokens // total:,} |",
                f"| Avg steps/entry | {total_steps / total:.1f} |",
                f"| Avg time/entry | {total_time / total:.1f}s |",
            ]

        lines += [
            "", "## Per-entry Results", "",
            "| Entry | Status | Steps | Tokens | Time (s) |",
            "|-------|--------|-------|--------|----------|",
        ]
        for r in results:
            lines.append(
                f"| {r['task_id']} | {r['status']} | {r.get('steps', '-')} | "
                f"{r.get('tokens', '-')} | {r.get('duration_s', '-')} |"
            )

        if metadata:
            meta_lines = ["## Metadata\n"]
            for k, v in sorted(metadata.items()):
                meta_lines.append(f"- **{k}**: {v}\n")
            meta_lines.append("\n")
            lines = meta_lines + lines
        output = "\n".join(lines)
        self.path.write_text(output)
        # Also write machine-parseable JSON
        json_output = {
            "metadata": metadata or {},
            "summary": {
                "total": total, "passed": passed, "failed": failed, "errors": errors,
                "total_tokens": total_tokens, "total_steps": total_steps,
                "wall_seconds": wall_seconds, "entry_sum_seconds": total_time,
            },
            "entries": results,
        }
        self.json_path.write_text(json.dumps(json_output, indent=2))
        print(f"\n  Results saved: {self.path}")

        print("\n" + "=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  Entries: {total}  |  Pass: {passed}  |  Fail: {failed}  |  Error: {errors}")
        if total > 0:
            print(f"  Pass rate: {passed / total * 100:.1f}%")
        print(f"  Tokens: {total_tokens:,}  |  Steps: {total_steps}  |  Wall: {wall_seconds:.1f}s  |  EntrySum: {total_time:.1f}s")
        if total > 0:
            print(f"  Avg/entry: ~{total_tokens // total:,} tok  |  {total_steps / total:.1f} steps  |  {total_time / total:.1f}s")
        print("=" * 60)

        # Print teacher's code comparison table for student reference
        COL = 11
        agent_tok  = f"{total_tokens:>{COL},}"
        agent_avg  = f"{total_tokens // total:>{COL},}" if total else "???".rjust(COL)
        agent_wall = f"{wall_seconds:.1f}s".rjust(COL)
        agent_time = f"{total_time / total:.1f}s".rjust(COL) if total else "???".rjust(COL)
        agent_ret  = f"{sum(1 for r in results if r.get('steps', 0) > 3)}".rjust(COL)

        TCOL = 17
        teach_tok  = "287,196".rjust(TCOL)
        teach_avg  = "1,740".rjust(TCOL)
        teach_wall = "57.0s".rjust(TCOL)
        teach_time = "3.2s".rjust(TCOL)
        teach_ret  = "15".rjust(TCOL)

        print("\n  Your baseline vs teacher's code (164 entries, 164 workers):")
        print("  ┌────────────────────┬─────────────┬───────────────────┐")
        print("  │ Metric             │ Your agent  │ Teacher's code    │")
        print("  ├────────────────────┼─────────────┼───────────────────┤")
        print(f"  │ Total tokens       │ {agent_tok} │ {teach_tok} │")
        print(f"  │ Avg tok/entry      │ {agent_avg} │ {teach_avg} │")
        print(f"  │ Wall clock         │ {agent_wall} │ {teach_wall} │")
        print(f"  │ Avg time/entry     │ {agent_time} │ {teach_time} │")
        print(f"  │ Retry entries      │ {agent_ret} │ {teach_ret} │")
        print("  └────────────────────┴─────────────┴───────────────────┘")
        print("\n  Close the gap: improve FIXER_SYSTEM_PROMPT, tune reasoning_effort,\n"
              "  add tool descriptions, and experiment with thinking modes.")
        print("=" * 60)
        return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HumanEvalFix benchmark agent (skeleton to implement your assignment)",
    )
    parser.add_argument(
        "args", nargs="*",
        help="Number of entries (e.g. 5) or specific entry IDs (e.g. Python/0 Python/22)",
    )
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="Max concurrent worker threads (default: number of entries)",
    )
    parsed = parser.parse_args()

    print(f"  Loading config...")
    config = Config()

    print(f"  Loading HumanEvalFix...")
    dataset = HumanEvalFix()

    raw_args = parsed.args
    if not raw_args:
        entries = dataset.get_first_n(1)
        print("  No args given. Default: 1 entry")
    elif len(raw_args) == 1 and raw_args[0].isdigit():
        n = int(raw_args[0])
        entries = dataset.get_first_n(n)
        print(f"  Args: count={n}")
    else:
        entries = dataset.get_by_ids(raw_args)
        print(f"  Args: {len(entries)} specific entries")

    if not entries:
        print("  No entries to process.")
        sys.exit(1)

    num_entries = len(entries)
    max_workers = max(1, parsed.parallel if parsed.parallel is not None else num_entries)
    print(f"  Running {num_entries} entries with max {max_workers} concurrent workers...")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Pre-allocate results list (preserve input order)
    all_results: list = [None] * num_entries
    results_lock = threading.Lock()
    print_lock = threading.Lock()

    processed = 0
    started = 0

    def worker(entry: dict, idx: int):
        """Process one entry in a worker thread."""
        nonlocal processed, started
        task_id = HumanEvalFix.entry_key(entry)

        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=60.0,
        )

        with print_lock:
            started += 1
            print(f"  [{started}/{num_entries}] {task_id}  ...", flush=True)

        t0 = time.time()
        try:
            agent = FixerAgent(entry, client, config.model_name)
            steps, tokens, msgs, api_log = agent.run()
            duration = round(time.time() - t0, 2)

            log_path = api_log.save(task_id, results_dir)

            # Determine pass/fail from last tool result
            passed = False
            test_output = ""
            for m in reversed(msgs):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                content = m.get("content") if isinstance(m, dict) else (getattr(m, "content", None) or "")
                if role == "tool" and isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "passed" in parsed:
                            passed = parsed["passed"]
                            test_output = parsed.get("output", "") or parsed.get("error", "")
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass

            status = "pass" if passed else "fail"
            result = {
                "task_id": task_id,
                "entry_point": entry["entry_point"],
                "status": status,
                "steps": steps,
                "tokens": tokens,
                "duration_s": duration,
                "test_output": test_output,
                "api_log": str(log_path),
            }

            with results_lock:
                all_results[idx] = result

            with print_lock:
                processed += 1
                print(f"  [{processed}/{num_entries}] {task_id}  {status} "
                      f"({steps} steps, ~{tokens} tok, {duration}s)", flush=True)

        except Exception as e:
            duration = round(time.time() - t0, 2)
            result = {
                "task_id": task_id,
                "entry_point": entry.get("entry_point", ""),
                "status": "error",
                "error": str(e),
                "steps": 0,
                "tokens": 0,
                "duration_s": duration,
            }
            with results_lock:
                all_results[idx] = result
            with print_lock:
                processed += 1
                print(f"  [{processed}/{num_entries}] {task_id}  ERROR: {e}", flush=True)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    wall_t0 = time.time()
    for i, entry in enumerate(entries):
        futures.append(executor.submit(worker, entry, i))
    executor.shutdown(wait=True)

    for i, fut in enumerate(futures):
        if all_results[i] is None:
            try:
                fut.result()
            except Exception as exc:
                with print_lock:
                    print(f"  [{processed}/{num_entries}] entry idx={i}  UNHANDLED: {exc}")
    wall_elapsed = time.time() - wall_t0

    # Verify all results collected
    missing = [i for i, r in enumerate(all_results) if r is None]
    if missing:
        print(f"\n  WARNING: {len(missing)}/{num_entries} entries produced no results (indices: {missing[:10]}{'...' if len(missing) > 10 else ''})")
        all_results = [r for r in all_results if r is not None]
        if not all_results:
            print("  ERROR: No results collected. Exiting.")
            sys.exit(1)

    bm = BenchmarkResults()
    metadata = {
        "model": config.model_name,
        "timestamp": datetime.now().isoformat(),
    }
    bm.save(all_results, wall_elapsed, metadata=metadata)


if __name__ == "__main__":
    main()
