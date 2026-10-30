from __future__ import annotations

import hashlib
import io
import json
import random
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Prompt Optimizer Agent",
    page_icon="🧪",
    layout="wide",
)

SCENARIOS = {
    "support_ticket": {
        "name": "Support-ticket classification",
        "task": "classification",
        "baseline_score": 0.52,
        "target_score": 0.90,
        "test_cases": 60,
        "rubric": "Exact category match, stable JSON output, no hallucinated labels.",
        "strategies": [
            "baseline",
            "constraints",
            "definitions",
            "few-shot",
            "self-check",
            "optimized",
        ],
        "base_prompt": (
            "Classify the support ticket into exactly one of: billing, bug, "
            "feature_request, account. Ticket: {ticket}"
        ),
        "variants": [
            (
                "Classify the support ticket into exactly one of: billing, bug, "
                "feature_request, account. Ticket: {ticket}"
            ),
            (
                "Classify the support ticket into exactly one of: billing, bug, "
                "feature_request, account. Return only the category name and no "
                "explanation. Ticket: {ticket}"
            ),
            (
                "You are a support-triage agent. Classify the ticket into exactly "
                "one of: billing, bug, feature_request, account. Definitions: "
                "billing = payments, invoices, refunds; bug = errors, crashes, "
                "failures; feature_request = ideas or enhancements; account = "
                "login, profile, permissions. Return JSON: "
                '{"category": "one_label"}. Ticket: {ticket}'
            ),
            (
                "You are a support-triage agent. Classify into billing, bug, "
                "feature_request, or account. Examples: "
                'Ticket: I was charged twice this month. Output: {"category": "billing"}. '
                'Ticket: The export button throws an error. Output: {"category": "bug"}. '
                "Now classify the new ticket and return JSON only. Ticket: {ticket}"
            ),
            (
                "You are a production support-triage agent. First check for evidence "
                "of payments, errors, feature asks, or account access. If money is "
                "involved, prefer billing; otherwise prefer bug when failures are "
                'explicit. Return only JSON: {"category": "one_label"}. Ticket: {ticket}'
            ),
            (
                "Production-optimized triage: map the ticket to one lowercase "
                "category among billing, bug, feature_request, account. Preserve "
                "stability, do not add keys, do not add explanations. Output JSON: "
                '{"category": "one_label"}. Ticket: {ticket}'
            ),
        ],
    },
    "math_word_problem": {
        "name": "Math word-problem solving",
        "task": "math",
        "baseline_score": 0.44,
        "target_score": 0.86,
        "test_cases": 45,
        "rubric": "Correct final numeric answer, sound arithmetic, concise reasoning.",
        "strategies": [
            "baseline",
            "structured-plan",
            "step-verify",
            "few-shot",
            "self-check",
            "optimized",
        ],
        "base_prompt": "Solve the math word problem. Problem: {problem}",
        "variants": [
            "Solve the math word problem. Problem: {problem}",
            (
                "You are a careful math solver. Identify known values, choose the "
                "operation, compute, and end with Final answer: <number>. "
                "Problem: {problem}"
            ),
            (
                "Solve step by step, but keep the final response concise. Verify "
                "arithmetic and units. End with Final answer: <number>. "
                "Problem: {problem}"
            ),
            (
                "Example: Problem: A store has 3 boxes of 12 pencils. How many "
                "pencils? Reasoning: 3 * 12 = 36. Final answer: 36. Now solve the "
                "new problem and end with Final answer: <number>. Problem: {problem}"
            ),
            (
                "Before answering, check whether the question asks for a total, "
                "difference, rate, or proportion. Recompute if the result seems "
                "implausible. End with Final answer: <number>. Problem: {problem}"
            ),
            (
                "Production-optimized solver: parse quantities, choose the minimal "
                "arithmetic plan, verify units, and return only the final line "
                "Final answer: <number>. Problem: {problem}"
            ),
        ],
    },
    "summarization": {
        "name": "Rubric-based summarization",
        "task": "summarization",
        "baseline_score": 0.58,
        "target_score": 0.84,
        "test_cases": 50,
        "rubric": "Faithfulness, coverage of key entities, brevity, no hallucination.",
        "strategies": [
            "baseline",
            "faithfulness",
            "rubric-aware",
            "style-example",
            "self-check",
            "optimized",
        ],
        "base_prompt": "Summarize the article in three sentences. Article: {article}",
        "variants": [
            "Summarize the article in three sentences. Article: {article}",
            (
                "Summarize the article faithfully in three sentences. Do not add "
                "facts. Article: {article}"
            ),
            (
                "Summarize with a rubric: faithfulness, coverage of key entities, "
                "and brevity. Use three plain sentences. Article: {article}"
            ),
            (
                "Summarize the article in three sentences. Style example: A new "
                "policy was announced. It affects existing users. The rollout begins "
                "next quarter. Now summarize faithfully. Article: {article}"
            ),
            (
                "Before finalizing, verify every sentence is directly supported by "
                "the article. Remove speculation. Use three sentences. Article: {article}"
            ),
            (
                "Production-optimized summary: extract the main event, impact, and "
                "next step in three faithful sentences. No new facts. Article: {article}"
            ),
        ],
    },
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def get_scenario(scenario_id: str | None = None) -> dict:
    sid = scenario_id or st.session_state.get("scenario_id", "support_ticket")
    if sid not in SCENARIOS:
        sid = "support_ticket"
    return SCENARIOS[sid]


def add_event(
    node: str,
    message: str,
    evidence_id: str | None = None,
    tokens: int = 0,
    cost: float = 0.0,
    level: str = "info",
) -> None:
    if "events" not in st.session_state:
        st.session_state.events = []

    st.session_state.events.append(
        {
            "ts": utc_now(),
            "node": node,
            "message": message,
            "evidence_id": evidence_id or "-",
            "tokens": int(tokens),
            "cost": round(float(cost), 6),
            "level": level,
        }
    )


def reset_simulation(scenario_id: str | None = None) -> None:
    sid = scenario_id or st.session_state.get("scenario_id", "support_ticket")
    if sid not in SCENARIOS:
        sid = "support_ticket"

    scen = SCENARIOS[sid]

    st.session_state.update(
        {
            "initialized": True,
            "run_id": make_run_id(),
            "scenario_id": sid,
            "status": "running",
            "phase": "ingest",
            "generation": 0,
            "max_generations": 5,
            "waiting_human": False,
            "wait_ticks": 0,
            "events": [],
            "generations": [],
            "current_candidate": scen["base_prompt"],
            "best_prompt": scen["base_prompt"],
            "validation_score": scen["baseline_score"],
            "test_score": None,
            "overfit_gap": None,
            "token_cost_total": 0.0,
            "tokens_total": 0,
            "latency_total_ms": 0,
            "degraded": False,
        }
    )

    add_event(
        "bootstrap",
        "Loaded bundled demo data and started deterministic A/B simulation.",
        "E-000",
    )


def init_state() -> None:
    if not st.session_state.get("initialized", False):
        reset_simulation("support_ticket")


def get_candidate_prompt(scenario_id: str, generation: int) -> str:
    variants = SCENARIOS[scenario_id]["variants"]
    return variants[max(0, min(generation, len(variants) - 1))]


def simulate_evaluation(
    scenario_id: str,
    generation: int,
    prompt: str,
    split: str = "validation",
) -> dict:
    scen = SCENARIOS[scenario_id]
    strictness = float(st.session_state.get("strictness", 0.7))
    dry_run = bool(st.session_state.get("dry_run", False))

    seed_key = (
        f"{scenario_id}|{split}|{generation}|{prompt[:90]}|"
        f"{strictness:.2f}|{dry_run}"
    )
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    base = scen["baseline_score"]
    target = scen["target_score"]
    max_g = float(st.session_state.get("max_generations", 5))
    progress = max(0.0, min(1.0, generation / max_g))

    score = base + (target - base) * progress
    score += rng.uniform(-0.022, 0.028)
    score -= strictness * 0.015

    if split == "test":
        score -= rng.uniform(0.004, 0.035)

    tokens = int(720 + generation * 165 + rng.randint(0, 240))

    if dry_run:
        cost = 0.0
    else:
        cost = round(0.00012 * tokens + 0.0016 + generation * 0.00035, 6)

    latency = int(380 + generation * 85 + rng.randint(0, 320))
    if dry_run:
        latency = int(latency * 0.6)

    return {
        "score": clamp(score),
        "tokens": tokens,
        "cost": cost,
        "latency_ms": latency,
    }


def check_budget() -> None:
    s = st.session_state
    budget_cap = float(st.session_state.get("budget_cap", 2.0))

    if s.token_cost_total > budget_cap and s.status != "degraded":
        s.degraded = True
        s.status = "degraded"
        s.phase = "complete"
        s.waiting_human = False

        if s.test_score is None:
            s.test_score = round(max(0.0, s.validation_score - 0.03), 4)
            s.overfit_gap = 0.03

        add_event(
            "guardrail",
            f"Budget cap ${budget_cap:.2f} reached. Returning degraded result and stopping the loop.",
            "G-BUDGET",
            level="warning",
        )


def record_generation(generation: int) -> None:
    s = st.session_state
    sid = s.scenario_id
    scen = SCENARIOS[sid]

    prompt = get_candidate_prompt(sid, generation)
    result = simulate_evaluation(sid, generation, prompt, "validation")
    strategy = scen["strategies"][min(generation, len(scen["strategies"]) - 1)]

    record = {
        "generation": generation,
        "strategy": strategy,
        "prompt": prompt,
        "mean_score": round(result["score"], 4),
        "cost": result["cost"],
        "latency_ms": result["latency_ms"],
        "tokens": result["tokens"],
        "split": "validation",
    }

    s.generations.append(record)
    s.current_candidate = prompt
    s.token_cost_total += record["cost"]
    s.tokens_total += record["tokens"]
    s.latency_total_ms += record["latency_ms"]

    if record["mean_score"] >= s.validation_score:
        s.validation_score = record["mean_score"]
        s.best_prompt = prompt

    add_event(
        "evaluate",
        f"Generation {generation} validation score {record['mean_score']:.3f} using {strategy}.",
        f"E-G{generation}",
        record["tokens"],
        record["cost"],
    )

    check_budget()


def approve_gate(note: str = "approved") -> None:
    s = st.session_state
    if not s.waiting_human:
        return

    s.waiting_human = False
    s.wait_ticks = 0
    s.generation += 1
    s.phase = "generate"

    add_event(
        "human_gate",
        f"Human gate approved continuation. {note}",
        "H-APPROVE",
    )


def reject_gate() -> None:
    s = st.session_state
    if not s.waiting_human:
        return

    s.waiting_human = False
    s.wait_ticks = 0
    s.phase = "test"

    add_event(
        "human_gate",
        "Human gate rejected further variants. Finalizing best-so-far prompt.",
        "H-REJECT",
    )


def finalize_test() -> None:
    s = st.session_state

    if s.test_score is not None:
        return

    prompt = s.best_prompt
    result = simulate_evaluation(
        s.scenario_id,
        s.max_generations,
        prompt,
        split="test",
    )

    s.token_cost_total += result["cost"]
    s.tokens_total += result["tokens"]
    s.latency_total_ms += result["latency_ms"]

    s.test_score = round(result["score"], 4)
    s.overfit_gap = round(s.validation_score - s.test_score, 4)
    s.phase = "complete"
    s.status = "degraded" if s.degraded else "complete"

    add_event(
        "final_test",
        f"Test set evaluated exactly once. Test score {s.test_score:.3f}.",
        "E-TEST",
        result["tokens"],
        result["cost"],
    )

    check_budget()


def maybe_advance() -> None:
    s = st.session_state

    if s.status != "running":
        return

    if s.waiting_human:
        if bool(st.session_state.get("auto_play", True)):
            s.wait_ticks += 1
            if bool(st.session_state.get("auto_approve", True)) and s.wait_ticks >= 4:
                approve_gate("Auto-approved because auto-approve is enabled.")
        return

    if not bool(st.session_state.get("auto_play", True)):
        return

    dry_run = bool(st.session_state.get("dry_run", False))

    if s.phase == "ingest":
        add_event(
            "ingest",
            "Loaded prompt, bundled test cases, and evaluator rubric.",
            "E-001",
        )
        s.phase = "parse"

    elif s.phase == "parse":
        add_event(
            "perception",
            "Parsed task type, output contract, and guardrails.",
            "E-002",
        )
        s.phase = "baseline"

    elif s.phase == "baseline":
        record_generation(0)
        if s.status == "running":
            s.generation = 1
            s.phase = "generate"

    elif s.phase == "generate":
        prompt = get_candidate_prompt(s.scenario_id, s.generation)
        s.current_candidate = prompt
        strategy = SCENARIOS[s.scenario_id]["strategies"][
            min(s.generation, len(SCENARIOS[s.scenario_id]["strategies"]) - 1)
        ]

        add_event(
            "generate",
            f"Generated variant G{s.generation} using strategy '{strategy}'.",
            f"E-GEN{s.generation}",
            180,
            0.0 if dry_run else 0.0002,
        )
        s.phase = "evaluate"

    elif s.phase == "evaluate":
        record_generation(s.generation)
        if s.status == "running":
            s.phase = "select"

    elif s.phase == "select":
        if s.generations:
            best = max(s.generations, key=lambda x: x["mean_score"])
            if best["mean_score"] >= s.validation_score:
                s.validation_score = best["mean_score"]
                s.best_prompt = best["prompt"]

        add_event(
            "select",
            f"Selected top candidate through generation {s.generation}.",
            "E-SELECT",
        )

        if s.status != "running":
            return

        if s.generation == 3:
            s.phase = "human_gate"
            s.waiting_human = True
            s.wait_ticks = 0
            add_event(
                "human_gate",
                "Generation 3 candidate requires human approval before continuing.",
                "H-GATE",
            )
        elif s.generation >= s.max_generations:
            s.phase = "test"
        else:
            s.generation += 1
            s.phase = "generate"

    elif s.phase == "human_gate":
        s.waiting_human = True

    elif s.phase == "test":
        finalize_test()


def build_final_output() -> dict:
    s = st.session_state
    scen = get_scenario()

    generations = []
    for g in s.generations:
        generations.append(
            {
                "prompt": g["prompt"],
                "mean_score": round(float(g["mean_score"]), 4),
                "cost": round(float(g["cost"]), 6),
                "latency": int(g["latency_ms"]),
            }
        )

    test_score = s.test_score if s.test_score is not None else 0.0
    overfit_gap = (
        s.overfit_gap
        if s.overfit_gap is not None
        else round(s.validation_score - test_score, 4)
    )

    return {
        "run_id": s.run_id,
        "base_prompt": scen["base_prompt"],
        "generations": generations,
        "best_prompt": s.best_prompt,
        "validation_score": round(float(s.validation_score), 4),
        "test_score": round(float(test_score), 4),
        "overfit_gap": round(float(overfit_gap), 4),
        "token_cost_total": round(float(s.token_cost_total), 6),
    }


def validate_output(payload: dict) -> list[str]:
    errors: list[str] = []

    required_fields = [
        "run_id",
        "base_prompt",
        "generations",
        "best_prompt",
        "validation_score",
        "test_score",
        "overfit_gap",
        "token_cost_total",
    ]

    for field in required_fields:
        if field not in payload:
            errors.append(f"Missing required field: {field}")

    if errors:
        return errors

    for field in ["validation_score", "test_score"]:
        value = payload[field]
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            errors.append(f"{field} must be a number between 0 and 1.")

    if not isinstance(payload["generations"], list):
        errors.append("generations must be a list.")

    if payload["token_cost_total"] < 0:
        errors.append("token_cost_total must be >= 0.")

    if not isinstance(payload["overfit_gap"], (int, float)) or not (
        -1.0 <= float(payload["overfit_gap"]) <= 1.0
    ):
        errors.append("overfit_gap must be between -1 and 1.")

    for i, gen in enumerate(payload.get("generations", [])):
        if not isinstance(gen.get("prompt"), str) or not gen.get("prompt").strip():
            errors.append(f"generations[{i}].prompt must be a non-empty string.")

        mean_score = gen.get("mean_score")
        if not isinstance(mean_score, (int, float)) or not (0.0 <= float(mean_score) <= 1.0):
            errors.append(f"generations[{i}].mean_score must be between 0 and 1.")

        cost = gen.get("cost")
        if not isinstance(cost, (int, float)) or float(cost) < 0:
            errors.append(f"generations[{i}].cost must be >= 0.")

        latency = gen.get("latency")
        if not isinstance(latency, (int, float)) or float(latency) < 0:
            errors.append(f"generations[{i}].latency must be >= 0.")

    return errors


@st.cache_data(show_spinner=False)
def make_demo_video() -> bytes | None:
    try:
        frames = []
        width, height = 900, 480
        scores = [0.52, 0.59, 0.66, 0.72, 0.79, 0.84]

        for i in range(20):
            progress = i / 19.0
            gen = min(int(progress * 5), 5)

            img = Image.new("RGB", (width, height), (12, 16, 28))
            d = ImageDraw.Draw(img)

            d.rectangle([0, 0, width, 56], fill=(20, 28, 48))
            d.text((16, 16), "Prompt Optimizer Agent - live demo", fill=(240, 244, 255))
            d.text(
                (16, 66),
                "Scenario: Support-ticket classification",
                fill=(180, 190, 220),
            )
            d.text((16, 86), f"Generation {gen}/5", fill=(180, 190, 220))

            d.rectangle([60, 140, 840, 380], outline=(70, 80, 110))

            for j, score in enumerate(scores[: gen + 1]):
                bar_h = int(score * 220)
                x0 = 90 + j * 120
                x1 = x0 + 70
                y0 = 360 - bar_h
                y1 = 360
                d.rectangle([x0, y0, x1, y1], fill=(0, 190, 170))
                d.text((x0, y0 - 18), f"{score:.2f}", fill=(220, 240, 240))

            d.ellipse([820, 80, 840, 100], fill=(0, 220, 120))
            d.text(
                (16, 400),
                "A/B evaluation loop: validation scores improving",
                fill=(160, 170, 200),
            )
            d.text(
                (16, 420),
                f"Best prompt score: {max(scores[:gen + 1]):.2f}",
                fill=(240, 240, 255),
            )

            frames.append(img)

        bio = io.BytesIO()
        frames[0].save(
            bio,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=220,
            loop=0,
        )
        return bio.getvalue()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def make_run_video(run_id: str, scenario_name: str, scores: tuple) -> bytes | None:
    if not scores:
        return None

    try:
        frames = []
        width, height = 860, 420
        n_frames = 18

        for i in range(n_frames):
            progress = i / (n_frames - 1)
            visible = max(1, int(progress * len(scores)))

            img = Image.new("RGB", (width, height), (11, 15, 25))
            d = ImageDraw.Draw(img)

            d.rectangle([0, 0, width, 72], fill=(18, 26, 44))
            d.text((16, 12), "Prompt Optimizer Agent - run replay", fill=(240, 244, 255))
            d.text((16, 34), scenario_name, fill=(180, 190, 220))
            d.text((16, 52), f"Run: {run_id}", fill=(140, 150, 180))

            d.rectangle([50, 100, 810, 340], outline=(70, 80, 110))

            step = 740 / max(1, len(scores) - 1) if len(scores) > 1 else 740
            prev_x = prev_y = None

            for idx, score in enumerate(scores[:visible]):
                x = int(70 + idx * step)
                y = int(320 - float(score) * 200)

                if prev_x is not None and prev_y is not None:
                    d.line([prev_x, prev_y, x, y], fill=(0, 190, 170), width=3)

                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 220, 190))
                prev_x, prev_y = x, y

            best = max(scores[:visible])
            d.text((16, 360), f"Best score so far: {best:.3f}", fill=(240, 240, 255))
            d.text(
                (16, 380),
                "Test set is touched exactly once at the end.",
                fill=(160, 170, 200),
            )

            frames.append(img)

        bio = io.BytesIO()
        frames[0].save(
            bio,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        return bio.getvalue()
    except Exception:
        return None


init_state()

st.sidebar.title("Prompt Optimizer Controls")
st.sidebar.caption(
    "No API key is required. The app ships with realistic demo data and a "
    "deterministic simulator."
)

scenario_ids = list(SCENARIOS.keys())
current_idx = scenario_ids.index(st.session_state.get("scenario_id", "support_ticket"))

selected_scenario = st.sidebar.selectbox(
    "Sample scenario",
    scenario_ids,
    index=current_idx,
    format_func=lambda x: SCENARIOS[x]["name"],
)

st.sidebar.slider(
    "Strictness",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    key="strictness",
)

st.sidebar.number_input(
    "Budget cap (USD)",
    min_value=0.05,
    value=2.0,
    step=0.05,
    key="budget_cap",
)

st.sidebar.checkbox("Auto-play simulation", value=True, key="auto_play")
st.sidebar.checkbox("Auto-approve human gate", value=True, key="auto_approve")
st.sidebar.checkbox("Dry run (no simulated spend)", value=False, key="dry_run")

speed_label = st.sidebar.selectbox(
    "Simulation speed",
    ["Fast", "Normal", "Slow"],
    index=1,
)

uploaded = st.sidebar.file_uploader("Optional prompt/test cases JSON", type=["json"])

if uploaded is not None:
    try:
        payload = json.loads(uploaded.getvalue().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Root must be a JSON object.")
        st.sidebar.success(
            "Uploaded JSON parsed successfully. The demo continues with bundled test cases."
        )
    except Exception as exc:
        st.sidebar.error(
            f"Invalid upload: {exc}. Provide a JSON object such as "
            "{{'prompt': '...', 'test_cases': [...]}}."
        )

if st.sidebar.button("Restart demo", type="primary", use_container_width=True):
    reset_simulation(selected_scenario)
    st.rerun()

if selected_scenario != st.session_state.get("scenario_id"):
    reset_simulation(selected_scenario)
    st.rerun()

st.title("Prompt Optimizer Agent")
st.caption(
    "Automatic A/B prompt evaluation loop with observable, safe, human-gated demo simulation."
)

tab_live, tab_results, tab_audit, tab_about = st.tabs(
    ["Live Run", "Results", "Audit Log", "About"]
)

with tab_live:
    live_status = st.empty()
    live_progress = st.empty()
    live_kpis = st.empty()
    live_chart = st.empty()
    live_table = st.empty()
    live_human = st.empty()
    live_events = st.empty()
    live_video = st.empty()

    with st.expander("Automatically generated live demo recording", expanded=True):
        demo_video = make_demo_video()
        if demo_video:
            st.image(
                demo_video,
                caption="Generated immediately on startup (animated demo video).",
                use_container_width=True,
            )
            st.download_button(
                "Download live demo video (GIF)",
                data=demo_video,
                file_name="prompt_optimizer_live_demo.gif",
                mime="image/gif",
                key="startup_demo_video",
            )
        else:
            st.info("Video generation is unavailable in this environment.")

with tab_results:
    results_summary = st.empty()
    results_table = st.empty()
    results_prompt = st.empty()
    results_video = st.empty()
    results_json = st.empty()
    download_area = st.empty()

with tab_audit:
    st.text_input("Filter events", value="", key="audit_filter")
    audit_table = st.empty()

with tab_about:
    st.markdown(
        """
        ### What this demo does

        This app simulates the production design of a **Prompt Optimizer Agent**:

        - Ingests a prompt and a bundled test-case scenario.
        - Generates prompt variants.
        - Runs an automatic A/B evaluation loop on validation data.
        - Uses a human gate before continuing beyond generation 3.
        - Touches the test set exactly once at the end.
        - Produces a final result object with best prompt, scores, and cost.

        ### Final output schema

        ```json
        {
          "run_id": "string",
          "base_prompt": "string",
          "generations": [
            {
              "prompt": "string",
              "mean_score": 0.0,
              "cost": 0.0,
              "latency": 0
            }
          ],
          "best_prompt": "string",
          "validation_score": 0.0,
          "test_score": 0.0,
          "overfit_gap": 0.0,
          "token_cost_total": 0.0
        }
        ```

        ### Validation rules

        - All required fields must be present.
        - `validation_score` and `test_score` must be between 0 and 1.
        - `token_cost_total` must be non-negative.
        - Each generation must contain a non-empty prompt.
        - Each generation score, cost, and latency must be valid non-negative numbers.

        ### Guardrails

        - The test set is touched exactly once at the end.
        - A budget cap can stop the loop and return a degraded result.
        - The human gate blocks further optimization until approved or rejected.
        - All events are written to the audit log.
        - No API key is required for this local demo simulation.

        ### What the AI is not allowed to do in this demo

        - It may not call external paid APIs.
        - It may not peek at the held-out test set during validation loops.
        - It may not exceed the configured budget cap without flagging degraded mode.
        - It may not bypass the human gate.
        """
    )


def render_live() -> None:
    s = st.session_state
    scen = get_scenario()

    if s.status == "complete":
        live_status.success(
            f"{scen['name']} completed. Phase `{s.phase}`. Run `{s.run_id}`."
        )
    elif s.status == "degraded":
        live_status.warning(
            f"{scen['name']} finished in degraded mode. Run `{s.run_id}`."
        )
    elif s.waiting_human:
        live_status.warning(
            f"{scen['name']} is waiting at the human gate. Approve or reject to continue."
        )
    else:
        live_status.info(
            f"{scen['name']} is running. Phase `{s.phase}` | "
            f"Generation {s.generation}/{s.max_generations} | Run `{s.run_id}`."
        )

    phase_order = [
        "ingest",
        "parse",
        "baseline",
        "generate",
        "evaluate",
        "select",
        "human_gate",
        "test",
        "complete",
    ]

    if s.phase in phase_order:
        progress = phase_order.index(s.phase) / max(1, len(phase_order) - 1)
    else:
        progress = 0.0

    if s.waiting_human:
        progress = max(progress, 0.58)

    if s.phase == "complete":
        progress = 1.0

    progress = min(
        1.0, progress + (s.generation / max(1, s.max_generations)) * 0.05
    )

    live_progress.progress(progress, text=f"Phase: {s.phase.replace('_', ' ')}")

    generations = s.generations

    if generations:
        best_val = max(g["mean_score"] for g in generations)
        baseline = generations[0]["mean_score"]
        improvement = best_val - baseline
    else:
        best_val = scen["baseline_score"]
        baseline = scen["baseline_score"]
        improvement = 0.0

    overfit = s.overfit_gap
    if overfit is None:
        risk = "Low"
    elif overfit < 0.05:
        risk = "Low"
    elif overfit < 0.08:
        risk = "Medium"
    else:
        risk = "High"

    with live_kpis.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Best validation score",
            f"{best_val:.3f}",
            f"{improvement:+.3f} vs baseline",
        )
        c2.metric("Generation", f"{s.generation}/{s.max_generations}")
        c3.metric("Simulated cost", f"${s.token_cost_total:.4f}")
        c4.metric("Overfit risk", risk)

    if generations:
        df = pd.DataFrame(generations)
        live_chart.line_chart(df.set_index("generation")["mean_score"])
        live_table.dataframe(
            df[
                [
                    "generation",
                    "strategy",
                    "mean_score",
                    "cost",
                    "latency_ms",
                ]
            ],
            use_container_width=True,
        )
    else:
        live_chart.info("Waiting for first evaluation...")
        live_table.info("No generations yet.")

    if s.waiting_human:
        gate_gen = next(
            (g for g in reversed(s.generations) if g["generation"] == 3),
            s.generations[-1] if s.generations else None,
        )

        with live_human.container():
            st.warning(
                "Human gate: the A/B loop wants to continue past generation 3. "
                "Review the candidate prompt before proceeding."
            )

            if gate_gen:
                st.code(gate_gen["prompt"], language="markdown")
                st.caption(
                    f"Validation score: {gate_gen['mean_score']:.3f} | "
                    f"Cost so far: ${s.token_cost_total:.4f}"
                )

            b1, b2, _ = st.columns([1, 1, 4])

            if b1.button(
                "Approve",
                key=f"approve_{s.run_id}",
                type="primary",
                use_container_width=True,
            ):
                approve_gate("Manual approval from UI.")

            if b2.button(
                "Reject",
                key=f"reject_{s.run_id}",
                use_container_width=True,
            ):
                reject_gate()

            if bool(st.session_state.get("auto_approve", True)):
                st.caption(
                    f"Auto-approve is on. Continuing in {max(0, 4 - s.wait_ticks)} tick(s)."
                )
    else:
        live_human.empty()

    events = s.events[-12:][::-1]

    if events:
        edf = pd.DataFrame(events)
        live_events.dataframe(
            edf[["ts", "node", "message", "evidence_id", "tokens", "cost"]],
            use_container_width=True,
        )
    else:
        live_events.info("No events yet.")

    if s.status in {"complete", "degraded"} and len(s.generations) > 1:
        scores_tuple = tuple(round(g["mean_score"], 3) for g in s.generations)
        run_video = make_run_video(s.run_id, scen["name"], scores_tuple)

        if run_video:
            with live_video.container():
                st.subheader("Run replay video")
                st.image(
                    run_video,
                    caption="Generated from this run's validation scores.",
                    use_container_width=True,
                )
                st.download_button(
                    "Download run replay video (GIF)",
                    data=run_video,
                    file_name=f"{s.run_id}_replay.gif",
                    mime="image/gif",
                    key=f"run_video_{s.run_id}",
                )
        else:
            live_video.empty()
    else:
        live_video.empty()


def render_results() -> None:
    s = st.session_state
    scen = get_scenario()

    output = build_final_output()
    errors = validate_output(output)

    with results_summary.container():
        if s.status == "complete":
            st.success(
                "Run complete. The test set was evaluated exactly once at the end."
            )
        elif s.status == "degraded":
            st.warning(
                "Run finished in degraded mode. Guardrails stopped the loop and the result is usable but should be reviewed."
            )
        else:
            st.info("Run in progress. Results will populate as the A/B loop completes.")

        if errors:
            st.error("Schema validation issues: " + "; ".join(errors))
        else:
            st.caption("Output schema validation passed.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Validation score", f"{output['validation_score']:.3f}")
        c2.metric("Test score", f"{output['test_score']:.3f}")
        c3.metric("Overfit gap", f"{output['overfit_gap']:+.3f}")
        c4.metric("Total cost", f"${output['token_cost_total']:.4f}")

    if s.generations:
        df = pd.DataFrame(s.generations)
        leaderboard = df.sort_values("mean_score", ascending=False).reset_index(drop=True)

        results_table.dataframe(
            leaderboard[
                [
                    "generation",
                    "strategy",
                    "mean_score",
                    "cost",
                    "latency_ms",
                    "tokens",
                ]
            ],
            use_container_width=True,
        )
    else:
        results_table.info("No generation results yet.")

    with results_prompt.container():
        st.subheader("Prompt comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Baseline**")
            st.code(scen["base_prompt"], language="markdown")

        with col2:
            st.markdown("**Best candidate**")
            st.code(s.best_prompt, language="markdown")

    if s.status in {"complete", "degraded"} and len(s.generations) > 1:
        scores_tuple = tuple(round(g["mean_score"], 3) for g in s.generations)
        run_video = make_run_video(s.run_id, scen["name"], scores_tuple)

        if run_video:
            with results_video.container():
                st.subheader("Run replay video")
                st.image(
                    run_video,
                    caption="Animated replay generated from this run.",
                    use_container_width=True,
                )
        else:
            results_video.empty()
    else:
        results_video.empty()

    results_json.json(output)

    with download_area.container():
        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Download result JSON",
                data=json.dumps(output, indent=2).encode("utf-8"),
                file_name=f"{s.run_id}.json",
                mime="application/json",
                key=f"download_json_{s.run_id}",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Download best prompt",
                data=(s.best_prompt or "").encode("utf-8"),
                file_name="best_prompt.txt",
                mime="text/plain",
                key=f"download_prompt_{s.run_id}",
                use_container_width=True,
            )


def render_audit() -> None:
    events = st.session_state.get("events", [])

    if not events:
        audit_table.info("No audit events yet.")
        return

    df = pd.DataFrame(events[::-1])
    query = st.session_state.get("audit_filter", "").strip()

    if query:
        mask = (
            df["message"].str.contains(query, case=False, na=False)
            | df["node"].str.contains(query, case=False, na=False)
            | df["evidence_id"].str.contains(query, case=False, na=False)
        )
        df = df[mask]

    if df.empty:
        audit_table.info("No audit events match the current filter.")
    else:
        audit_table.dataframe(df, use_container_width=True)


tick_interval = {"Fast": 0.45, "Normal": 0.8, "Slow": 1.2}.get(speed_label, 0.8)

fragment_decorator = getattr(st, "fragment", None)

if callable(fragment_decorator):

    @fragment_decorator(run_every=tick_interval)
    def live_loop() -> None:
        maybe_advance()
        render_live()
        render_results()
        render_audit()

    live_loop()

else:
    if st.session_state.get("auto_play", True):
        for _ in range(90):
            maybe_advance()
            render_live()
            render_results()
            render_audit()

            if st.session_state.status != "running":
                break

            time.sleep(tick_interval)
    else:
        render_live()
        render_results()
        render_audit()