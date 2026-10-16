# app.py
import ast
import csv
import io
import json
import operator
import os
import random
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI

APP_NAME = "Puzzle Generator For Finance Analysts"

# ---------------- Configuration ----------------
PROVIDERS = {
    "OpenAI": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "Anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-haiku-5-5"),
    "Google Gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.0-flash"),
    "OpenRouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
    "Groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "Ollama (local)": ("http://localhost:11434/v1", None, "llama3.1"),
    "Custom endpoint": ("", None, ""),
}

WORK_AREAS = {
    "Morning: data validation & reconciliation": [
        "AI reconciliation exceptions (invoice mismatches, ledger variances)",
        "Automated data pipelines and data quality checks (ERP, CRM, bank feeds)",
        "Cash flow forecasting and liquidity position",
    ],
    "Core forecasting & variance analysis": [
        "Rolling forecasts and machine learning model outputs",
        "Driver-based budget vs actual variance",
        "Macro sensitivity (interest rates, FX, input costs)",
    ],
    "Strategic modeling & business partnering": [
        "Scenario and stress testing (multi-variable, Monte Carlo)",
        "Capital budgeting, NPV/IRR and M&A accretion/dilution",
        "Cost efficiency and budget alignment with departments",
        "Unit economics, cohort analysis and ROI",
    ],
    "Reporting, treasury & executive communication": [
        "Executive summaries and board-level communication",
        "Treasury: short-term investments, debt drawdowns, credit facilities",
        "Working capital optimisation",
    ],
    "Governance, audit & compliance": [
        "Model validation and bias checks",
        "Data provenance, GAAP/IFRS and SOX controls",
    ],
    "Valuation & markets": [
        "DCF valuation and terminal value",
        "P/E, EV/EBITDA and relative valuation",
        "WACC and cost of capital",
        "Bond pricing, duration and yields",
        "Options basics and Greeks",
        "Credit ratios and debt covenants",
        "Earnings quality, guidance and margins",
        "Dividends, buybacks and share count",
        "Market psychology and investor behaviour",
        "Personal finance and household budgeting",
    ],
}
TOPIC_AREA = {t: area for area, topics in WORK_AREAS.items() for t in topics}
TOPICS = list(TOPIC_AREA)

FORMATS = ["Random", "Calculation", "Concept", "Case study", "Spot the error", "Best next action"]
DIFFICULTY_XP = {"Easy": 10, "Medium": 20, "Hard": 35}
TIME_LIMIT = 45

SYSTEM_PROMPT = (
    "You are a senior corporate finance lead and patient teacher who writes exam-quality, "
    "real-world puzzles for analysts working alongside AI tools. "
    "You output only one valid JSON object, with no markdown and no extra text."
)

AI_WORKPLACE_RULE = (
    " The scenario should involve an AI tool such as a forecasting model, reconciliation agent, "
    "cash dashboard or knowledge-graph query. The AI output must contain a plausible but checkable issue, "
    "and the best answer should show that a human analyst verifies before acting."
)

LEARNING_RULE = (
    ' Also include these keys: "market_behaviour" (2-3 simple sentences on why investors or managers would act '
    'in these circumstances); "real_life_connection" (2-3 sentences on how an ordinary person, household or '
    'small business faces a similar decision and what they can learn from it); "key_terms" (a list of up to 3 '
    'objects, each with "term" and "meaning" in plain English); "reflection" (one thoughtful question the learner '
    "should ask themselves before deciding). Explain any complex term in brackets."
)

FORMAT_SPECS = {
    "Calculation": (
        'Set "type" to "calc". Keys: "question"; "inputs" (object mapping names to numbers, '
        'fictional company figures in $ millions or per-share values, percentages as plain numbers like 8 for 8%); '
        '"formula" (a Python arithmetic expression using only the input names, with + - * / ** and parentheses); '
        '"answer" (the exact numeric result of the formula); "unit" (e.g. "$ per share", "%", "x", "$ millions"); '
        '"mistake_formulas" (3 formulas using the same input names that represent common analyst errors); '
        '"explanation" (step-by-step calculation, with the reason behind each step); "tip" (one practical takeaway).'
    ),
    "Concept": (
        'Set "type" to "mc". Keys: "question"; "options" (exactly 4 strings); '
        '"answer_index" (integer 0-3); "explanation" (why the answer is right and why the others are wrong); "tip".'
    ),
    "Case study": (
        'Set "type" to "mc". Write a short realistic fictional company or household scenario with numbers, then one question. '
        'Keys: "question" (scenario plus question); "options" (exactly 4 strings); "answer_index" (0-3); '
        '"explanation"; "tip".'
    ),
    "Spot the error": (
        'Set "type" to "mc". Write a short fictional analyst note, AI forecast summary or calculation containing exactly one flaw. '
        'Keys: "question" (the note plus "What is the flaw?"); "options" (exactly 4 strings describing possible flaws, '
        'one correct); "answer_index" (0-3); "explanation"; "tip".'
    ),
    "Best next action": (
        'Set "type" to "mc". Describe a moment in the analyst workday where an automated system or AI agent has '
        'flagged something (an anomaly, a forecast variance, a covenant risk, a model drift warning, a cash shortfall). '
        'Keys: "question" (the situation plus "What should the analyst do first?"); "options" (exactly 4 realistic '
        'actions, one clearly best, the others tempting but wrong, such as blindly accepting the AI or overreacting); '
        '"answer_index" (0-3); "explanation" (why the best action is right and what the others risk); "tip".'
    ),
}
FORMAT_SPECS = {k: v + AI_WORKPLACE_RULE + LEARNING_RULE for k, v in FORMAT_SPECS.items()}


# ---------------- Model access ----------------
def get_client(provider, base_url, api_key):
    env_name = PROVIDERS[provider][1]
    key = api_key or (os.environ.get(env_name, "") if env_name else "") or "not-needed"
    return OpenAI(base_url=base_url or None, api_key=key, timeout=60.0, max_retries=1)


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in model output:\n{text[:200]}...")
    return json.loads(text[start:end])


# ---------------- Safe arithmetic verifier ----------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def safe_eval(expr, names):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval"))


def to_number(v):
    if isinstance(v, (int, float)):
        return float(v)
    return float(re.sub(r"[^0-9.\-]", "", str(v)))


def distinct(v, others):
    return all(round(v, 2) != round(x, 2) for x in others)


# ---------------- Puzzle building ----------------
def build_mc(raw, meta):
    options = [str(o) for o in raw["options"]]
    idx = int(raw["answer_index"])
    if len(options) != 4 or not 0 <= idx <= 3:
        raise ValueError("Multiple-choice puzzle must have 4 options and a valid answer_index")
    return {**meta, "type": "mc", "question": raw["question"], "options": options,
            "answer_index": idx, "explanation": raw["explanation"],
            "tip": raw.get("tip", ""), "verified": False}


def build_calc(raw, meta):
    names = {k: to_number(v) for k, v in raw["inputs"].items()}
    correct = safe_eval(raw["formula"], names)
    if abs(correct - to_number(raw["answer"])) > max(0.01, 0.005 * abs(correct)):
        raise ValueError(f"Arithmetic check failed: formula evaluates to {correct:,.2f}, but your stated answer was {raw['answer']}.")

    unit = raw.get("unit", "")
    pool = [(correct, True)]
    for f in raw.get("mistake_formulas", []):
        try:
            v = safe_eval(f, names)
        except Exception:
            continue
        if len(pool) < 4 and distinct(v, [x for x, _ in pool]):
            pool.append((v, False))
    for m in (1.1, 0.9, 1.25, 0.8):
        if len(pool) >= 4:
            break
        v = correct * m
        if distinct(v, [x for x, _ in pool]):
            pool.append((v, False))
    if len(pool) != 4:
        raise ValueError("Could not build four distinct options from the formula and mistake formulas.")

    random.shuffle(pool)
    options = [f"{v:,.2f} {unit}".strip() for v, _ in pool]
    idx = next(i for i, (_, ok) in enumerate(pool) if ok)
    explanation = f"{raw['explanation']}\n\nFormula: `{raw['formula']}` with inputs {raw['inputs']}."
    return {**meta, "type": "calc", "question": raw["question"], "options": options,
            "answer_index": idx, "explanation": explanation,
            "tip": raw.get("tip", ""), "verified": True}


def build_prompt(fmt, topic, difficulty, avoid):
    recent = "\n".join(f"- {q}" for q in avoid[-8:]) or "(none yet)"
    area = TOPIC_AREA[topic]
    return f"""Give another real-world corporate finance puzzle to solve.

Workday area: {area}
Specific responsibility: {topic}
Difficulty: {difficulty}
Format: {FORMAT_SPECS[fmt]}

Rules:
- You may use complex finance words, but explain each one in brackets for a beginner.
  Example: "terminal value (the estimated value of all cash flows after the forecast period)".
- Use a fictional company (ERP-style figures, CRM pipeline, bank statement, or board pack wording where relevant).
- Use a realistic trigger such as an earnings surprise, an interest rate change, a covenant test,
  a supplier price hike, an FX swing, a merger rumour or a bank liquidity scare.
- Connect the puzzle to everyday life too: a household budget, a loan, savings, or a small business.
- Keep the question under 150 words and make it read like a real analyst desk note or dashboard alert.
- Be educational, not advice. Do not recommend buying or selling any real security.
- Be original. Do not repeat or closely resemble these recent questions:
{recent}

Return ONLY one JSON object."""


def generate_puzzle_with_self_correction(client, model, fmt, topic, difficulty, avoid):
    """
    Attempts to generate a valid puzzle up to 3 times. If an error occurs (such as an arithmetic 
    mismatch or JSON syntax error), it feeds the exact error message back into the conversation 
    prompt so the AI can correct its mistake.
    """
    base_user_prompt = build_prompt(fmt, topic, difficulty, avoid)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": base_user_prompt}
    ]
    
    errors = []
    for attempt in range(1, 4):
        st.write(f"🔄 **Attempt {attempt} of 3:** Sending request to `{model}`...")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
            text = resp.choices[0].message.content if resp.choices else ""
            if not text or not text.strip():
                raise ValueError("The model returned an empty response.")
            
            raw = parse_json(text)
            key_terms = [t for t in raw.get("key_terms", []) if isinstance(t, dict) and t.get("term")]
            meta = {
                "topic": topic, "area": TOPIC_AREA[topic], "format": fmt, "difficulty": difficulty, "model": model,
                "market_behaviour": raw.get("market_behaviour", ""),
                "real_life": raw.get("real_life_connection", ""),
                "reflection": raw.get("reflection", ""),
                "key_terms": key_terms,
            }
            if raw.get("type") == "calc" or fmt == "Calculation":
                puzzle = build_calc(raw, meta)
            else:
                puzzle = build_mc(raw, meta)
            
            st.write(f"✅ Attempt {attempt} validated successfully!")
            return puzzle
            
        except Exception as e:
            err_msg = f"❌ Attempt {attempt} failed: {type(e).__name__}: {e}"
            errors.append(err_msg)
            st.warning(err_msg)
            
            # If we have attempts left, append assistant output and corrective feedback to messages
            if attempt < 3:
                st.info(f"🛠️ Informing AI about the error and requesting correction...")
                messages.append({"role": "assistant", "content": text if 'text' in locals() and text else "{}"})
                
                correction_prompt = (
                    f"Your previous output failed validation with the following error:\n"
                    f"```\n{e}\n```\n"
                    f"Please analyze this error carefully. If this was an arithmetic error, re-evaluate your formula and inputs "
                    f"so that the mathematical result strictly matches your stated answer. If this was a JSON or formatting error, ensure "
                    f"your response contains valid JSON with all required keys. Return ONLY the corrected JSON object."
                )
                messages.append({"role": "user", "content": correction_prompt})
                
    raise ValueError("Failed to generate a valid puzzle after 3 self-correction attempts:\n" + "\n".join(errors))


# ---------------- Game state ----------------
def init_state():
    defaults = {
        "puzzle": None, "answered": False, "history": [], "topic_stats": {},
        "xp": 0, "last_xp": 0, "streak": 0, "best_streak": 0, "seen": [],
        "explanations": {}, "verified_calc_correct": 0, "pid": 0, "start_time": 0.0,
        "glossary": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def pick_topic(topic_choice, adaptive):
    if topic_choice != "Random":
        return topic_choice
    if not adaptive:
        return random.choice(TOPICS)
    weights = []
    for t in TOPICS:
        s = st.session_state.topic_stats.get(t)
        if not s or s["total"] == 0:
            weights.append(1.5)
        else:
            weights.append(0.5 + 2 * (1 - s["correct"] / s["total"]))
    return random.choices(TOPICS, weights)[0]


def submit(pid):
    p = st.session_state.puzzle
    choice = st.session_state.get(f"choice_{pid}")
    elapsed = time.time() - st.session_state.start_time
    timed_out = st.session_state.get("timed", False) and elapsed > TIME_LIMIT
    correct_text = p["options"][p["answer_index"]]
    is_correct = (choice == correct_text) and not timed_out

    st.session_state.answered = True
    stats = st.session_state.topic_stats.setdefault(p["topic"], {"correct": 0, "total": 0})
    stats["total"] += 1
    stats["correct"] += int(is_correct)

    xp = 0
    if is_correct:
        xp = DIFFICULTY_XP[p["difficulty"]]
        xp += 10 if p["verified"] else 0
        xp += 5 if elapsed < 30 else 0
        st.session_state.streak += 1
        st.session_state.best_streak = max(st.session_state.best_streak, st.session_state.streak)
        if p["verified"]:
            st.session_state.verified_calc_correct += 1
    else:
        st.session_state.streak = 0
    st.session_state.xp += xp
    st.session_state.last_xp = xp

    st.session_state.history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "area": p["area"], "topic": p["topic"], "format": p["format"], "difficulty": p["difficulty"],
        "question": p["question"], "your_answer": choice or "(none)",
        "correct_answer": correct_text, "correct": is_correct,
        "timed_out": timed_out, "elapsed_s": round(elapsed, 1), "model": p["model"],
    })


def area_stats(area):
    correct = total = 0
    for t in WORK_AREAS[area]:
        s = st.session_state.topic_stats.get(t)
        if s:
            correct += s["correct"]
            total += s["total"]
    return {"correct": correct, "total": total}


def badges():
    s = st.session_state
    out = []
    if s.best_streak >= 3:
        out.append("🔥 Hot streak (3+)")
    if s.best_streak >= 10:
        out.append("🏆 Unstoppable (10+)")
    if s.verified_calc_correct >= 5:
        out.append("🧮 Calc ace (5 verified)")
    total = sum(v["total"] for v in s.topic_stats.values())
    correct = sum(v["correct"] for v in s.topic_stats.values())
    if total >= 10 and correct / total >= 0.8:
        out.append("🎯 Sharp analyst (80%+ over 10)")
    if len(s.glossary) >= 15:
        out.append("📘 Jargon buster (15 terms learned)")
    if all(area_stats(a)["total"] >= 3 for a in WORK_AREAS):
        out.append("🧭 Full-day analyst (practised every workday area)")
    return out


def competency_status(stats):
    if stats["total"] == 0:
        return "⚪ Not started"
    acc = stats["correct"] / stats["total"]
    if stats["total"] >= 5 and acc >= 0.75:
        return "🟢 Proficient"
    if acc >= 0.5:
        return "🟡 Developing"
    return "🔴 Needs practice"


def study_plan():
    scored = []
    for topic, s in st.session_state.topic_stats.items():
        if s["total"] >= 2:
            scored.append((s["correct"] / s["total"], topic, s["total"]))
    scored.sort()
    return scored[:3]


# ---------------- UI ----------------
st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="centered")
init_state()

with st.sidebar:
    st.header("⚙️ Model")
    provider = st.selectbox("Provider", list(PROVIDERS), key="provider")
    default_url, _, default_model = PROVIDERS[provider]
    base_url = st.text_input("Base URL", value=default_url, key=f"url_{provider}")
    model = st.text_input("Model name", value=default_model, key=f"model_{provider}",
                          help="Any model your endpoint serves")
    api_key = st.text_input("API key (optional if set as env var)", type="password", key=f"key_{provider}")

    st.header("🎛 Game settings")
    topic_choice = st.selectbox("Responsibility", ["Random"] + TOPICS, key="topic")
    fmt_choice = st.selectbox("Format", FORMATS, key="fmt")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="diff")
    st.checkbox(f"Timed mode ({TIME_LIMIT}s)", key="timed")
    st.checkbox("Adaptive: focus on weak responsibilities", value=True, key="adaptive")

    st.header("📊 Your stats")
    level = st.session_state.xp // 100 + 1
    st.metric("Level", level)
    st.progress((st.session_state.xp % 100) / 100, text=f"{st.session_state.xp} XP")
    st.metric("Current streak", st.session_state.streak)
    total = sum(v["total"] for v in st.session_state.topic_stats.values())
    correct = sum(v["correct"] for v in st.session_state.topic_stats.values())
    acc = round(100 * correct / total) if total else 0
    st.metric("Accuracy", f"{acc}%", f"{correct}/{total}")
    
