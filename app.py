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
    "expert-level real-world puzzles for analysts working alongside AI tools. "
    "Design rigorous options where all distractors are highly plausible, sophisticated, and mimic real professional misjudgments "
    "so the correct answer cannot be guessed easily by elimination."
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
        '"mistake_formulas" (3 sophisticated formulas using the same input names that represent common, subtle analyst calculation errors); '
        '"explanation" (step-by-step calculation, with the reason behind each step); "tip" (one practical takeaway).'
    ),
    "Concept": (
        'Set "type" to "mc". Keys: "question"; "options" (exactly 4 strings: one correct definition/application, and three highly plausible, '
        'expert-sounding misconceptions or half-truths that require deep conceptual clarity to distinguish); '
        '"answer_index" (integer 0-3); "explanation" (why the answer is right and why the others are wrong); "tip".'
    ),
    "Case study": (
        'Set "type" to "mc". Write a short realistic fictional company or household scenario with detailed numbers, then one challenging question. '
        'Keys: "question" (scenario plus question); "options" (exactly 4 strings: one correct conclusion, three subtle analytical traps '
        'derived from typical misinterpretations of the scenario figures); "answer_index" (0-3); '
        '"explanation"; "tip".'
    ),
    "Spot the error": (
        'Set "type" to "mc". Write a short fictional analyst note, AI forecast summary or calculation containing exactly one subtle flaw. '
        'Keys: "question" (the note plus "What is the flaw?"); "options" (exactly 4 strings describing possible flaws where incorrect options '
        'highlight other professional-sounding but non-existent or secondary issues); "answer_index" (0-3); "explanation"; "tip".'
    ),
    "Best next action": (
        'Set "type" to "mc". Describe a moment in the analyst workday where an automated system or AI agent has '
        'flagged something (an anomaly, a forecast variance, a covenant risk, a model drift warning, a cash shortfall). '
        'Keys: "question" (the situation plus "What should the analyst do first?"); "options" (exactly 4 realistic '
        'actions, all sounding tempting and professional, but only one is optimal while others risk premature automation trust, '
        'ignoring edge-case provenance, or overreacting); "answer_index" (0-3); "explanation" (why the best action is right and what the others risk); "tip".'
    ),
}
FORMAT_SPECS = {k: v + AI_WORKPLACE_RULE + LEARNING_RULE for k, v in FORMAT_SPECS.items()}


# ---------------- Model access & Troubleshooting ----------------
def get_client(provider, base_url, api_key):
    env_name = PROVIDERS[provider][1]
    key = api_key or (os.environ.get(env_name, "") if env_name else "") or "not-needed"
    return OpenAI(base_url=base_url or None, api_key=key, timeout=60.0, max_retries=1)


def get_recommendations(err_str):
    err_lower = err_str.lower()
    recs = []
    
    if "401" in err_lower or "authentication" in err_lower or "api key" in err_lower or "incorrect api key" in err_lower or "invalid key" in err_lower:
        recs.append("Check whether the entered API Key matches the model name and selected provider.")
        recs.append("Ensure you haven't mismatched API keys (e.g., using an Anthropic key under the OpenAI provider).")
        recs.append("Verify if your environment variable (like `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) is properly exported if left blank.")
    elif "404" in err_lower or "model" in err_lower or "does not exist" in err_lower or "not found" in err_lower:
        recs.append("Check whether the AI company/provider selected in the sidebar dropdown matches the exact model name.")
        recs.append("Verify the model identifier string (e.g., OpenRouter models require the format `provider/model-name`, whereas OpenAI uses `gpt-4o-mini`).")
    elif "json" in err_lower or "expecting value" in err_lower or "no json object" in err_lower:
        recs.append("The model failed to return strict JSON. Try switching to a more structured model like `gpt-4o-mini`, `gemini-2.0-flash`, or `claude-haiku-5-5`.")
        recs.append("If using a local model via Ollama, ensure it is large enough (at least 7B+ parameters) to reliably follow JSON formatting rules.")
    elif "arithmetic" in err_lower or "formula" in err_lower:
        recs.append("The AI generated a math formula that diverged from its stated answer. The self-correction loop will attempt to fix this automatically.")
    else:
        recs.append("Check whether the API Key matches the model name and provider selection.")
        recs.append("Verify your active internet connection or local server status (if using Ollama).")
        recs.append("Ensure your Base URL is correct for custom or local endpoints.")
    
    return recs


def render_error_with_recommendations(err_message):
    st.error(f"**Error Details:**\n\n`{err_message}`")
    recs = get_recommendations(err_message)
    st.markdown("### 🛠️ Recommended Actions")
    for r in recs:
        st.markdown(f"- {r}")


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
    # Tighter perturbation multipliers (e.g., 3% to 7% margins) so calculation options are much harder to guess
    for m in (1.03, 0.97, 1.06, 0.94):
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
    return f"""Give another expert-level corporate finance puzzle to solve.

Workday area: {area}
Specific responsibility: {topic}
Difficulty: {difficulty}
Format: {FORMAT_SPECS[fmt]}

Rules:
- Make options extremely rigorous and difficult to guess by elimination. All distractors must sound professional and mimic real analytical traps.
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
                raise ValueError("The model returned an empty response. Check if your API Key matches the selected model.")
            
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
            err_msg = f"Attempt {attempt} failed: {type(e).__name__}: {e}"
            errors.append(err_msg)
            st.warning(f"❌ {err_msg}")
            
            if attempt < 3:
                st.info(f"🛠️ Informing AI about the error and requesting self-correction...")
                messages.append({"role": "assistant", "content": text if 'text' in locals() and text else "{}"})
                
                correction_prompt = (
                    f"Your previous output failed validation with the following error:\n"
                    f"```\n{e}\n```\n"
                    f"Please analyze this error carefully. If this was an arithmetic error, re-evaluate your formula and inputs "
                    f"so that the mathematical result strictly matches your stated answer. If this was a JSON or formatting error, ensure "
                    f"your response contains valid JSON with all required keys. Return ONLY the corrected JSON object."
                )
                messages.append({"role": "user", "content": correction_prompt})
                
    full_error_str = " | ".join(errors)
    raise ValueError(f"Failed to generate a valid puzzle after 3 self-correction attempts. Summary: {full_error_str}")


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
    for b in badges():
        st.write(b)

    with st.expander("📤 Share your score"):
        st.code(f"{APP_NAME}: Level {level}, {correct}/{total} correct ({acc}%), "
                f"best streak {st.session_state.best_streak}")
    if st.session_state.history:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(st.session_state.history[0].keys()))
        writer.writeheader()
        writer.writerows(st.session_state.history)
        st.download_button("⬇️ Download history (CSV)", buf.getvalue(),
                           "puzzle_history.csv", "text/csv")

st.title(f"📈 {APP_NAME}")
st.caption("Practise the real workday of a corporate finance analyst in the AI era: "
           "reconciliation, forecasting, scenario analysis, treasury, reporting and governance. "
           "Calculation puzzles are verified in Python. Educational only, not investment advice.")

client = get_client(provider, base_url, api_key)

with st.expander("🔌 Test model connection"):
    if st.button("Run connection test"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Reply with the single word OK."}],
                temperature=0.7,
            )
            reply = resp.choices[0].message.content if resp.choices else ""
            st.success(f"Connected successfully! Model replied: {reply.strip()[:80]}")
        except Exception as e:
            render_error_with_recommendations(str(e))

tab_play, tab_gloss, tab_progress = st.tabs(["🎯 Practice", "📘 Glossary", "📈 Progress"])

# ---- Practice tab ----
with tab_play:
    if st.button("🎲 New puzzle", type="primary", use_container_width=True):
        topic = pick_topic(topic_choice, st.session_state.adaptive)
        fmt = random.choice(FORMATS[1:]) if fmt_choice == "Random" else fmt_choice
        puzzle = None
        
        with st.status(f"Asking {model} for a hard {fmt.lower()} puzzle...", expanded=True) as status:
            try:
                puzzle = generate_puzzle_with_self_correction(client, model, fmt, topic, difficulty, st.session_state.seen)
                status.update(label="Puzzle ready & validated!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Could not generate a valid puzzle", state="error", expanded=True)
                render_error_with_recommendations(str(e))

        if puzzle:
            st.session_state.pid += 1
            puzzle["id"] = st.session_state.pid
            st.session_state.puzzle = puzzle
            st.session_state.answered = False
            st.session_state.start_time = time.time()
            st.session_state.seen.append(puzzle["question"])
            for t in puzzle.get("key_terms", []):
                st.session_state.glossary[t["term"]] = t.get("meaning", "")

    p = st.session_state.puzzle
    if p:
        pid = p["id"]
        answered = st.session_state.answered
        badge = "✅ Verified by Python" if p["verified"] else "🤖 AI-checked"
        st.caption(f"🏢 {p['area']} · {p['topic']} · {p['format']} · {p['difficulty']} · {badge} · "
                   f"generated by `{p['model']}`")
        st.markdown(f"### {p['question']}")

        st.radio("Your answer", p["options"], index=None, key=f"choice_{pid}", disabled=answered)
        if not answered:
            if st.session_state.get("timed", False):
                st.caption(f"⏱ {TIME_LIMIT}s limit, checked when you submit")
            st.button("Submit answer", type="primary",
                      disabled=st.session_state.get(f"choice_{pid}") is None,
                      on_click=submit, args=(pid,))
        else:
            last = st.session_state.history[-1]
            correct_text = p["options"][p["answer_index"]]
            if last["timed_out"]:
                st.warning(f"Time's up. The answer was: {correct_text}")
            elif last["correct"]:
                st.success(f"Correct! +{st.session_state.last_xp} XP")
            else:
                st.error(f"Not quite. The answer is: {correct_text}")

            st.info(p["explanation"])
            if p.get("tip"):
                st.markdown(f"**💡 Analyst tip:** {p['tip']}")
            if p.get("market_behaviour"):
                st.markdown(f"**📰 Why the market or management moves:** {p['market_behaviour']}")
            if p.get("real_life"):
                st.markdown(f"**🏠 In real life:** {p['real_life']}")
            if p.get("reflection"):
                st.markdown(f"**🤔 Think about it:** {p['reflection']}")
            st.markdown(f"**🧭 Skill practised:** {p['area']} → *{p['topic']}*")
            if p.get("key_terms"):
                with st.expander("🔑 Key terms in this puzzle"):
                    for t in p["key_terms"]:
                        st.markdown(f"**{t['term']}**: {t.get('meaning', '')}")

            if pid in st.session_state.explanations:
                st.markdown(st.session_state.explanations[pid])
            elif st.button("🧠 Deep dive: how a pro analyst thinks about this"):
                with st.spinner("Writing a deep dive..."):
                    try:
                        resp_dd = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a patient corporate finance mentor for analysts. Use simple words and explain jargon in brackets. Answer in this order, under 280 words: 1) the core concept in plain English; 2) the step-by-step thinking a senior analyst would use when an AI tool raises this issue; 3) what to verify before trusting or acting on the AI output; 4) how the same idea applies to a personal money decision; 5) two common mistakes analysts make and how to avoid them."},
                                {"role": "user", "content": f"Workday area: {p['area']}\nPuzzle: {p['question']}\nCorrect answer: {correct_text}\nExplanation: {p['explanation']}"}
                            ],
                            temperature=0.7,
                        )
                        st.session_state.explanations[pid] = resp_dd.choices[0].message.content
                        st.rerun()
                    except Exception as e:
                        render_error_with_recommendations(str(e))
    else:
        st.info("Pick your settings in the sidebar and press **New puzzle** to start.")

# ---- Glossary tab ----
with tab_gloss:
    gloss = st.session_state.glossary
    st.subheader("Your finance glossary")
    if gloss:
        st.caption(f"{len(gloss)} terms learned so far. Terms are added as you get new puzzles.")
        query = st.text_input("Search terms", key="gloss_search").strip().lower()
        for term in sorted(gloss):
            if query and query not in term.lower() and query not in gloss[term].lower():
                continue
            st.markdown(f"**{term}**: {gloss[term]}")
    else:
        st.info("Your glossary fills up as you solve puzzles.")

# ---- Progress tab ----
with tab_progress:
    st.subheader("Competency by workday area")
    area_rows = {}
    for area in WORK_AREAS:
        s = area_stats(area)
        if s["total"] > 0:
            area_rows[area] = round(100 * s["correct"] / s["total"])
    if area_rows:
        df = pd.DataFrame({"Accuracy %": area_rows}).sort_values("Accuracy %")
        st.bar_chart(df)
    else:
        st.info("Answer a few puzzles to see your accuracy by workday area.")

    st.subheader("Skill checklist")
    checklist = []
    for area, topics in WORK_AREAS.items():
        for t in topics:
            s = st.session_state.topic_stats.get(t, {"correct": 0, "total": 0})
            checklist.append({
                "Area": area,
                "Responsibility": t,
                "Attempts": s["total"],
                "Accuracy": f"{round(100 * s['correct'] / s['total'])}%" if s["total"] else "-",
                "Status": competency_status(s),
            })
    st.dataframe(pd.DataFrame(checklist), use_container_width=True, hide_index=True)

    st.subheader("Your study plan")
    plan = study_plan()
    if plan:
        st.write("Focus on these responsibilities next. Each one maps to a real task on the analyst's desk:")
        for acc_t, topic, n in plan:
            st.markdown(f"- **{topic}** ({TOPIC_AREA[topic]}): {round(acc_t * 100)}% over {n} puzzles. "
                        "Try Hard difficulty with the Best next action format on this topic.")
    else:
        st.info("After at least two puzzles in a responsibility, your weakest areas will appear here.")

    st.subheader("Habits of a strong AI-era analyst")
    st.markdown(
        "**Verify before you trust.** Check any AI output against a source record before it reaches a CFO deck. "
        "**Explain the story.** For every variance, write one sentence on the business driver, not just the number. "
        "**Stress-test the forecast.** Ask what happens to the answer if interest rates, FX or input costs move. "
        "**Keep the audit trail.** Note which model, data source and assumption produced each conclusion."
    )
