"""
Routed multi-agent chat
──────────────────────────────────────
* Groq (LLama-3) = domain expert
* Gemini (Flash) = critic / validator
* check_agent()  = tiny Gemini call that returns 0 (new Q) or 1 (clarification)
"""

import os, sys, requests
from google import genai
from google.genai import types as gtypes   # same alias you used

# ─── API KEYS ──────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not (GROQ_API_KEY and GEMINI_API_KEY):
    sys.exit("⚠  Export GROQ_API_KEY & GEMINI_API_KEY before running")

# ─── GEMINI CLIENT -----------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ─── GROQ WRAPPER ------------------------------------------------------
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

def call_groq_expert(question: str) -> str:
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system",
             "content": ("You are Agent-A, a concise domain expert. "
                         "Answer the user's question accurately in one paragraph.")},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
    }
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json=body, timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ─── GEMINI CRITIC & ROUTER ────────────────────────────────────────────
def call_gemini_critic(question: str, answer: str) -> str:
    critic_sys = ("You are Agent-B, a meticulous critic. "
                  "Evaluate the expert's answer for correctness, clarity, and completeness. "
                  "If fixes are needed, provide them; otherwise confirm it is correct.")
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Original question: {question}\n\nExpert answer: {answer}\n\nYour critique:",
        config=gtypes.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=500,
            system_instruction=critic_sys
        )
    )
    return resp.text.strip()

def check_agent(user_turn: str) -> int:
    """
    0 → new info request   → talk to Groq
    1 → clarification ask  → talk to Gemini critic
    """
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_turn,
        config=gtypes.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1,
            system_instruction=("Reply with **1** only if the user is querying regarding "
                            "an earlier answer; otherwise reply **0**. Output just the digit, nothing else.")
        )
    )
    return int(resp.text.strip()[0]) if resp.text.strip()[0] in "01" else 0

# ─── MAIN LOOP ─────────────────────────────────────────────────────────
def run_multi_agent_chat() -> None:
    print("Dual-Agent Chat  (exit › quit)\n")
    last_q, last_a = None, None

    while True:
        try:
            user = input("[YOU] ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Bye!"); break

        if user.lower() in {"exit", "quit"}: print("👋  Bye!"); break
        if not user: continue

        if check_agent(user) == 0 or last_a is None:
            try:
                last_q, last_a = user, call_groq_expert(user)
                print(f"[AGENT-A] {last_a}\n")
            except Exception as e:
                print(f"[Groq-ERR] {e}\n")
        else:
            try:
                critic = call_gemini_critic(last_q, last_a)
                print(f"[AGENT-B] {critic}\n")
            except Exception as e:
                print(f"[Gemini-ERR] {e}\n")

if __name__ == "__main__":
    run_multi_agent_chat()
