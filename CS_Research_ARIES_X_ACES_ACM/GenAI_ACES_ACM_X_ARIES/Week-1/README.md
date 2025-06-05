# Week 1: Zero to One

---

## Focus Areas
- **What are APIs and why they matter?**  
- **The Art of Asking:** Introduction to Prompting  
- **Your First Chatbots:** Using Loops!

---

## Objective
Our goal this week is to help you understand the basics of how we use LLMs (along with something called **APIs**) and how to give clear instructions to these AIs (this is called **“prompting”**). You’ll also get a feel for how to make simple programs that can have a basic chat with a user. This will get you ready for all the exciting AI projects ahead.

---

## What You’ll Learn and Build
By the end of this week, you will:
1. **Understand APIs:**  
   - What an API is  
   - Why APIs are so useful in AI workflows  
2. **Master Prompting Basics:**  
   - Beginner techniques for “prompting” (i.e., effective ways to ask or instruct an AI)  
3. **Build a Simple Chatbot:**  
   - Create a conversational program using basic `while` loops  
   - (Bonus) Explore how chatbots can “remember” earlier conversation parts (context management)

---

## Resources

### APIs
- **A very comprehensive guide by ARIES**  
  <https://docs.google.com/presentation/d/1wx-y71bfDyBCUu9PiGue5WkijynNdIzu/edit?usp=drivesdk&ouid=109388428999110480610&rtpof=true&sd=true>  
  *(Explains what APIs are and how they work.)*  
- **Play around with:**  
  - Google AI Studio: <https://aistudio.google.com>  
  - Groq Cloud Console: <https://console.groq.com>  

### Prompt Engineering
- **OpenAI Prompting Guide**  
  <https://platform.openai.com/docs/guides/text?api-mode=responses>  
- **Anthropic’s Prompt Engineering Docs**  
  <https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview>  
  - *See “Let Claude Think (CoT)”, “Use XML Tags”, and “Chain Complex Prompts”*  
- **Extra Deep Dive**  
  <https://www.promptingguide.ai>  

---

## Assignment: Build a Minimal Chat Agent Using a While Loop and LLM APIs

### Problem Statement
At the heart of every intelligent chatbot—or agent—lies a simple interaction loop: take input, process it, respond. While modern frameworks manage complex multi-agent conversations, a basic version can be built with nothing more than a `while` loop and a few well-structured API calls.

You will:
- Repeatedly ask the user for input.  
- Pass the query to a Large Language Model (LLM) like **Groq** or **Gemini**.  
- Display the LLM’s response.  
- Continue until the user types `exit` or `quit`.

#### Multi-Agent Bonus
Simulate **two agents** with different system prompts:
- **Agent A (Groq):** Domain Expert  
- **Agent B (Gemini):** Critic or Validator  

They can even converse with each other before giving you a final answer—all inside one `while` loop.

---

### Your Task
Complete the following function in Python:

```python
def run_chat_agent():
    """
    Simulates a chatbot agent using a while loop and LLM API calls.

    Your implementation should:
    - Continuously prompt the user for input.
    - Use either Groq or Gemini API to respond.
    - Allow the user to type 'exit' or 'quit' to end the conversation.

    Optional:
    - Add validation before making the API call.
    - Simulate multiple agents by calling different LLMs with distinct system prompts.
    - Route the user query through two agents (e.g., responder and critic).

    Hint:
    Use `input()` to capture user queries, and wrap your API calls inside functions like:
    `call_groq(prompt)` or `call_gemini(prompt, system_prompt=None)`
    """
    pass  # TODO: Implement your conversational loop here
```

#### Example Interaction
```text
[USER] What’s the capital of Australia?  
[GROQ AGENT] Canberra is the capital of Australia.

[USER] Is that correct?  
[GEMINI CRITIC] Yes, that is correct. Canberra is the official capital.

[USER] exit  
Goodbye!
```

## High Order Thinking: Maintaining Context

A basic chat loop works, but a truly robust agent needs to handle conversational memory and token constraints. Think about:

- **Memory Management**  
  - How will you store past user and agent messages?  
  - Where will you keep this history (in-memory list, file, database)?  

- **Token & Context Budget**  
  - How many tokens can your LLM handle per request?  
  - Should you truncate, summarize, or drop older turns to stay within limits?  

- **Role-Specific Views**  
  - If you simulate multiple agents, does each need the full history or just a filtered subset?  
  - Could Agent A see all turns while Agent B only sees the last few?  

- **Context Filtering & Summarization**  
  - Can you automatically summarize earlier exchanges to preserve meaning but reduce length?  
  - What criteria decide which turns to drop or compress?

> **Stretch Goal:**  
> Implement a simple context buffer that retains the last _N_ messages. On each API call, prepend those messages (or a concise summary) to your prompt so the bot “remembers” the conversation.

---

## Submission Details

- **Deliverable:**  
  A Python script (`my_simple_bot.py`) implementing your loop-based agent.

- **Requirements:**  
  1. Greet the user.  
  2. Ask at least 2–3 different questions.  
  3. Provide keyword-based replies (no external AI calls).  
  4. End the chat gracefully when the user types `exit` or `quit`.

- **How to Submit:**  
  We'll let you know soon via the WhatsApp group.

> **Tip:** Keep your code clean, well-commented, and focused on loops and conditionals. No complex libraries needed!

---

## Content Contributors

- Purushottam Sharma  
- Anmol Goel  
- Jahnabi Roy  
- Himanshi Bhandari  
- Tamanna  
- Reyansh  
- Nideesh  
- Avaneesh  
- Ishant  
