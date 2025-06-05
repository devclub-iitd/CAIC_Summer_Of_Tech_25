# Week 2

### Chaining Into Better Frameworks

### By: ARIES X ACES ACM

# **Objective**

LangChain is a framework that simplifies working with Large Language Models (LLMs). This week, you'll learn how to:

* Define prompts that accept structured input  
* Use output parsers to ensure your LLM returns clean, usable data  
* Chain together components like prompts, models, and formatters  
* Route input to different agents depending on intent

By the end of the week, you’ll know how to build scalable, well-structured AI agents that are easy to switch between models like OpenAI, Groq, and Gemini using LangChain.

# **What You’ll Learn and Build**

### Understand LangChain Basics:

* What LangChain is and what problems it solves  
* How LangChain helps orchestrate tools, prompts, and agents

### Use LangChain Expression Language (LCEL):

* Use `|` to link components into a pipeline  
* Run `.invoke()` and `.batch()` methods for single and multiple inputs

### Parse Model Outputs Effectively:

* Structure model responses using `PydanticOutputParser`  
* Use `OutputFixingParser` to fix and validate model outputs automatically

### Route Logic via Prompts and Classifiers:

* Use `RunnableLambda` to direct queries to appropriate agents (e.g., payment, delivery, product support)

# **Weekly Task:**

## **Build the Medium Article LangChain Bot**

Using the concepts above, implement the bot explained in the Medium article given in resources.

### *Instructions:*

* Implement the LangChain-based bot that rewrites product descriptions using a structured prompt  
* Use either **Groq** or **Gemini** as the backend LLM(your main task for this week for which you need to refer to the documentation)  
* Handle different input cases via `PromptTemplate`  
* Validate output using `PydanticOutputParser` or `OutputFixingParser`

### *Key Features Your Bot Must Have:*

* A structured `PromptTemplate` using the CO-STAR format (Context, Objective, Style, Tone, Audience, Response)  
* Integration with your chosen model (Groq or Gemini)  
* Output formatting via parsers  
* Runnable pipeline using LangChain's LCEL  
* Sample test execution via `.invoke()`

### *Video Tutorial to Follow Along*:

* LangChain Tutorial on YouTube: [LangChain Tutorial for Beginners](https://youtu.be/cQUUkZnyoD0?si=YXBQRE9UZ_w-02H3)

### *Resources to Refer*

* Medium Article (Basics of LangChain with an Example): [No-Nonsense Guide to LangChain](https://medium.com/@sureshraghu0706/no-nonsense-guide-to-langchain-a3521d725abf)  
* LangChain Official Documentation: [Docs](https://docs.langchain.com/)


# **Submission Details**

### *Deliverable:* A Python script or Jupyter notebook that:

* Defines prompt templates and output schemas  
* Sets up the LangChain pipeline  
* Executes on at least one test case


### *How to Submit:* We’ll collect submissions via a Google Form shared in your WhatsApp group.

### *Tip:* LangChain helps you think like a backend developer building intelligent workflows—not just prompting. Use its structure to modularize your LLM interactions and scale fast. 

# **Bonus Read:**

* Learn about CO-STAR Prompt Framework (used in this task)  
* Try integrating memory for personalization  
* Try building similar pipelines with `RunnableLambda`
* [OpenAI Swarm - An Educational Agentic Framework](https://github.com/openai/swarm/tree/main)

Good luck and happy chaining\! 🔗

