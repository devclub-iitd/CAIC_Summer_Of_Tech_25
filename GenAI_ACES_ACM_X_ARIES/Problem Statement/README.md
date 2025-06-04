# **Gen AI Problem Statement**

## **Brief Background**

Preparing for consulting internships is a time-intensive and resource-heavy process. Candidates are required to develop proficiency across a wide spectrum of case types—including Market Entry, Market Growth, Pricing, Profitability, and Guesstimates. The standard preparation method relies heavily on peer case groups, where one individual acts as the interviewer and another as the interviewee. This format presents several key challenges:

1. **Dependency on Group Availability**: Effective practice requires coordination with peers, limiting flexibility and consistency—especially for individuals preferring to study on their own schedule.

2. **Inefficiency for Interviewers**: The interviewer must invest time in fully understanding the case beforehand and, in doing so, forfeits the opportunity to attempt the case themselves.

3. **Limited Practice Resources**: Quality case materials are finite and often recycled, reducing the effectiveness of repeated practice.

These constraints make solo preparation inefficient and disadvantageous, especially when candidates are short on time or lack access to reliable case prep groups.

## **The Task**

To address these issues, we propose the development of an intelligent case preparation bot that replicates the dynamics of a case interview session. The bot will:

* Generate unique, parameterized case scenarios (based on case type, difficulty level, industry, and company).

* Simulate an interviewer by presenting the problem and responding to user queries in real time.

* Adaptively reveal data and insights as the user progresses through the case.

* Provide constructive feedback and suggest areas for improvement once the case concludes.

* Extra features outside of these specifications are encouraged.

This solution enables candidates to practice high-quality case interviews independently, on-demand, and with a level of interaction and feedback similar to group practice. The bot aims to reduce preparation friction while improving the depth and consistency of interview readiness.

## **What do case preparations look like?**  
Here’s an example from Case Compendium to make things clearer for you:  

#### **Your client is XYZ Energy. They are concerned about declining profits and would like your help in diagnosing the problem.**

Sure. What exactly does the company do? What geography is it located in?

#### **The company is in the oil business in India. They extract and sell oil to third-party distributors.**

Got it. How much have profits declined by and since when has this been happening? Have profits gone down across India or is the decline limited to a specific region?

#### **The profits have seen a decline of about 2-3%. This has been happening for the last 2 months. Profits have gone down in the Eastern region of the country.**

Have our competitors too seen a similar decline?

#### **No.**

Okay. Profits are equal to Revenues minus Costs. Has there been a change in revenue in the last 2 months?

#### **The revenues have remained stable.**

Sounds like the declining profits are driven by increasing costs then. I would like to draw the value chain of the company and see which costs have seen an increase.

#### **Go ahead.**

Sure. The different steps of the value chain are as follows:  
1\) Licensing: Getting the requisite permissions to commence operations.  
2\) Exploration: Identification of prospective sites to extract oil and conducting studies to assess the sites.  
3\) Extraction: Setting up the drill and other machinery and extracting the oil. This stage also involves money spent on labour.  
4\) Transportation: Oil is usually transported in pipelines. Maybe this stage has seen an increase in repair costs, cost of setting up new pipelines, etc.  
5\) Refining: Converting the crude oil extracted into usable forms.  
6\) Storage: Money spent on tanks or reservoirs may have seen an increase.  
7\) Distribution: Distribution costs may have increased.  
Do we know which stage of the value chain has seen an increase in costs?

#### **In fact, our extraction costs are higher than usual. Why do you think this may be the case?**

Interesting. We can further divide extraction costs into:  
1\) Cost of machinery (drills, motors, etc.)  
2\) Power  
3\) Labour  
4\) Restoration of site  
Am I missing out on any costs at this stage?

#### **You have listed most of the relevant costs. The power used at this stage to run the machinery is unusually high.**

Okay. This could be due to the following reasons:  
1\) Old machinery: Due to wear and tear, our machinery may not be as efficient as before.  
2\) Terrain: Maybe the terrain is different and our drills have to expend more energy to extract oil.  
3\) Wastage: It is possible that power is getting wasted at some point such as transmission.  
4\) Deepness: More power may be expended to extract the oil if it is located deeper in the ground.

#### **Spot on. We had taken up a new project in Assam a couple of months ago. The terrain has been difficult to work on and our costs have seen an increase. What would you recommend?**

The following solutions can be explored:  
1\) Using manpower to loosen the terrain to make it easier for the machinery  
2\) Changing our machinery to ensure energy efficiency  
3\) Recovering the cost from the end-consumers

#### **Good. We can end the case here.**

## **Final Goal**   
Implement a functional conversational bot capable of answering case prep queries with context-awareness and minimal hallucination. Stay tuned for more instructions\! 