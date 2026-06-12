== The result of a few hours of hackathon at the MLLS Retreat 2026

The HotPotQA subset used are the 'distractor' setting in which you are given a question and a set of text passages retrieved from wikipedia. Only a few of the passages are relevant to answering the questions and the task is thus to filter the irrelevant when answering.
The hackathon revolved around multi-agent systems and error estimation. In the current setup, each text passage is being provided to an agent. All agents then take turns presenting their infomation and pieces together their partial information. Once an answer is agreed on, the agents terminate the discussion by outputting 'CONCENSUS'.
Lastly an agent extracts the answer to better match the formatting of the dataset and provides a simple error estimation based on the conversation of the agents. 
Finally a token-based F1 metric and an Exact Match metric (provided by the dataset-host) are calculated. A mean aggregate of these metrics are displayed along a calibration curve of the F1 score vs estimated likelihood (using the F1 score for calibration is slightly unconventional but serves as an estimate).

An .env with the field 'llm_token = ...' is required to run the agents. The setup is currently using a locally hosted 'google/gemma-4-26b-a4b' model by this can be changed inside 'autogen_client.py'

The whole pipeline looks like the following:

Running and saving the benchmark: 'uv run src/hackathon/benchmark.py'
Analyzing last run benchmark: 'uv run src/hackathon/analyze_benchmark.py'

Possible roadmap:

- Implement different strategies for agent discussions e.g. consensus keyword required from all of the agents before submitting the answer, using a manager agent to pick the next speaker / choose when to submit
- Give each agent a whole Wikipedia article instead of a snippet (might be paired with a small context winow to force agents to choose which information to share, avoiding dumping of whole articles in the conversation)
- Predicting the uncertainty estimation by having agents vote, use a mean of their individual self assesment
- etc. etc.
