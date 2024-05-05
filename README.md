# papers
Here are some notes on leading papers relating to AI agents, memory systems, and evaluating agents. I made these notes while I was reading and thinking about these papers as a way to better understand them and be able to come back to them. 
1. Generative Agents by Park et al 2023
	1. goal: generate believable proxies/replications of human behavior using llm pipeline (uses gpt3.5). 
	2. approach: created a 2D environment of a "town" similar to the sims where agents are assigned roles. Visualized for agents using just text (i.e. no actual image sensory input going into agent, just visual description of objects and such). 
	3. created and observed 25 agents in tandem in various "scenarios" such a party throwing scenario, school extracurricular, and valentines day. 
	4. ![[Pasted image 20240505175840.png]] img1: example object labeling in the enviroment (these labels are fed into the agent)
	5. Agent architecture:
		1. Backstory: Very short 10 line backstories. Eg: "John Lin is a pharmacy shopkeeper at the Willow Market and Pharmacy who loves to help people. He is always looking for ways to make the process of getting medication easier for his customers; John Lin is living with his wife, Mei Lin, who is a college professor, and son, Eddy Lin, who is a student studying music theory; John Lin loves his family very much; John Lin has known the old couple next-door, Sam Moore and Jennifer Moore, for a few years; John Lin thinks Sam Moore is a kind and nice man; John Lin knows his neighbor, Yuriko Yamamoto, well; John Lin knows of his neighbors, Tamara Taylor and Carmen Ortiz, but has not met them before; John Lin and Tom Moreno are colleagues at The Willows Market and Pharmacy; John Lin and Tom Moreno are friends and like to discuss local politics together; John Lin knows the Moreno family somewhat well — the husband Tom Moreno and the wife Jane Moreno."
		2. Agents can perform "actions", regular speech, and environmental interaction. programmed to do things such as wakeup and sleep, brush teeth, etc. these morning routines are more hard coded than something that is learnt it seems. for instance, no agent would ever skip a shower if they woke up too late (potential to add randomness to actions such as sleep time and waking up sampling such that sometimes u wake up really late?). 
		3. Agent memory: associative social memory system. They use a memory stream that has exact words from past convos stored for a certain amount of flops. they then take "keywords" and store them for longer. No biological justification for this, and frankly no good benchmarks either. Closest they come is giving this example: "at the start, Sam does not know Latoya Williams. While taking a walk in Johnson Park, Sam runs into Latoya, and they introduce themselves. Latoya mentions that she is working on a photography project: “I’m here to take some photos for a project I’m working on.” In a later interaction, Sam’s interactions with Latoya indicate a memory of that interaction, as he asks “Hi, Latoya. How is your project going?” and she replies “Hi, Sam. It’s going well!”"![[Pasted image 20240505180250.png]]
		4. memory retreival: they assign metrics to each memory in stream: recency, importance, and relevance on scale (0,1) and compare by sum to retrieve. no embeddings or cos similarity. just brute force. (one of the reasons why this project takes 50$ an hour per agent to run, although these costs are decreasing, esp if you use smth like Llama 3). Agents also reflection on recent memory stream to determine some meta goals (they also maintain this goal state throughout which is heavily tied to the agent "backstory" so the agent is basically always zoned in on that). 
		5. Evals: just made some scenarios and eyeballed. basic questions to test memory of events but no counterfactual testing or rigourous math metrics. Ablation testing results:![[Pasted image 20240505180809.png]]
		6. All the code is on [[https://github.com/joonspk-research/generative_agents]].
2. LyfeGame by mit metaconscious lab and robert yang (2023). 
	1. heavily builds on park et al to make 3 improvements: better memory and lesser cost and 3d enviromental input. ![[Pasted image 20240505181114.png]]
	2. architecture: add onto park et al with a new brain system that contains a self monitor unit (which builds upon park et al's reflection but solidifies it). exists in 3d enviroment but alas all input is still in text so the 3d part of the environment mainly serves aesthetics. 
		1. Key insight: Summarize and forget memory. they cite McClelland et al 1995 to show that the hippocampus implements a summarize and forget mechanism and add memory decay as a hyperparam and at a regular timestep, summarize recent experiences, store the summary and forget the exact experiences. Also connected to the self-monitoring and goal adherence pipeline since this helps identify emerging goals and themes. they have longer, almost 1 page agent backstories which each have around 3 lines of biological details and then around 8-9 1 line long term memory experiences.
		2. Evals: they implement a murder mystery scenario: ![[Pasted image 20240505182104.png]]
		3. Some more notes on memory:
			it uses openai's text-embedding-ada-002 model to generate embeddings of text strings. use cosine similarity between embeddings is used for similarity search
			a "forgetting algorithm" is used to maintain diversity in the stored memories by removing redundant items above a similarity threshold
			a "cluster-then-summarize" transformation clusters related incoming memories and summarizes each cluster into a high-level description to reduce memory size while preserving semantic identity
			there is a tiered memory structure with a small "workmem" for immediate items, "recentmem" where the forgetting algorithm is applied, and "longmem" for long-term storage where cluster-then-summarize is used along with the forgetting algorithm again. 
		4. costs 10x less than park et al since they cut down on Llm usage for memory. 
3. exploring the intersection of large language models and agent-based modeling via prompt engineering by edward junprung
	1. goal: simulate human-driven social interactions and behaviors using large language models (llms) like gpt-3.5-turbo. combines agent-based modeling (abm) with llms to enhance understanding of human behavior.
	2. approach: demonstrates llm-driven simulations through prompt engineering, allowing exploration of potential outcomes by adjusting personas for each llm agent. categorizes simulations into one-to-one, one-to-many, and many-to-many.
	3. one-to-one simulation: two-agent adversarial negotiation for buying/selling a pokemon card. agents are given personas that dictate their behavior and goals. emergent strategies arise, not explicitly defined in the simulation.
	4. one-to-many simulation: six-agent collaborative murder mystery. one captain interrogates five passengers (one is the killer) to identify the murderer based on an eyewitness report. uses a memory stream to share context across agents.
	5. many-to-many simulation: acknowledged as relevant for real-world scenarios like fake news spread, but not explored due to implementation complexity and gpt-3.5-turbo's token limit.
	6. they are also heavily inspired by park et al. (2023). use prompt engineering, reinforcement learning human feedback (rlhf) for fine-tuning llms, and training llms to retrieve information beyond their parameters.
4. AGENTBENCH paper, liu et al:
	1. Problem Formulation:
		- Interactive evaluation of LLM-as-Agent modeled as a Partially Observable Markov Decision Process (S, A, T, R, U, O) 
		- State space S, action space A, transition function T : S × A → S, reward function R, task instruction space U, observation space O
		- Agent denoted as M, evaluated using Chain-of-Thought (CoT) prompting
	2. Benchmark Construction:
		- 8 environments across code, game, web categories testing various LLM abilities
		- OS: Bash command execution in Ubuntu Docker. Metric: Success Rate (SR)  
		- DB: Multi-table SQL queries and database operations. Metric: SR
		- KG: Querying large knowledge graphs (e.g. Freebase) using predefined APIs. Metric: Answer F1 
		- DCG: Aquawar digital card game, turn-based strategy against a baseline agent. Metric: Win rate
		- LTP: Lateral thinking puzzle solving through iterative question-answering. Metric: Game progress (%) 
		- HH: Task completion in ALFWorld textual household environment. Metric: SR
		- WS: Web shopping interactions on simulated e-commerce site. Metric: Matching reward
		- WB: Web navigation and element interaction on real websites. Metric: Step SR
		- Prompt engineering: Multi-turn trajectories with CoT, 1-3 shot examples, constrained action spaces
	3. Evaluation Framework:
		- Decoupled server-client architecture for flexible deployment 
		- Task and agent servers communicate via HTTP, allows distributed evaluation
		- Agents wrapped in Docker for environment isolation
		- Evaluation client applies max-flow algorithm for optimal task allocation
		- Prompt truncation to 3500 tokens, handle both chat and text-completion models
	4. Results Analysis: 
		- GPT-4 achieves 4.01 overall score vs 0.51 avg for open-source models, but still fails on complex tasks
		- ![[Pasted image 20240505183903.png]]
		- Breakdown of failure reasons: Task Limit Exceeded (67.9% on KG), Invalid Format (53.3% on DB, 38.5% on DCG), Invalid Action (64.1% on HH)
		- Code training (CodeLlama vs Llama2) helps procedural tasks (WS) but hurts strategy (DCG)
		- Alignment data quality (Vicuna-13b vs Llama2-13b) improves overall performance  
		- Unexpected similarity in Llama2-13b and 70b performance, scaling law not followed
	5. examples of eval implementation: 
		1. they have a decoupled server-client architecture where the Task Server, Agent Server, and Evaluation Client can be deployed separately and communicate via HTTP. This allows for collaborative evaluation of multiple agents and tasks simultaneously.
		2. the eval client uses network flow algorithms, specifically the Edmonds-Karp (CrYing in 124) max-flow algorithm, to optimize the assignment of tasks to agents. It models the problem as a flow network with agent and task nodes. 
		3. The max-flow algorithm is applied to allocate the optimal number of samples for each agent-task pair to maximize evaluation throughput. 
5. User Behavior Simulation with Large Language Model based Agents By wang et al:
	1. Goal: Simulate high-quality user behavior data using large language model (LLM) 
	2. based agents. Existing simulation methods have limitations like simplified user decision processes, dependence on real data, and restriction to single environments.
	3. Approach:
		- Designed an LLM-based agent framework composed of a profile module, memory module, and action module
		- Profile module assigns agents with characters like demographics, personality traits, interests
		- Memory module enables agents to remember past behaviors and evolve, following human cognitive mechanisms (sensory, short-term, long-term memory)
		- Action module determines simulated behaviors like searching, browsing, clicking items, chatting, broadcasting
		- Built an intervenable and resettable sandbox environment for agents to interact with web recommender, chat, broadcast
		- Agent activity levels modeled by Pareto distribution to match real-world long-tail user behavior patterns
	4. experiment: 
		- Compared simulated behaviors of LLM agents with baselines (Embedding, RecSim) and real humans
		- Evaluated key functions of memory module via adversarial subjective evaluation
		- Studied information cocoons and user conformity phenomena using the simulator
	5.  results:
		- Simulated behaviors very close to real humans', outperforming baselines by 68% and only 8% lower than real humans
		- Memory module effectively retains informative and relevant knowledge to support agent behaviors
		- Simulator can reproduce information cocoon phenomenon and test mitigation strategies like increasing recommendation diversity or social connections
		- User conformity behaviors emerge as agents are influenced by friends' opinions, with attitude changes correlated with social network size
6. liu et al: Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory":
	1. goal:
		- Memory-augmented LLMs rely on iterative recalling and repeated reasoning over history to generate high-quality responses in long-term interactions
		- Repeated recall-reason steps over the same history for different questions can produce biased, inconsistent reasoning
		- In contrast, humans keep thoughts in memory and recall them without repeated reasoning
	2. think-in-memory: 
		- enable llms to maintain an evolved memory for storing historical thoughts along the conversation stream
		- 2 stages:
		  1) Recalling stage: Before generating a response, llm recalls relevant thoughts from memory
		  2) Post-thinking stage: after generating a response, llm incorporates historical and new thoughts to update the memory
		- no repeated reasoning by saving post-thinking thoughts as history
	3. Memory Storage:
		- Stores inductive thoughts, defined as text containing a relation triple (Eh, ri, Et) between head entity Eh and tail entity Et via relation ri
		- uses Locality-Sensitive Hashing (LSH) to assign thoughts to hash indices for efficient storage and retrieval
		- LSH maps nearby thought vectors to the same hash index with high probability using random projection: F(x) = arg max([xR; −xR]), where R is a random matrix
		- Supports insert, forget and merge operations on stored thoughts
	4. Memory Retrieval: 
		- Two-stage retrieval process:
		  1) LSH-based retrieval: Hashes query to find its nearest thought group 
		  2) Similarity-based retrieval: Computes pairwise similarity between query and thoughts in the nearest group to retrieve top-k relevant thoughts
		- Improves efficiency by computing pairwise similarity only within the nearest group rather than the entire memory
	5. Memory Updating:
		- Organizes stored thoughts using insert, forget and merge operations to mirror human cognitive processes
		  - Insert: Store new thoughts in memory (uses few-shot prompts with LLMs to generate inductive thoughts)
		  - Forget: Remove unnecessary or contradictory thoughts 
		  - Merge: Combine similar thoughts with the same head entity
		- Operations allow for dynamic updates and evolution of stored thoughts
	6. Experiments and Results:
		- Evaluated on multi-turn dialogue datasets: KdConv (Chinese), GVD (English & Chinese), RMD (Chinese medical conversations)
		- Compared TiM-augmented LLMs (ChatGLM, Baichuan2) with baselines (no memory, SiliconFriend)  
		- Metrics: Retrieval accuracy, response correctness, contextual coherence
		- TiM significantly enhances LLM performance across all metrics, datasets, and languages
		- Reduces retrieval time by ~0.1ms compared to pairwise similarity over entire memory
		- Achieves high top-k retrieval accuracy (e.g. 0.973 for top-10 on KdConv)
		- Real-world medical case study: TiM helps LLM provide more accurate and comprehensive diagnosis by recalling relevant symptoms
7. AdA planner by Sun et al:
	1. goal:
		- improve llms as autonomous agents for sequential decision-making tasks
		- existing methods either take greedy actions without planning or use static plans that can't adapt to environment feedback
		- performance degrades as problem complexity and plan horizon increase
	2. adaplanner:
		- closed-loop approach allowing llm agent to adaptively refine self-generated plan based on environment feedback
		- llm acts as both planner to generate initial plan and refiner to modify plan
		- two refinement strategies:
		1) in-plan refinement: extract useful info from aligned observations to improve upcoming actions
		2) out-of-plan refinement: proactively revise entire plan when observations deviate from predictions
		- uses code-style llm prompts to reduce ambiguity and mitigate hallucination across diverse tasks and environments
		- skill discovery mechanism leverages successful plans as few-shot exemplars to improve sample efficiency
		- ![[Pasted image 20240505191329.png]]
	1. code-based prompting:
		- uses pythonic code prompts instead of natural language
		- reduces llm misinterpretation and hallucination during plan generation and refinement
		- designed for initial planning, feedback generation, and in-episode refinement stages
	2. adaptive closed-loop refinement: 
		- agent selects key timestamps to evaluate success of each sub-goal
		- if sub-goal fails, environment sends trajectory back to refiner for plan revision
		- in-plan refinement uses ask_llm() to parse observations and extract info for future actions
		- out-of-plan refinement generates revised plan and resumes execution from intermediate checkpoint
	3. skill discovery:
		- memory module archives successful plans and trajectories
		- improves planning when solving similar tasks
		- two stages: 1) acquisition of candidate skills, 2) filtering to retain only performance-boosting skills  
	4. experiments and results:
		- outperforms sota on alfworld (91.79% success) and miniwob++ (91.11% success) while using 2-600x fewer samples
		- ablations show importance of closed-loop refinement, code interface, and skill discovery
		- code interface helps mitigate hallucination, especially with gpt-3.5-turbo 
		- skill discovery nearly doubles success rate in alfworld and improves miniwob++ by 15%
8. Reason for Future, Act for Now by liu et al:
	1. goal: 
		- translating llm reasoning into real-world actions remains challenging
		- unclear how to complete tasks with minimum environment interactions through internal reasoning mechanism
		- aims to develop principled framework with provable regret guarantees
	2. reason for future, act for now (rafa):
		- orchestrates reasoning and acting in closed-loop
		- reasoning routine learns from memory buffer and plans future trajectory over long horizon 
		- at each step, takes initial action of planned trajectory, stores feedback in buffer, replans from new state
		- key idea is to cast reasoning as learning and planning in bayesian adaptive mdps
		- prompts llms to form updated posterior (learning) and generate optimal trajectory that maximizes value function (planning)
		- learning and planning subroutines emulate actor-critic update for mdps in-context
		- combines long-term reasoning with short-term acting
	3. bridging llm and rl:
		- formalizes reasoning and acting under bayesian adaptive mdp framework 
		- memory buffer is information state, llm maps it to optimized action through reasoning
		- reasoning composed of learning and planning subroutines
		- learning: forms updated posterior of unknown environment
		- planning: generates optimal policy that maximizes value function 
		- emulates actor-model or actor-critic update in-context, bypasses explicit parameter updates
	4. algorithm:
		- learning subroutine forms updated posterior from memory buffer
		- planning subroutine generates optimal trajectory that maximizes value function
		- reasoning uses memory buffer as context, reduced uncertainty improves planning
		- acting executes initial action of planned trajectory, stores feedback, reinvokes reasoning 
		- switching condition decides when to incorporate newest history chunk 
	5. theory:
		- proves rafa achieves √t regret bound
		- highlights interplay between pretraining knowledge and uncertainty reduction 
		- information ratio characterizes tail behavior of posterior distribution
		- applies to general class of linear kernel mdps 
	6. experiments:
		- outperforms baselines on game of 24, alfworld, blocksworld, tic-tac-toe
		- achieves nearly perfect scores on some benchmarks
		- ablations validate importance of closed-loop refinement, code interface, skill discovery
		- introduces first autonomous llm agent with provable regret guarantees
9. clembench by chalamalasetti et al:
	1. goal: 
		- evaluate chat-optimized llms as conversational agents through game play 
		- expose cllms to constrained game-like settings to challenge specific capabilities
	1. clembench framework:
		- flexible, extensible framework for implementing dialogue games as test instruments
		- enables fast evaluation on large set of models
		- games are text-based, turn-based 
		- uses programmatic "game master" to control game flow and ensure rules are followed
		- easy to implement more games and extend benchmark through community contributions
	2. games in v1.0:
		- taboo: describing concepts without using taboo words, challenges language & world model
		- wordle variants: letter-based word guessing, challenges language & world model, conversational grounding
		- drawing: giving & following drawing instructions, challenges multimodal situation model 
		- reference: identifying referred image out of set, challenges multimodal situation model
		- private/shared: form filling with probing of private vs shared information, challenges agent & discourse model
	3. results:
		- instruction following generally good in best models, difference between gpt-3 and newer models
		- performance tracks development cycle, with newer models performing better
		- performance metrics not saturated; wide gap to assumed human performance  
	4. todo:
		- test non-english language abilities
		- enable human-model game play 
		- experiment with >2 player games and multimodal context
		- use games to analyze model abilities across size variations and training checkpoints
		- optimize individual models on games
		- example template: ![[Pasted image 20240505191209.png]]
	5. conclusions: 
		- shows cllms can serve as models of interactive agents in constrained language games
		- games span breadth of situated language understanding capabilities 
		- game play evaluation is complementary to classical nlp task or preference-based evaluation
		- provides avenue into evaluating interactive language use abilities of cllms
