# LinkedIn Posts

## 1. Graduation Post

After several years of work, exams, projects, uncertainty, and a lot of learning, I have completed my degree in Computer Engineering at Universidad Pablo de Olavide.

I am grateful for everything this stage has given me: a stronger technical foundation, good people, difficult courses, late nights debugging, and the feeling of slowly becoming more capable than I was when I started.

During this time, I also had the opportunity to study at Alma Mater Studiorum - Universita di Bologna through Erasmus, where I took courses related to Artificial Intelligence, Deep Learning, and Cognition and Neuroscience. That experience helped me see computer science from a broader perspective and made me even more interested in AI and software systems.

My final degree project focused on meta-learning for feature selection in classification tasks, and it received a final grade of 8/10. It was a technically challenging project, especially because the results were not always clean or easy to interpret, but that made it more valuable as a learning experience.

Now I am focusing on building real portfolio projects around Python backend development, applied AI systems, LLM applications, RAG, and tool-calling architectures.

I am looking for my first solid opportunity as a junior Software Engineer, ideally in a team where I can keep growing around backend engineering, AI systems, and useful AI-enabled products.

This feels like the end of one stage, but more importantly, the beginning of the next one.

## 2. Cerno Project Post

I have been building Cerno, an AI chess coach for Lichess players.

Cerno is a portfolio MVP that explores how chess engines, retrieval systems, and LLM-based reasoning can work together to turn chess games into personalized training recommendations.

The idea is simple: engine analysis can tell you where a move was bad, but improving as a player often requires something more useful:

- What weaknesses appear repeatedly?
- Which parts of the game are causing the most problems?
- What chess theory is relevant to those mistakes?
- What should the player train next?

The current backend can analyze Lichess games or PGN input, run Stockfish analysis, store user history, build weakness profiles, retrieve chess theory with ChromaDB, and generate recommendations using an LLM.

Tech stack:

- Python
- FastAPI
- PostgreSQL
- ChromaDB
- Stockfish
- python-chess
- OpenAI API
- Docker / Docker Compose
- Next.js / React

The RAG knowledge base currently includes 14 chess studies and 358 indexed chunks.

This project has helped me connect several areas I care about: backend engineering, applied AI, retrieval-augmented generation, tool-calling concepts, and product thinking.

It is still an MVP, but that is exactly why it has been useful: it forced me to make practical decisions about architecture, persistence, domain tools, LLM interaction, and how to turn analysis into something a user could actually understand.

## 3. Final Degree Project Post

For my final degree project, I worked on meta-learning for the optimal selection of univariate feature selection filters in classification tasks.

The question behind the project was:

Can we use information about a dataset to recommend which feature selection filter is likely to work best?

The project involved OpenML datasets, meta-features, classification tasks, cross-validation, and several feature selection methods, including Chi-squared, ANOVA F-test, and Mutual Information.

I also experimented with different modeling approaches, including Random Forest, Gradient Boosting, Soft Voting Ensembles, and a hybrid classification/regression perspective.

The results were moderate rather than spectacular, which made the project more interesting in a very practical way. It forced me to think carefully about:

- How to design machine learning experiments
- How to compare methods fairly
- How much signal meta-features actually contain
- Why generalizing model-selection decisions across datasets is difficult
- How to communicate limitations honestly

The project received a final grade of 8/10, but the most valuable part was not the grade. It was learning how messy real experimentation can be, and how important it is to reason clearly when results do not fully match your initial expectations.

That experience is one of the reasons I am now especially interested in applied AI systems, backend engineering, and building tools where machine learning is connected to real workflows.
