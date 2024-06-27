# Efficient Learning from Verbal Feedback (ELVF)
## Overview
Provided with a simple verbal feedback "Do not talk about elephant", LLM struggles to adhere to the requirement. When asked "What is the largest land mammal on Earth?" GPT-4o fails to stick to the feedback. 

Human picks up this concept much more efficiently. This is a combination of reasoning, short-term memory, and long-term adaptation. 

This code base aims to mimic that behavior.

### (I) Self-Consistency Searching: 

Given a verbal feedback, LLM struggles to follow it on some queries and nails it on others. In order to enhance LLM's performance, we iteratively asks LLM to check whether its response "make sense" -- this adaptively allocate computation to obtain a good response on all queries, which LLM itself is happy with. [REASONING]
```bash
python src/sample_v2.py
```

### (II) Self-Knowledge Clustering: 

Different People has different understanding on the same concept. They pick up things at different speed. Balancing training data distribution is critical for efficient learning, we aims at achieving this uniform distribution by 'self-knowledge clustering' technique, inspired by the LLM2LLM paper. This piece is used to answer the question of "how much training data do we need at the very least ?" question, a iteratively growing training set combined with a fixed test set could be useful (?)

### (III) Representation FineTune for Fast Adaptation:

Adaptation in representation vectors are shown to be much more data efficient than PEFT method. We incorporate that into the codebase to search for a good way of piecing it together. Mathematically we have 
$$\sigma ( W (x + \Delta x) + b) \approx \sigma ( (W + \Delta W) x + (b + \Delta b))$$
which means we could use the representation-adaptor to train our weight-adaptor, which bypasses much of the data curation process. Note that in here the representation vector adaptor is $\Delta x$ and PEFT adaptor is $(\Delta W, \Delta b)$.

### (IV) Self-Distillation FineTuning: 

This Fine Tuning algorithm aims at improving the efficiency in fine-tuning process. LLM learns a compression of the knowledge corpus used in training, such knowlegde is revealed in its complicated logit vector prediction. Supervision with a one-hot vector at a very small scales leads to collapses of such logit vector prediction, this is what happens with traditional Supervised Fine-tuning algorithm. To adrress this, we propose a 'Self Distillation Fine Tuing' algorithm, which provides a similar complicated dense logit vector as supervision, aiming to achieve a better efficiency in learning. [ADAPTATION]

```bash
python -m src.train_peft --arg_file configs/config_dft_v1.json
```
To conduct supervised fine-tuning
```bash
python -m src.train_peft --arg_file configs/config_sft_v3.json
```


