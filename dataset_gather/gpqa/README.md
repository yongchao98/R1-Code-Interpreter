---
license: cc-by-4.0
viewer: true
extra_gated_prompt: >-
  You agree to NOT reveal examples from this dataset in plain text or images
  online, to reduce the risk of leakage into foundation model training corpora.
extra_gated_fields:
  I accept these terms: checkbox
configs:
- config_name: gpqa_extended
  data_files: gpqa_extended.csv
- config_name: gpqa_main
  data_files: gpqa_main.csv
- config_name: gpqa_diamond
  data_files: gpqa_diamond.csv
- config_name: gpqa_experts
  data_files: gpqa_experts.csv
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- open-domain-qa
- open-book-qa
- multiple-choice-qa
pretty_name: GPQA
size_categories:
- n<1K
---

# Dataset Card for GPQA

<!-- Provide a quick summary of the dataset. -->

GPQA is a multiple-choice, Q&A dataset of very hard questions written and validated by experts in biology, physics, and chemistry. When attempting questions out of their own domain (e.g., a physicist answers a chemistry question), these experts get only 34% accuracy, despite spending >30m with full access to Google.

We request that you **do not reveal examples from this dataset in plain text or images online**, to reduce the risk of leakage into foundation model training corpora. 

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

We present GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry. We ensure that the questions are high-quality and extremely difficult: experts who have or are pursuing PhDs in the corresponding domains reach 65% accuracy (74% when discounting clear mistakes the experts identified in retrospect), while highly skilled non-expert validators only reach 34% accuracy, despite spending on average over 30 minutes with unrestricted access to the web (i.e., the questions are "Google-proof"). The questions are also difficult for state-of-the-art AI systems, with our strongest GPT-4 based baseline achieving 39% accuracy. If we are to use future AI systems to help us answer very hard questions, for example, when developing new scientific knowledge, we need to develop scalable oversight methods that enable humans to supervise their outputs, which may be difficult even if the supervisors are themselves skilled and knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier AI systems should enable realistic scalable oversight experiments, which we hope can help devise ways for human experts to reliably get truthful information from AI systems that surpass human capabilities.


- **Curated by:** David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
- **License:** CC BY 4.0

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://github.com/idavidrein/gpqa
- **Paper:** https://arxiv.org/abs/2311.12022

## Uses

The dataset is primarily intended to be used for scalable oversight experiments, although it can also be used for more general LLM capabilities benchmarking.

## Dataset Card Contact

David Rein: idavidrein@gmail.com

---
Submit corrections to examples in GPQA via this form: https://forms.gle/iTY4zMETNsPhJq8R9

---