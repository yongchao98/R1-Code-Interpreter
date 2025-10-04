# R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning

Our code is based on [Llama-factory](https://github.com/hiyouga/LLaMA-Factory)/[VeRL](https://github.com/volcengine/verl)/[Search-R1](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file) for the SFT and RL training and [SymBench](https://github.com/yongchao98/CodeSteer-v1.0/tree/main)/[BIG-Bench-Hard](https://github.com/yongchao98/R1-Code-Interpreter/tree/main)/[reasoning-gym](https://github.com/open-thought/reasoning-gym) for datasets/benchmarks of reasoning/planning tasks.

## 📝 Introduction
R1-Code-Interpreter is the first framework to train LLMs for step-by-step code reasoning using multi-turn supervised fine-tuning and reinforcement learning. By curating 144 diverse reasoning and planning tasks, we enable Qwen-2.5 models (3B/7B/14B) to autonomously decide when and how to invoke code. Our best model, R1-CI-14B, outperforms GPT-4o (text-only) and approaches GPT-4o with Code Interpreter, showing emergent self-checking behavior via code generation.
<p align="center">
  <img src="Open-images/example.png" width="86%">
</p>

## 🏆 Performance
<p align="center">
  <img src="Open-images/R1-Code-Interpreter-fig1.pdf" width="96%">
</p>

## 🤖 Dataset
The implemented tasks are now available on huggingface-hub:
| Model Name | HF Link                                                |
| ---------- | ------------------------------------------------------------ |
| R1-Code-Interpreter-Data     | [🤗 yongchao98/R1-Code-Interpreter-Data](https://huggingface.co/datasets/yongchao98/R1-Code-Interpreter-Data) |

## 🤖 Model
R1-CI-14B/7B/3B are now available on huggingface-hub:
| Model Name | HF Checkpoint                                                | Size                                                    |
| ---------- | ------------------------------------------------------------ | :------: |
| R1-Code-Interpreter-14B     | [🤗 yongchao98/R1-Code-Interpreter-14B](https://huggingface.co/yongchao98/R1-Code-Interpreter-14B) | **14B**
| R1-Code-Interpreter-7B     | [🤗 yongchao98/R1-Code-Interpreter-7B](https://huggingface.co/yongchao98/R1-Code-Interpreter-7B) | **7B**
| R1-Code-Interpreter-3B     | [🤗 yongchao98/R1-Code-Interpreter-3B](https://huggingface.co/yongchao98/R1-Code-Interpreter-3B) | **3B**

## 🚀 Get Started

### Direct usage (Inference)
First we create the environment for inference and SFT training.
```
git clone https://github.com/yongchao98/R1-Code-Interpreter.git
cd R1-Code-Interpreter
conda create -n llama_factory_infer python=3.11
conda activate llama_factory_infer
cd LLaMA-Factory
pip install -r requirements.txt
cd ..
```
(In benchmark_inference_test.py, fill your python local path of current directory in line 28 and choose desired model type in line 30; In generation_models.py and Search-R1/r1_code_inter/generation_models.py, fill in your OpenAI API for GPT-4o calling to extract the answer). Then we can run the testing R1-CI models with:
```
python benchmark_inference_test.py
```

### SFT training
Then for SFT training, we'd better create another environment. We can do this by running the following command:
```
conda create -n llama_factory_SFT python=3.11
conda activate llama_factory_SFT
cd LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install deepspeed==0.15.2
```
```
cd ..
sh finetune_qwen_7b_1M.sh
```

### GRPO training

Then for GRPO training, we'd better create another environment. We can do this by running the following command:

```
cd R1-Code-Interpreter
conda deactivate
conda create -n R1_code_inter python=3.11
conda activate R1_code_inter
pip install reasoning-gym
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
pip install --upgrade huggingface_hub
huggingface-cli login
cd ../Search-R1
pip install -r requirements.txt
pip3 install flash-attn --no-build-isolation
cd ..
```

(In Search-R1/train_grpo_3B.sh, fill your wandb key and python local path in line 1 and line 2; In r1_code_inter/generation_models.py and ../generation_models.py, fill in your OpenAI API for GPT-4o calling to extract the answer):
```
cd Search-R1
sh train_grpo_3B.sh
```

## ✍️ Citation
```md
@misc{chen2025r1codeinterpretertrainingllmsreason,
      title={R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning}, 
      author={Yongchao Chen and Yueying Liu and Junwei Zhou and Yilun Hao and Jingquan Wang and Yang Zhang and Chuchu Fan},
      year={2025},
      eprint={2505.21668},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.21668}, 
}
```
