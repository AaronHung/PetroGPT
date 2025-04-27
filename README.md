# PetroGPT

## GitHub Repository: **PetroGPT** & **WestSeverus-7B-DPO-v2**

### **PetroGPT**

focusing on pretrain and finetune domain specific LLMs in Gas/Oil Refinement/Chemistry domains including MoE (Mixture of Expert) model -- WestSeverus-7B-DPO-v2 is a powerful multi-domain language model, combining Mistral-7B, FerdanoGPT (math-focused), and WestLake for scientific, mathematical, and programming tasks. Ranked #1 on Open LLM leaderboard, available on Hugging Face.

#### **Description:**

WestSeverus-7B-DPO-v2 is an advanced multi-domain language model developed by combining Mistral-7B, FerdanoGPT (math-focused), and WestLake (context and reasoning-enhanced). The model excels in scientific, mathematical, and programming tasks, leveraging the latest advancements in model fusion techniques such as DARE (Drop And Rescale). It achieved a top ranking on the Open LLM leaderboard for one week and is publicly available for use and research on Hugging Face.

- **Key Capabilities:**
  - Scientific knowledge
  - Mathematical reasoning
  - Programming (coding tasks)
- **Fusion Method:** DARE (Drop And Rescale) to effectively combine the strengths of multiple specialized models.

- **Hugging Face Repo:** [PetroGPT/WestSeverus-7B-DPO-v2](https://huggingface.co/PetroGPT/WestSeverus-7B-DPO-v2)

- **Reference:** WestSeverus model is cited in the paper "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch" (arXiv: 2311.03099). See the reference here: [arXiv:2311.03099](https://arxiv.org/pdf/2311.03099)

---

````markdown
# WestSeverus-7B-DPO-v2

WestSeverus-7B-DPO-v2 is a state-of-the-art large language model that integrates multiple specialized models to achieve exceptional performance in scientific knowledge, mathematical reasoning, and programming tasks. It is built upon **Mistral-7B**, enhanced with mathematical reasoning capabilities from **FerdanoGPT**, and further improved with scientific and programming knowledge from **WestLake**. The model utilizes the **DARE (Drop And Rescale)** technique for model fusion, achieving outstanding results on multi-task benchmarks.

## Features

- **Scientific Knowledge:** The model excels in scientific domains including physics, chemistry, and general knowledge.
- **Mathematical Reasoning:** Strong performance on tasks requiring advanced mathematical reasoning, such as **GSM8K** and **MATH**.
- **Programming:** Efficient at solving programming challenges from datasets like **HumanEval** and **MBPP**.

## Model Training and Fusion

- **Base Model:** Mistral-7B
- **Enhancements:**
  - **FerdanoGPT** (for math capabilities)
  - **WestLake** (for scientific and programming expertise)
- **Fusion Technique:** **DARE (Drop And Rescale)** is used to merge the models effectively, dropping and rescaling parameters to maintain performance across tasks.

## How to Use

You can easily load and use the model with Hugging Face's `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "PetroGPT/WestSeverus-7B-DPO-v2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "What is the integral of x^2?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs['input_ids'])

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
````

## License

This repository is licensed under the **MIT License**.

## References

- WestSeverus model is cited in the paper **"Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch"**. You can access the paper on [arXiv](https://arxiv.org/pdf/2311.03099).
- The model is publicly available on Hugging Face at [PetroGPT/WestSeverus-7B-DPO-v2](https://huggingface.co/PetroGPT/WestSeverus-7B-DPO-v2).
