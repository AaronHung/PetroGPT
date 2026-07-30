

# PetroGPT

## Repositorio de GitHub: **PetroGPT** & **WestSeverus-7B-DPO-v2**

### **PetroGPT**

enfocado en el preentrenamiento y ajuste fino de LLM específicos del dominio en áreas de Refinación de Gas/Petróleo/Química, incluyendo el modelo MoE (Mixture of Expert) -- WestSeverus-7B-DPO-v2 es un poderoso modelo de lenguaje multidiominio, que combina Mistral-7B, FerdanoGPT (enfocado en matemáticas) y WestLake para tareas científicas, matemáticas y de programación. Clasificado como #1 en el ranking Open LLM, disponible en Hugging Face.

#### **Descripción:**

WestSeverus-7B-DPO-v2 es un modelo de lenguaje multidiominio avanzado desarrollado mediante la combinación de Mistral-7B, FerdanoGPT (enfocado en matemáticas) y WestLake (mejorado en contexto y razonamiento). El modelo destaca en tareas científicas, matemáticas y de programación, aprovechando los últimos avances en técnicas de fusión de modelos como DARE (Drop And Rescale). Alcanzó un ranking principal en el tablero de clasificación Open LLM durante una semana y está disponible públicamente para su uso e investigación en Hugging Face.

- **Capacidades Clave:**
  - Conocimiento científico
  - Razonamiento matemático
  - Programación (tareas de codificación)
- **Método de Fusión:** DARE (Drop And Rescale) para combinar eficazmente las fortalezas de múltiples modelos especializados.

- **Repositorio de Hugging Face:** [PetroGPT/WestSeverus-7B-DPO-v2](https://huggingface.co/PetroGPT/WestSeverus-7B-DPO-v2)

- **Referencia:** El modelo WestSeverus es citado en el artículo "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch" (arXiv: 2311.03099). Consulte la referencia aquí: [arXiv:2311.03099](https://arxiv.org/pdf/2311.03099)

---

# WestSeverus-7B-DPO-v2

WestSeverus-7B-DPO-v2 es un modelo de lenguaje grande de vanguardia que integra múltiples modelos especializados para lograr un rendimiento excepcional en conocimiento científico, razonamiento matemático y tareas de programación. Se basa en **Mistral-7B**, mejorado con capacidades de razonamiento matemático de **FerdanoGPT**, y ulteriormente mejorado con conocimiento científico y de programación de **WestLake**. El modelo utiliza la técnica de fusión de modelos **DARE (Drop And Rescale)**, logrando resultados destacados en benchmarks multitarea.

## Características

- **Conocimiento Científico:** El modelo destaca en dominios científicos que incluyen física, química y conocimiento general.
- **Razonamiento Matemático:** Fuerte rendimiento en tareas que requieren razonamiento matemático avanzado, como **GSM8K** y **MATH**.
- **Programación:** Eficiente para resolver desafíos de programación de conjuntos de datos como **HumanEval** y **MBPP**.

## Entrenamiento y Fusión del Modelo

- **Modelo Base:** Mistral-7B
- **Mejoras:**
  - **FerdanoGPT** (para capacidades matemáticas)
  - **WestLake** (para conocimiento científico y de programación)
- **Técnica de Fusión:** Se utiliza **DARE (Drop And Rescale)** para fusionar los modelos eficazmente, eliminando y reescalando parámetros para mantener el rendimiento en diversas tareas.

## Cómo Usarlo

Puede cargar y usar el modelo fácilmente con la biblioteca `transformers` de Hugging Face:

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

## Licencia

Este repositorio está licenciado bajo la **Licencia MIT**.

## Referencias

- El modelo WestSeverus es citado en el artículo **"Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch"**. Puede acceder al artículo en [arXiv](https://arxiv.org/pdf/2311.03099).
- El modelo está disponible públicamente en Hugging Face en [PetroGPT/WestSeverus-7B-DPO-v2](https://huggingface.co/PetroGPT/WestSeverus-7B-DPO-v2).
