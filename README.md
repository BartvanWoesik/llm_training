# LLM Training

# 🧠 LLM Training Repository

This repository contains the training and experimentation setup for our internal LLM-based systems. It is structured around three core modules and uses [**UV**](https://github.com/astral-sh/uv) as the Python package manager for fast, reproducible environments.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) installed:
  ```bash
  curl -Ls https://astral.sh/uv/install.sh | sh
  ```
  or
  ```bash
  pip install uv
  ```

### 2. Environment Setup

To create a virtual environment and install all dependencies:

```bash
uv sync
```

This will install all packages listed in `pyproject.toml`.

To add a new package:

```bash
uv add <package-name>
```

---

## 🧩 Project Modules

### 1. `review_analyses/` – *Learn the Basics*

This module is for foundational training and experimentation. It includes:

- Prompt engineering examples  
- Basic fine-tuning scripts  
- Evaluation notebooks

Use this to understand the data and model behavior.

---

### 2. `mlflow_tracking/` – *Observability & Tracing*

This module integrates **MLflow** for:

- Experiment tracking  
- Model versioning  
- Metric logging

To launch the MLflow UI:

```bash
mlflow ui
```

---

### 3. `chatbot_agent/` – *MCP / Agent*

This module focuses on building a chatbot agent using the **Modular Conversational Pipeline (MCP)** architecture. It includes:

- Agent orchestration logic  
- Tool use and memory handling  
- Integration with trained models

---

## 📦 Development Tips

- Use `uv sync` to ensure consistent environments.  
- Use `.env` files for secrets (never commit them).  

---

## 📬 Questions or Contributions?

Feel free to open issues or submit pull requests. For internal discussions, reach out via Teams or the project channel.
