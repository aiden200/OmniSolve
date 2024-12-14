# Project Name: **OmniSolve**

## Overview

OmniSolve is an advanced AI system designed to bridge the gap between multimodal reasoning, embodied simulation, and real-world impact. This project integrates up-to-date AI models with simulation environments to create a comprehensive framework for reasoning, dataset generation, benchmarking, and simulation. The primary goal is to advance research in multimodal AI, with applications in disaster response, decision-making, and robotics.

---

## Features

### **1. Metadata Retrieval-Augmented Generation (Metadata RAG)**

Leverages metadata and contextual knowledge to enhance retrieval-augmented generation pipelines. Key components include:

- **Efficient Vector Stores** for managing multimodal embeddings.
- **Enhanced Contextual Retrieval** to align with domain-specific applications.

### **2. VideoLLM**

A framework for integrating video data with large language models (LLMs). Features include:

- **Preprocessing Pipelines** for video segmentation and annotation.
- **LLM Integration** to analyze temporal and multimodal video inputs.
- **Inference Pipelines** for real-time decision-making and reasoning.

### **3. Tree of Thoughts (ToT)**

A novel reasoning framework to structure and evaluate decision-making paths:

- **Hierarchical Planning Algorithms** for efficient reasoning.
- **Search Techniques** to explore optimal solutions.
- **Dynamic Context Updates** to improve robustness in changing environments.

### **4. AI2-Thor Simulation**

A simulation environment to train and test embodied AI agents:

- **Interactive 3D Environments** for realistic scenarios.
- **Simulation Data Collection** for reinforcement learning and dataset generation.
- **Customizable Scenarios** to model disaster response, such as the Ohio train derailment.

### **5. Dataset Generation (Ohio Train Derailment)**

A pipeline to generate datasets relevant to disaster scenarios:

- **Synthetic Dataset Creation** for rare or hypothetical events.
- **Real-World Data Integration** to enhance model accuracy.
- **Preprocessing Tools** for cleaning and structuring data.

### **6. Benchmarking with Public Datasets**

Comprehensive evaluation of models using state-of-the-art public datasets:

- **Evaluation Metrics** to ensure performance consistency.
- **Visualization Tools** for comparing results across benchmarks.
- **Dataset Support** for popular datasets like ImageNet, COCO, and OpenImages.

### **7. Embodied Retrieval-Augmented Generation (Embodied RAG)**

Combines embodied reasoning with retrieval-augmented generation:

- **Contextual Reasoning** utilizing 3DSGs (3D Scene Graphs).
- **Knowledge Integration** from dynamic environments.
- **Real-Time Adaptation** to evolving scenarios.

---

## Project Structure

```plaintext
project_root/
├── metadata_rag/                 # Metadata Retrieval-Augmented Generation module
├── video_llm/                    # VideoLLM integration module
├── tree_of_thoughts/             # Tree of Thoughts reasoning module
├── ai2_thor_simulation/          # AI2-Thor simulation environment
├── dataset_generation/           # Dataset generation pipelines
├── benchmarking/                 # Benchmarking tools and evaluations
├── embodied_rag/                 # Embodied Retrieval-Augmented Generation module
├── configs/                      # YAML configuration files
├── scripts/                      # Standalone scripts for executing modules
├── tests/                        # Unit and integration tests
├── utils/                        # Shared utilities
├── examples/                     # Example usage scripts
```

---

## Getting Started

### **1. Clone the Repository**

```bash
git clone https://github.com/aiden200/OmniSolve.git
cd OmniSolve
```

### **2. Set Up the Environment**

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
# .\venv\Scripts\activate  # Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run an Example Script**

Run a demo of Metadata RAG:

```bash
python scripts/run_metadata_rag.py
```

---

## Usage

Each module can be executed independently or integrated into larger workflows. Configuration files in `configs/` allow customization of parameters.

### Example: Running AI2-Thor Simulation

```bash
python scripts/run_ai2_thor.py --config configs/ai2_thor.yaml
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments

Special thanks to ARL-W, the open-source community, researchers from JHU APL, and MIT's CSAIL lab to contributing to advancements in multimodal AI, simulation environments, and disaster response modeling.

This project builds upon the foundational work of the following:

- VideoLLM: Inspired by the research and frameworks provided by [this paper](https://arxiv.org/abs/2411.17991).

- Tree of Thoughts (ToT): Based on the methodologies and algorithms developed by [this paper](https://arxiv.org/abs/2305.10601).

- Metadata RAG: Derived from concepts introduced by [this paper](https://arxiv.org/html/2410.23968v1).

---

## Contact

For questions or collaboration opportunities, please contact:

- **Project Lead**: Aiden Chang
- **Email**: aidenchang@gmail.com
- **GitHub**: [https://github.com/aiden200](https://github.com/aiden200)

---

## Future Work

- Integration with larger embodied AI systems.
<!-- - Expanding support for real-time multimodal inference. -->
- Scaling AI2-Thor simulations for more complex scenarios.
- Continuous benchmarking with updated datasets and metrics.
