# Syllabus alignment - CSE276 Artificial Intelligence Foundations

"Implemented" = present in code and exercised by tests or the dashboard. "Discussed" = covered only in documentation/report.

| Syllabus concept | Project feature | Implementation file | Demonstration method | Status |
|---|---|---|---|---|
| Unit I: problem formulation, performance measures | Intrusion detection formulated as percept → rule decision; precision/recall/F1/FPR | `detection/intrusion_engine.py`, `evaluation/metrics.py` | Evaluation tab | Implemented |
| Unit I: AI applications (smart cities / security) | Smart-building monitoring | whole project | Demo script | Implemented (application) |
| Unit I: AI lifecycle | model selection → inference → evaluation → human review | `docs/architecture.md`, `evaluation/evaluate.py` | Walk through workflow | Implemented + discussed |
| Unit II: production systems / expert systems | Condition-action rules R1-R4 with explanations | `detection/intrusion_engine.py` | Alert detail view shows rule + evidence | Implemented (small production system) |
| Unit II: search, logic, Bayes | - | - | - | **Not covered** |
| Unit III: intelligent agents, PEAS | Sensors = video/sim events; actuators = alerts; environment = building | `detection/intrusion_engine.py`, `simulation/event_simulator.py` | Explain PEAS (`docs/architecture.md`) | Implemented as model-based reflex agent |
| Unit III: goal/utility/learning/multi-agent | - | - | - | **Not covered** (discussed only as future work) |
| Unit IV: deep learning, CNN detectors | Pretrained YOLOv8n person detection | `detection/person_detector.py` | Video Monitor tab | Implemented (inference only, no training) |
| Unit IV: supervised learning concept | Pretrained model was trained supervised on COCO | report | Viva | Discussed |
| Unit IV: Edge AI | Local on-device inference, no cloud | `detection/person_detector.py` | Offline run after model download | Implemented |
| Unit IV: Tiny AI | Nano model is small (~6.5 MB) but runs on a laptop, not a microcontroller | report | Viva | Discussed only |
| Unit V: Explainable AI, transparency | Reason, rule and evidence per alert | `detection/intrusion_engine.py`, `dashboard/views.py` | Alerts tab | Implemented (rule-based explanation, not model-internal XAI) |
| Unit V: privacy and security, governance | Local processing, temp-file deletion, no faces/identity, human review | `dashboard/views.py` | Responsible AI tab | Implemented |
| Unit V: fairness and bias | Limitations of COCO detector discussed; no demographic evaluation done | report | Viva | Discussed only |
| Unit V: Generative AI, LLMs, RAG, prompts | - | - | - | **Not covered** |
| Unit VI: Python ecosystem, PyTorch | Ultralytics/PyTorch, OpenCV, Streamlit, Pandas | `requirements.txt` | Setup | Implemented |
| Unit VI: AI development workflow | download → load → infer → evaluate | `scripts/download_model.py`, `evaluation/evaluate.py` | Demo | Implemented |
| Unit VI: Hugging Face, Ollama, LangChain, TensorFlow, vector DBs | - | - | - | **Not used** |

Related practicals: *Intelligent Autonomous Agent for Smart Environments*, *Investigate Responsible AI using Explainability*, *AI application using the modern AI ecosystem*, and the *Capstone Mini Project* (safety/monitoring theme).
