# EdgeGuard AI: Edge AI-Based Intrusion Detection for Smart Buildings

**Course:** CSE276 Artificial Intelligence Foundations  
**Student:** [Name] · **Roll No.:** [ ] · **Programme:** B.Tech CSE (AI & ML) · **Guide:** [ ] · **Date:** [ ]

## 1. Abstract
EdgeGuard AI is a software-only prototype that detects potentially suspicious activity in a simulated smart building. A pretrained YOLOv8n network detects people in uploaded video locally on a laptop; a separate explainable rule engine combines the detection location, a configurable access schedule and simulated door/motion events to raise alerts. Alerts are logged in SQLite and reviewed by a human in a Streamlit dashboard. The rule classifier was verified on 93 labelled synthetic scenarios; the person detector's speed and accuracy on real footage were not measured in the development sandbox and are left to the user's hardware (Section 12).

## 2. Introduction
Camera systems produce more footage than people can watch. Edge AI runs models close to the data source, reducing bandwidth and privacy exposure. This project applies that idea to building security and aligns with course topics on intelligent agents, deep learning, Edge AI and responsible AI.

## 3. Problem statement
Detecting that a person exists is insufficient for security: presence is normal in public areas and permitted hours. The problem is to decide, transparently, when a detected person or a sensor event is worth a human's attention, without cloud processing, special hardware, or claims about identity or intent.

## 4. Objectives
Run local pretrained person detection; simulate rooms, schedules, doors and motion; implement explainable rules with duplicate suppression; store history; evaluate honestly; document responsible-AI limits.

## 5. Existing approach and proposed solution
*Existing:* human monitoring of CCTV, or simple motion triggers that alert on any movement and produce many false alarms; cloud analytics raise privacy and cost concerns. *Proposed:* local deep-learning detection plus context-aware rules (room, time, building state), persistence/cooldown filtering, and human disposition of alerts.

## 6. System requirements
Hardware: any modern laptop (target MacBook Air M3, 16 GB); no external hardware. Software: Python 3.10-3.13, packages in `requirements.txt`. Internet once for installation and model download.

## 7. Architecture
See `docs/architecture.md`. Pipeline: video/simulated event → Observation → IntrusionEngine → Alert → SQLite → dashboard. Modules are separated into detection, simulation, database, evaluation and dashboard packages.

## 8. AI methodology
**Perception:** YOLOv8n (pretrained on COCO, only the "person" class used), inference only, CPU by default, confidence threshold user-set. A person is "in zone" when the bottom-centre of the box lies in the configured camera zone. **Decision:** a small production system: R1 person in restricted zone out of hours (High), R2 door open out of hours (Medium), R3 motion while building closed (Low), R4 several alerts within a window (High). **Agent view:** model-based reflex agent with state (cooldowns, recent-alert window). **Filtering:** persistence over N frames, re-fire gap, per-(room, rule) cooldown.

## 9. Implementation
Python with OpenCV for decoding/drawing, Ultralytics for inference, SQLite for storage, Pandas for tables, Streamlit for the UI. Time for rules comes from observation timestamps so the same engine serves simulation, video (user-assumed clock) and tests. Terminology enforced in the UI: observed event → rule-based alert → human-confirmed incident.

## 10. Testing methodology
60 pytest tests (unit, integration, headless-UI). Evaluation script scores the rule classifier on a labelled synthetic dataset and, optionally, times the detector on a user video. Detailed in `docs/testing.md`.

## 11. Responsible AI and privacy
Local processing; uploaded video deleted after use; only metadata stored; no facial recognition or identity tracking; alerts are advisory; human review required; limitations of COCO-trained detectors (occlusion, lighting, small objects, dataset bias) acknowledged; YOLO licensed AGPL-3.0; simulated data always labelled.

## 12. Actual results
**Measured in development (Linux sandbox):** 59 tests passed, 1 skipped. Rule classifier on 93 synthetic scenarios: TP 51, FP 0, TN 42, FN 0; accuracy, precision, recall, F1 = 1.000; false-positive rate 0.000; alert-count match 1.000. These labels derive from the written policy, so the result confirms correct implementation, not real-world accuracy.  
**Not measured in the sandbox:** YOLO inference (PyTorch could not be installed there). Insert your own numbers from `python -m evaluation.evaluate --video clip.mp4` on the MacBook:

| Item | Your measurement |
|---|---|
| Device / video resolution | |
| Mean inference ms/frame | |
| Inference FPS / pipeline FPS | |
| Frames processed | |

Detector accuracy (precision/recall for people) was not evaluated because no annotated footage is included.

## 13. Limitations
Single camera at a time; no tracking, so dwell time and counting are not supported; assumed video clock; simulated sensors; rules cover restricted rooms only; no authentication; not a certified security product.

## 14. Future enhancements
Polygon zones, live RTSP input, annotated evaluation set for detector accuracy, tracking and dwell-time rules, authentication, optional Apple-GPU benchmarking, additional agent types.

## 15. Conclusion
EdgeGuard AI demonstrates a complete Edge-AI workflow - pretrained model selection, local inference, rule-based reasoning, explanation, storage, evaluation and human review - while keeping claims limited to what was implemented and measured.

## 16. References
1. Russell, S. & Norvig, P., *Artificial Intelligence: A Modern Approach*, Pearson (course text book).
2. Ultralytics YOLO documentation and repository: https://github.com/ultralytics/ultralytics (AGPL-3.0).
3. Streamlit documentation: https://docs.streamlit.io
4. OpenCV documentation: https://docs.opencv.org
5. SQLite documentation: https://www.sqlite.org/docs.html
6. Lin, T.-Y. et al., "Microsoft COCO: Common Objects in Context" (dataset on which the model was pretrained).
