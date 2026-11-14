# Likely viva questions and accurate answers

1. **What is Edge AI here?** Inference runs locally on the laptop; no video is sent to a cloud service. Internet is used once to download the weights.
2. **Did you train a model?** No. I used pretrained YOLOv8n (COCO). My work is the system: zones, rules, simulation, storage, evaluation.
3. **Is person detection the same as intrusion detection?** No. Detection answers "is a person visible"; intrusion rules combine location, time and building state.
4. **Why rules instead of ML for the decision?** Access policy is explicit, so rules are transparent, testable and explainable. Learning would need labelled incident data I don't have.
5. **Which agent type is it?** A model-based reflex agent: condition-action rules plus internal state (cooldown timers, alert window). Not goal-, utility- or learning-based.
6. **What are the PEAS?** Performance: correct alerts, few duplicates; Environment: simulated building + video; Actuators: alerts/logs; Sensors: video frames, simulated door/motion events.
7. **How do you avoid an alert on every frame?** Persistence gate (N consecutive frames, re-fire gap) plus per-(room, rule) cooldown.
8. **What does your evaluation show?** On 93 synthetic scenarios the classifier matched all labels (precision, recall, F1 = 1.0). That is expected because labels come from the same policy; it verifies the implementation, not real-world accuracy.
9. **Detector accuracy?** Not measured - I have no annotated footage. I only measure speed (ms/frame, FPS) on my own machine.
10. **How is the video's time determined?** Videos lack a trusted clock, so the user states the assumed start time.
11. **False positives/negatives?** Misses from occlusion/darkness; false detections of person-like objects; misconfigured zones. Hence human review.
12. **Privacy measures?** Local processing, temp file deleted, only metadata stored, no face recognition or identity tracking.
13. **Licence?** Ultralytics YOLO is AGPL-3.0; fine for a college project, but redistribution/deployment as a service has obligations.
14. **Why SQLite?** Built-in, zero configuration, persistent, local.
15. **What is Tiny AI vs your project?** Tiny AI targets microcontrollers with KB-scale models; YOLOv8n (~6.5 MB) is "small" but runs on a laptop CPU, so this is Edge AI, not Tiny AI.
16. **Which explainability technique?** Rule traces: each alert lists the rule, thresholds and evidence. It does not explain neural-network internals (no Grad-CAM etc.).
17. **Future work?** Per-zone polygons, real RTSP camera, annotated test set for detector accuracy, tracking-free dwell-time rules, authentication for dashboard.
