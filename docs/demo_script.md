# 3-5 minute demonstration script

Before: run `./scripts/run.sh`, have a short people-walking clip ready, take screenshots as you go (see checklist).

1. **(0:00-0:30) Problem.** "Cameras detect people, but a person in a lobby is normal. A person in the server room at 3 AM deserves a human look. EdgeGuard separates detection from intrusion."
2. **(0:30-1:00) Overview.** Show empty dashboard: "No fabricated data - everything is zero until events exist." Point at sidebar: local processing, model status.
3. **(1:00-2:00) Simulator.** Generate *Normal door opening* → no alert. Generate *Door after closing time* → Medium alert. Generate *Multiple suspicious* → R4 High escalation. Mention SIMULATED labels.
4. **(2:00-2:45) Explainability.** Alerts tab → open an alert: reason, rule, evidence. Click *Acknowledge*. "Only a human can confirm an incident."
5. **(2:45-4:00) Video.** Video Monitor: choose Server Room, set assumed time 03:00, adjust zone, upload clip, Process. Show boxes, red when inside zone, ms/frame. Then set 11:00 and re-run → no alert (permitted hours). "Detection stays the same; the decision changes."
6. **(4:00-4:30) Evaluation.** Run evaluation: show confusion matrix; state honestly that labels come from policy and detector accuracy wasn't measured.
7. **(4:30-5:00) Responsible AI.** Local processing, no face recognition, AGPL licence, limitations.
