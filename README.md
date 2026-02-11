# ACE-Shield: Actionable Certainty Equivalence for Safe Autonomous Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/Paper-IEEE-red)](main.pdf)

> **"A robot that forgets how to open a door is an inconvenience; a robot that confidently walks through a glass pane is a liability."**

OFFICIAL IMPLEMENTATION of **"The Certainty Manifold: Actionable Certainty Equivalence for Safe Autonomous Adaptation"**.

## 🚀 The Crisis: The Fidelity-Actionability Gap
Standard continual learning agents often achieve perfect predictive fidelity on novel inputs while taking catastrophically wrong actions. ACE closes this gap by enforcing **Actionable Certainty Equivalence**: an agent may only update its policy where its epistemic uncertainty is bounded by aleatoric noise.

## 🛠️ Installation

```bash
git clone https://github.com/anannya-mane/ACE-Shield.git
cd ACE-Shield
pip install -r requirements.txt
