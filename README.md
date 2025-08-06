# 🚨 GemmaVision - AI-Powered Security Surveillance System

**Submission for The Gemma 3n Impact Challenge**

An intelligent surveillance system powered by Google's Gemma 3n multimodal AI model for real-time anomaly detection and security monitoring.

## 🎯 Overview

GemmaVision is an on-premise security surveillance solution that leverages Gemma 3n's vision capabilities to detect and classify security incidents in real-time. The system prioritizes privacy by processing data locally while providing government and private security teams with automated threat detection.

## 🚀 Key Features

* **Real-time Camera Surveillance** - Monitor multiple camera feeds simultaneously
* **AI-Powered Anomaly Detection** - Classify incidents: Abuse, Assault, Fighting, Shooting, Burglary, Robbery, Stealing, Shoplifting, Vandalism, Arson, Explosion, Road Accidents, Arrests
* **Web Interface** - Stream processing and offline dashboard
* **Privacy-First Design** - On-premise processing, no cloud data transmission
* **Benchmark Testing** - Comprehensive evaluation framework with UCF Crime dataset

## 🛠 Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management and environment setup.

1. **Install uv** (if you don't have it):
   ```bash
   pip install uv
   ```

2. **Install project dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Run the application or scripts** as described in the relevant sections below.

*uv* is a drop-in replacement for pip and venv, providing faster installs and isolated environments. For more details, see the [uv documentation](https://github.com/astral-sh/uv).



## 📊 Dataset & Evaluation

The system is trained and evaluated on the [UCF Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/), a comprehensive collection of real-world surveillance footage for anomaly detection research.

- 🎯 AI-powered visual analysis (via Gemma 3n)
- 🔍 Advanced entity and keyword search
- 🗂️ Auto-generated folder templates for mission logs and data categorization
- 🧠 Real-time threat detection and alerting pipeline
- 🖼️ Clean, human-readable output reports and summaries

---



## Real time feedback / Scenarios where the model warns of an aggression