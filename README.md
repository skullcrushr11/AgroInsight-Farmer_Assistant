# AgroInsight: AI-Powered Farming Assistant

A comprehensive farming assistance application leverages Agentic Ai workflow (using langgraph) to help farmers make informed decisions.

## Main Features

1. **Crop Recommendation**: Get personalized crop recommendations based on soil conditions and environmental factors.
2. **Yield Prediction**: Predict crop yields using machine learning based on various agricultural parameters.
3. **Fertilizer Recommendation**: Get customized fertilizer recommendations based on soil color or soil type.
4. **Plant Disease Detection**: Upload plant images to detect diseases using computer vision.
5. **Agricultural Knowledge Base**: Ask general farming questions and get accurate, contextual answers.

## Setup Instructions

### Prerequisites

1. Install LM Studio:
   - Download and install LM Studio
   - Download the Falcon3-10b-instruct model
   - Go to Developer window
   - Load the Falcon model and start the server

2. Clone the repository:
```bash
git clone https://github.com/skullcrushr11/AgroInsight-Farmer_Assistant.git
```

3. Create a virtual environment:
```bash
python -m venv venv
```

4. Activate virtual environment (Windows PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

5. Install dependencies:
```bash
pip install -r requirements.txt
```

6. Install ffmpeg (using Chocolatey, preferably in an Administrator powershell):
```bash
choco install ffmpeg
```

7. Start the application:
```bash
streamlit run langgraphapp.py
```

The application will open in your default web browser. Enjoy using AgroInsight!
