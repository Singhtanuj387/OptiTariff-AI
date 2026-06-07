# ChargeSync AI: Agentic AI-Based Dynamic Tariff Optimization for EV Networks

## 📌 Project Overview
The rapid electrification of mobility has exposed a critical gap in EV charging infrastructure: static, fixed-rate tariff models that remain blind to real-world operational dynamics. Charging stations operating on flat pricing face peak-hour congestion, off-peak underutilization, and grid instability.

**ChargeSync AI** is a multi-agent AI framework designed to solve this. Utilizing real-world, large-scale EV charging session data, this system autonomously predicts demand, recommends dynamic tariffs in real-time, and continuously learns from outcomes to maximize revenue, balance grid demand, and eliminate wait times.

## 🧠 Multi-Agent Architecture
The solution operates using three specialized autonomous agents working in a continuous feedback loop:

1. **Demand Prediction Agent (The "Eyes")**
   - **Role:** Forecasts station utilization rates and congestion probability.
   - **Tech Stack:** XGBoost (GPU-accelerated), PyTorch LSTM, Random Forest.
   - **Accuracy:** Achieves >0.99 R² score in demand forecasting.
2. **Tariff Pricing Agent (The "Brain")**
   - **Role:** Translates demand forecasts into optimal dynamic tariffs.
   - **Tech Stack:** Rule-based Tiering + Gradient Boosting Optimization.
   - **Strategy:** Applies heavy surge pricing during predicted congestion (>80% utilization) and steep discounts during idle off-peak hours (<30% utilization).
3. **Monitoring & Learning Agent (The "Optimizer")**
   - **Role:** Evaluates pricing decisions against simulated real-world behaviors and continuously improves the pricing strategy.
   - **Tech Stack:** Episode-based Feedback Loop measuring Price Elasticity.

## 🚀 Key Achievements
- **Processed Scale:** Handled over 2.1 million raw wide-format timestamps and occupancy records from the UrbanEV dataset, aggregated into 170k+ hourly records.
- **Congestion Elimination:** Station congestion dropped to a mere **0.23%**.
- **Wait Time Reduction:** Simulated queue times plummeted to essentially **0.00 minutes**.
- **Off-Peak Uplift:** Successfully drove a **+6.5%** increase in off-peak charging through automated discounts.

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU (CUDA Support Recommended for XGBoost and PyTorch acceleration)

### Installation
1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd socbiz-openproject
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you install the CUDA-enabled version of PyTorch if running on GPU).*

## 💻 Usage & Execution

The pipeline is managed via the `main.py` orchestrator script. 

To run the **entire pipeline end-to-end**:
```bash
python main.py --all
```
*(Note on Windows: If you encounter Unicode display issues in your terminal, run `$env:PYTHONIOENCODING="utf-8"` before executing).*

### Running Individual Phases
You can also run specific phases of the pipeline sequentially:
```bash
python main.py --preprocess     # Phase 1: Data Cleaning & Feature Engineering
python main.py --eda            # Phase 2: Exploratory Data Analysis Plots
python main.py --demand-agent   # Phase 3: Train XGBoost/LSTM Models
python main.py --tariff-agent   # Phase 4: Generate Dynamic Pricing
python main.py --monitor-agent  # Phase 5: Run Feedback & Learning Loop
python main.py --evaluate       # Phase 6: Compute Final Metrics
```

## 📂 Directory Structure

```text
socbiz-openproject/
│
├── datasets/                   # Raw input datasets (UrbanEV & ACN)
├── outputs/
│   ├── processed/              # Cleaned & Engineered CSV datasets
│   ├── eda/                    # Heatmaps, Congestion plots (PNGs)
│   ├── models/                 # Model outputs, feature importance, predictions
│   ├── evaluation/             # Final evaluation metrics and dashboard visualizations
│   └── presentation/           # Final generated slide deck (PDF/Markdown)
│
├── src/
│   ├── config.py               # Global hyperparameters and path definitions
│   ├── utils.py                # Shared helpers (logging, formatting)
│   ├── data_preprocessing.py   # JSON/CSV parsing & temporal aggregation
│   ├── feature_engineering.py  # Lag features & cyclical encodings
│   ├── eda.py                  # Matplotlib/Seaborn visualization scripts
│   ├── demand_prediction_agent.py # PyTorch LSTM & XGBoost Training
│   ├── tariff_pricing_agent.py    # Dynamic tariff calculation engine
│   ├── monitoring_agent.py        # Feedback loop and episode evaluation
│   └── evaluation.py              # Final metric computation and reporting
│
├── main.py                     # Master Orchestration Script
├── create_pdf.py               # Script to generate PDF Presentation Deck
└── README.md                   # Project Documentation
```

## 📊 Visual Evidence
All evaluation graphs, heatmaps, and demand modeling comparisons are automatically exported to the `outputs/evaluation/` and `outputs/eda/` directories upon pipeline completion. A fully formatted 7-slide presentation is also generated as a PDF in the `outputs/presentation/` folder.
