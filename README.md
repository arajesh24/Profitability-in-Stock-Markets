# Stock Market Profitability and Risk Analysis System

A cloud-based system that analyzes stock market trends using advanced pattern detection and Monte Carlo simulations to predict price movements and assess trading risks.

## Core Purpose
The system identifies potential buying/selling opportunities by detecting specific candlestick patterns:
- Three White Soldiers: Three consecutive rising days (bullish signal)
- Three Black Crows: Three consecutive falling days (bearish signal)

For each signal detected, the system:
1. Calculates risk metrics (VAR95/99) through parallel Monte Carlo simulations
2. Determines potential profit/loss over specified timeframes
3. Helps traders make informed decisions by quantifying both opportunity and risk

## Key Features
- Pattern Detection: Automated identification of trading signals
- Risk Assessment: Parallel Monte Carlo simulations to calculate Value at Risk
- Profit Analysis: Historical performance evaluation of detected signals
- Scalable Architecture: Parallel processing via AWS Lambda/EC2
- Audit Trail: Complete history of analyses and performance

## Technical Stack
- Frontend/API: Google App Engine
- Computation: AWS Lambda, EC2
- Storage: AWS S3
- Data Source: Yahoo Finance (GOOG)

## Performance
- Successfully processes 25,000+ Monte Carlo simulations per analysis
- Supports parallel execution for faster risk calculations
- Real-time signal detection and risk assessment

The system helps traders balance potential returns with associated risks by providing quantitative metrics for decision-making in stock trading.

## Installation
```bash
git clone [repository-url]
cd [project-directory]
pip install -r requirements.txt
python index.py
```

## API Endpoints met 14/15
```
/warmup (lambda)
/scaled_ready
/get_warmup_cost
/get_endpoints
/analyse (lambda)
/get_sig_vars9599
/get_avg_vars9599
/get_sig_profit_loss
/get_tot_profit_loss
/get_time_cost
/get_audit
/reset
/terminate
/scaled_terminated
```

## Required Environment Variables
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

## Usage Example
```bash
# Warm up Lambda with parallel factor of 4
curl -X POST -H "Content-Type: application/json" -d '{"s":"lambda","r":4}' /warmup

# Run analysis with 25000 shots
curl -X POST -H "Content-Type: application/json" -d '{"h":101,"d":25000,"t":"buy","p":7}' /analyse
```

## Contact
Anisha Rajesh: anisharajesh42@gmail.com
Project URL: https://lsa-stock-market.appspot.com
