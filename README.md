# Stock Market Profitability and Risk Analysis System

A cloud-based system for analyzing stock market risks and profitability using Monte Carlo simulations and parallel processing for GOOG stock.

## Features
- Real-time stock data analysis from Yahoo Finance 
- Monte Carlo simulations for risk assessment (VAR95/99)
- Parallel processing with AWS Lambda/EC2
- Trading signal detection (Three White Soldiers, Three Black Crows)
- Profit/loss calculations
- Audit logging in AWS S3

## Tech Stack
- Google App Engine (Frontend/API)
- AWS Lambda & EC2 (Computation)
- AWS S3 (Storage)
- Python, Flask, Pandas
- yfinance for market data

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
