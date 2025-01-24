#!/usr/bin/env python3

### THIS IS THE CODE IN EC2 TO PERFORM ANALYSIS ###
from flask import Flask, request, jsonify
import json
import random
import statistics
app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json().get('data', [])
    data = json.loads(data)
    price_history = request.get_json().get('price_history', 101)
    shots = request.get_json().get('shots', 10000)
    buy_or_sell = request.get_json().get('buy_or_sell', 'buy')
    no_of_days = request.get_json().get('no_of_days', 30)

    print(f"Type of received data: {type(data)}")  # This should now be <class 'list'>
    print(f"Received data with {data} entries")
    print(f"Received data with {len(data)} entries")
    print(f"Price history: {price_history}, Shots: {shots}, Buy/Sell: {buy_or_sell}, Days: {no_of_days}")

    results = risk_analysis(data, price_history, shots, buy_or_sell)
    return jsonify(results)

def risk_analysis(data, price_history, shots, buy_or_sell):
    results = []
    for i in range(price_history, len(data)):
        if (buy_or_sell == 'buy' and data[i].get('Buy', 0) == 1) or (buy_or_sell == 'sell' and data[i].get('Sell', 0) == 1):
            close_prices = [data[j]['Close'] for j in range(i - price_history, i)]
            pct_changes = [close_prices[k + 1] / close_prices[k] - 1 for k in range(len(close_prices) - 1)]
            mean = statistics.mean(pct_changes)
            std = statistics.stdev(pct_changes) if len(pct_changes) > 1 else 0
            simulated = [random.gauss(mean, std) for _ in range(shots)]
            simulated.sort(reverse=True)
            var95 = simulated[int(len(simulated) * 0.95)]
            var99 = simulated[int(len(simulated) * 0.99)]
            results.append({'var95': var95, 'var99': var99})
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

