#Referenced from coursework core python code#

###STEPS TO RESOLVE IMPORT REQUESTS###
#create a new folder lamdba_function
#Wrote the code warmup.py in local.
#In the terminal pip3 install requests -t .
#The above step creates all the dependencies in folder structure. zip the folder lambda_function and upload this zip file to AWS lambda function. It overwrites the code already present if any. 
#The zip upload size is limited to 10mb.

#Have avoided the use of pandas and numpy in lambda as suggested in coursework document. Have replaced it with statistics here.
#The warmup function is defined in GAE code which taken in the values of s and r. the warmup then checks if parallel processing is being achieved for the defined scale value 
#by running this aws lambda function warmup parallelly.

import json
import random
import requests
import statistics

    # Making the HTTP request to get_data to get the processed data from gae.

#def lambda_handler(event, context):
    #gae_service_url = 'https://lsa-stock-market.nw.r.appspot.com/get_data'
    #response = requests.get(gae_service_url)
    #if response.status_code != 200:
    #    return {
    #        'statusCode': 500,
    #       'body': json.dumps({'error': 'Failed to retrieve data from GAE service'})
    #    }

    #minhistory = params['minhistory']
    #shots = params['shots']
    #buy_sell = params['buy_sell']
    #results = []
    
    #Realised that the above approach doesn't work in curl gae url and results in endpoint timeout. Thus now passing the data as json from the GAE code
def lambda_handler(event, context):
    # The incoming 'data' is a stringified JSON, so we need to parse it first
    data_string = event.get('data', '[]')  # Default to an empty array as a string if 'data' is not found
    # print(data_string)
    try:
        # Attempt to parse the stringified JSON data
        data = json.loads(data_string)
        print(data)
    except json.JSONDecodeError:
        # If there is a decoding error, return an error message
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON format'})
        }
    
    # If data is parsed successfully, proceed to return it
    # print(data)  # Debugging: print to CloudWatch logs
    price_history = event.get('price_history', 101)
    shots = event.get('shots', 10000)
    buy_or_sell = event.get('buy_or_sell', 'buy')
    no_of_days = event.get('no_of_days', 30)


    ###This is the part that needs to run parallely###
    #Monte Carlo Simulation
    results = []
    for i in range(price_history, len(data)):
        if (buy_or_sell == 'buy' and data[i]['Buy'] == 1) or (buy_or_sell == 'sell' and data[i]['Sell'] == 1):
            # Extract closing prices for the given price history
            close_prices = [data[j]['Close'] for j in range(i - price_history, i)]
            pct_changes = [close_prices[k + 1] / close_prices[k] - 1 for k in range(len(close_prices) - 1)]

            # Calculate mean and standard deviation
            mean = statistics.mean(pct_changes)
            std = statistics.stdev(pct_changes) if len(pct_changes) > 1 else 0

            # Monte Carlo Simulation
            simulated = [random.gauss(mean, std) for _ in range(shots)]
            simulated.sort(reverse=True)

            # Calculate the 95th and 99th percentile
            var95 = simulated[int(len(simulated) * 0.95)]
            var99 = simulated[int(len(simulated) * 0.99)]

            results.append({'var95': var95, 'var99': var99})

    return json.dumps(results)
    
