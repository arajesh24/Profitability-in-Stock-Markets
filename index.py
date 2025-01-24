## Referenced code from LAB 1 for using flask, parallel logic of the lambda from LAB 3 & the coursework to fetch and detect signals, LAB 4 for boto3##
import os
import logging
import math
import time
import random
import pandas as pd
import numpy as np
import yfinance as yf
import json
import boto3
import http.client
import requests
from datetime import date, timedelta
from pandas_datareader import data as pdr
from flask import Flask, jsonify, request
from concurrent.futures import ThreadPoolExecutor

# Override yfinance with pandas_datareader
yf.pdr_override()

app = Flask(__name__) 
os.environ['AWS_SHARED_CREDENTIALS_FILE']='./cred' 

# Declaring the global variables
scalable_service = None
scale_out = 0
parallel = []
stock_data = None
ec2_public_dns=[]
ec2_instance_id=[]
warmup_ready = False
warmup_time = 0
execution_time = 0
cost = 0
results_storage={}
results = []
avg_results= {}
profit_loss_list = []
total_profit_loss = {}
buy_or_sell = None
price_history = 0
shots = 0
no_of_days = 0

# various Flask explanations available at:  
# https://flask.palletsprojects.com/en/1.1.x/quickstart/ 
s3_client = boto3.client('s3')
BUCKET_NAME = 'stock-market-auditing'  # i have created a s3 bucket to store the audit values.

## Referenced code from LAB 1
@app.route('/') 
def hello_world(): 
    return 'A Hello World message to show that something is working!'
     
## Referenced code from LAB 4  
@app.route('/cacheavoid/<name>')
def cacheavoid(name):
    # file exists?
    if not os.path.isfile( os.path.join(os.getcwd(), 'static/'+name) ): 
        return ( 'No such file ' + os.path.join(os.getcwd(), 'static/'+name) )
    f = open ( os.path.join(os.getcwd(), 'static/'+name) )
    contents = f.read()
    f.close()
    return contents 


#The real hassel was to fix the requirements.txt to make it work in GAE. Specially the numpy and scipy dependencies with python312. 

### STEPS TO BE FOLLOWED FOR OTHER ENDPOINT DEPLOYMENTS TO GAE ###
#Once the code is running in localhost.
#So pip3 list in terminal. 
#Update the requirements.txt. 
#Install these using pip3 install -r requirements.txt
#Close the terminal then do the deployment to GAE in a new terminal. The curl should now work for GAE url and give same results as in local host
## learned checking logs & fixing internal server errors on GAE after running gcloud app deploy :D ##

## Refering the code already provided in the Coursework ##
@app.route('/get_data')
def get_data():
    try:
        # fetching stock data from Yahoo Finance, we are retrieving 3 yrs of data 
        today = date.today()
        past3Yrsdata = today - timedelta(days=1095)

        # I'm using 'GOOG' as specified in the coursework, fetching of data should ideally be done only once. 
        data = pdr.get_data_yahoo('GOOG', start=past3Yrsdata, end=today)
        data = data.reset_index(drop=False)

        data['Date'] = pd.to_datetime(data['Date']).dt.date.astype(str)

        # Adding 2 columns Buy and Sell initially fill with 0
        data['Buy'] = 0
        data['Sell'] = 0
        data = data[['Date', 'Open', 'Close', 'Buy', 'Sell']]

        # The for loop detect trends based on the 2 signals specified in the coursework.
        # e.g. Three White Soldiers and Three Black Crows Then add the value 1 if it should buy or sell at the trend identified.
        for i in range(2, len(data)):
            body = 0.01

            # Three White Soldiers
            if (data.Close[i] - data.Open[i]) >= body \
               and data.Close[i] > data.Close[i-1] \
               and (data.Close[i-1] - data.Open[i-1]) >= body \
               and data.Close[i-1] > data.Close[i-2] \
               and (data.Close[i-2] - data.Open[i-2]) >= body:
                data.at[data.index[i], 'Buy'] = 1
                # print("Buy at ", data.index[i])

            # Three Black Crows
            if (data.Open[i] - data.Close[i]) >= body \
               and data.Close[i] < data.Close[i-1] \
               and (data.Open[i-1] - data.Close[i-1]) >= body \
               and data.Close[i-1] < data.Close[i-2] \
               and (data.Open[i-2] - data.Close[i-2]) >= body:
                data.at[data.index[i], 'Sell'] = 1
                # print("Sell at ", data.index[i])

        # checking if the buy/sell logic is working correctly by filtering the data and printing out the results.
        # filtered_data = data.loc['2021-09-25':'2021-09-29']
        # print(filtered_data)

        return data.to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)})
        
#Was earlier sending scalable_service and scale_out to AWS lambda function analyse, but calling /config from AWS lambda function worked in curl localhost but not in curl gae url thus /config is not being used now and i am commenting this out. the AWS lambda function makes use of global variables instead.

#@app.route('/config')
#def get_config():
#    global scalable_service, scale_out
#    return jsonify({
#        'scalable_service': scalable_service,
#        'scale_out': scale_out
#    })
    
# The warmup function will get the values of s and r. the user decides which service they wish to run the analysis in. e.g. lambda or ec2. and r is the no of times the analysis should run parallely. I have set the limit of r by the limitations learned in LAB 3. Exceeding these limit can lead to suspension of aws academy account. 
      
@app.route('/warmup', methods=['POST'])
def warmup_function():
    input_data = request.get_json()
    global parallel,stock_data,means, stds, scalable_service, scale_out, warmup_ready, warmup_time
    scalable_service = input_data.get('s').lower()
    scale_out = int(input_data.get('r'))
    
    stock_data = get_data()
    
    parallel = [value for value in range(scale_out)] #  represent IDs for each parallel execution.
    if scalable_service == 'lambda' and scale_out < 10: #AWS Academy restrictions for parallel execution of lambda
    
#here in warmup i am warming up the lambda function 'analyse' to check if it's ready for analysis. the warmup will have 2 inputs s and r and prints return ok in json format. On the server end it's returning the var95 and var99 values that ran parallelly with thread associated to each showing the parallel execusion worked as expected.

        start_time = time.time()
        lambda_warmup = invoke_lambda()
        end_time = time.time()
        print('LAMBDA WARMUP COMPLETED')
        warmup_ready = True
        warmup_time += end_time-start_time 
        return jsonify({'result': 'ok'})
    elif scalable_service == 'ec2' and scale_out < 20 : #AWS Academy restrictions for parallel execution of EC2 instance
        start_time = time.time()
        instances=createEC2instance()#This creates EC2 instances using AMI id's and setup.bash as the user_data to speed up the creation of new instance. The new instance gets created according to the scale_out value passed by the user in curl. the public DNS and ids are stored in global variables ec2_public_dns and ec2_instance_id
        for i in instances:
            i.wait_until_running()
            i.load()
            ec2_public_dns.append(i.public_dns_name)
            ec2_instance_id.append(i.id)
            print(ec2_public_dns)
            print(ec2_instance_id)
        end_time = time.time()
        print('EC2 WARMUP COMPLETED')
        warmup_ready = True
        warmup_time += end_time-start_time 
        return jsonify({'result': 'ok'})

    else:
        return jsonify({'result': 'error', 'message': 'Invalid Request choose between Lambda or EC2, reduce the scale_out'}), 400
        
#When the warmup is completed the value warmup_ready is updated. If warmup fails then the curl for scaled ready will return 'warm':'false'    
@app.route('/scaled_ready', methods=['GET'])
def scaled_ready_function(): 
    if warmup_ready == True:
        print('specified scale is readied for analysis.')
        return jsonify({'warm': 'true'})
    else:
    	return jsonify({'warm': 'false'})
    	
@app.route('/get_warmup_cost', methods=['GET'])
def get_warmup_cost():
    if not warmup_ready:
        return jsonify({'message': 'Warmup not yet complete. Call /warmup first.'}), 400

    cost = 0
    lambda_price_per_mb_per_second = 0.00001667
    if scalable_service == 'lambda':
        lambda_duration = warmup_time  
        lambda_cost = lambda_duration * lambda_price_per_mb_per_second
        cost += lambda_cost
    elif scalable_service == 'ec2':
        ec2_instance_type = 't2.micro'
        ec2_price_per_hour = 0.0067  
        warmup_hours = warmup_time / 3600
        cost += warmup_hours * ec2_price_per_hour
    else:
        return jsonify({'message': 'Invalid service'}), 400

    return jsonify({'billable_time': warmup_time, 'cost': cost})
#The get_endpoints Retrieves call strings relevant to directly calling each unique endpoint made available at warmup. Thus it returns the curl that was used during warmup.
@app.route('/get_endpoints', methods=['GET'])
def get_endpoints():
    base_url = "https://lsa-stock-market.appspot.com/warmup"
    headers = '-H "Content-Type: application/json"'
    data = f'{{"s": "{scalable_service}", "r": "{scale_out}"}}'
    curl_command = f'curl -X POST {headers} -d \'{data}\' {base_url}'

    return curl_command, 200, {'Content-Type': 'text/plain'} 
         
### code referenced from LAB 3 for parallel execution of lambda function using Threads###    
# Here we are invoking the /analyse aws lambda with  hardcoded values to warmup the /analyse in AWS lambda
def invoke_lambda():
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(getLambdaParallelExecutionResult, idx) for idx in range(len(parallel))]
        results = [future.result() for future in futures]
    return results

def getLambdaParallelExecutionResult(id):
    try:
        host = "o40kt6ujxd.execute-api.us-east-1.amazonaws.com" 
        c = http.client.HTTPSConnection(host)
        #print(stock_data)
        json_payload = {
            "data": stock_data,
            "price_history": 101,
            "shots" : 10000 ,
            "buy_or_sell": "sell",
            "no_of_days": 6
        }
        json_data = json.dumps(json_payload,indent = 4)
        #print(json_data)
        c.request("POST", "/default/analyse", json_data, headers={'Content-type': 'application/json'})  
        
        
        response = c.getresponse()
        data = response.read().decode('utf-8')
        print(data, " from Thread", id)  #To check if parallel runs are working, commenting it out as it loads the gae server with too much data. uncomment it to check the implementation.
        return data
    except IOError:
        print('Failed to open ', host)  #Provide the correct lambda address
        return "unusual behaviour of " + str(id)
  
### code referenced from LAB 3 ###
#connecting to host (invoke url) generated after deployement of lambda function in API gateway. 

@app.route('/analyse', methods=['POST'])
def analyse_function():
    global scalable_service, scale_out, results_storage, buy_or_sell, no_of_days, execution_time, price_history, shots
    
    input_data = request.get_json()
    price_history = input_data.get('h')
    shots = input_data.get('d')
    buy_or_sell = input_data.get('t')
    no_of_days = input_data.get('p')
    
    payload_json= {
    	'data': stock_data,
        'price_history': price_history,
        'shots': shots,
        'buy_or_sell': buy_or_sell,
        'no_of_days': no_of_days
    }

    
    if scalable_service == 'lambda' and scale_out < 10:
        start_time = time.time()
        lambda_warmup = invoke_analyse_lambda(payload_json)
        end_time = time.time()
        execution_time = end_time - start_time
        results_storage = lambda_warmup
        print('LAMBDA ANALYSIS COMPLETED')
        #return str(results_storage)
        return jsonify({'result': 'ok'})
    elif scalable_service == 'ec2' and scale_out < 20:
    #i did tried to implement analyse for ec2. it works when the user runs /var/www/cgi-bin/analyse.py first and then invoke the ec2 url on port 5000 and /process_data returned the correct values of var95 and 99.
    
        start_time = time.time()
        ####FOR TESTING###
        #ec2_analysis_results = send_data(payload_json)
        #ec2_analysis_results = ec2_invoke(payload_json)
        ec2_analysis_results = invoke_analyse_ec2(payload_json)
        end_time = time.time()
        execution_time = end_time - start_time
        #results_storage = ec2_analysis_results
        print('EC2 ANALYSIS COMPLETED')
        return jsonify({'result': 'ok'})
    else:
        return jsonify({'result': 'error', 'message': 'Invalid Request choose between Lambda or EC2, reduce the scale_out'}), 400
        
#Further the code runs the lambda function parallelly using python threads same as it's being implemented in LAB3. Here we are passing the values we got from the user instead of hardcoded values passed during /warmup.        
def invoke_analyse_lambda(payload):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(getAnalyseLambdaParallelExecutionResult, idx, payload) for idx in range(scale_out)]
        results = [future.result() for future in futures]
    return results

def getAnalyseLambdaParallelExecutionResult(id, payload_json):
    try:
        host = "o40kt6ujxd.execute-api.us-east-1.amazonaws.com" 
        c = http.client.HTTPSConnection(host)
        json_data = json.dumps(payload_json)
        #print(json_data)
        c.request("POST", "/default/analyse", json_data, headers={'Content-type': 'application/json'})  
 
        response = c.getresponse()
        data = response.read().decode('utf-8')
        print(data, " from Thread", id)  
        return data
    except IOError:
        print('Failed to open ', host)  
        return "unusual behaviour of " + str(id)
        
#After we ran /analyse the values of var95 and var99 pair obtained from each parallel thread execution is being stored in global variable called results. Thus in /get_sig_vars9599 what i have done is it's extracting the values of var95 and var99 obtained through parallel executions into seprerate lists then converting it into json.
@app.route('/get_sig_vars9599', methods=['GET'])
def get_sig_vars9599():
    global results

    for result_string in results_storage:
        try:

            clean_string = result_string.strip('"').replace('\\"', '"')
            results_list = json.loads(clean_string)
            var95_list = [result["var95"] for result in results_list if "var95" in result]
            var99_list = [result["var99"] for result in results_list if "var99" in result]
            results.append({'var95': var95_list, 'var99': var99_list})
            print(results)
        except json.JSONDecodeError as e:

            print(f"Error decoding JSON: {str(e)}")

    # Return a list where each item represents results from one thread
    return jsonify(results)
    
#We are calculating the avg of each signal then avg of all and storing the results obtained in global variable avg_results   
@app.route('/get_avg_vars9599', methods=['GET'])
def get_avg_vars9599():

    total_var95 = 0
    total_var99 = 0
    count_var95 = 0
    count_var99 = 0


    for entry in results:
        var95_values = entry['var95']
        var99_values = entry['var99']


        total_var95 += sum(var95_values)
        total_var99 += sum(var99_values)
        count_var95 += len(var95_values)
        count_var99 += len(var99_values)


    avg_var95 = total_var95 / count_var95 if count_var95 else 0
    avg_var99 = total_var99 / count_var99 if count_var99 else 0

    # Format the output as in coursework
    global avg_results
    avg_results = {
        'var95': round(avg_var95, 4),  
        'var99': round(avg_var99, 4)
    }

    return jsonify(avg_results)

@app.route('/get_sig_profit_loss', methods=['GET'])
def get_sig_profit_loss():
    global buy_or_sell, no_of_days, profit_loss_list, stock_data
    try:

        stock_data = json.loads(stock_data)

        df = pd.DataFrame(stock_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        if df.empty:
            return jsonify({'error': 'Data frame is empty'}), 400

        
        for index, row in df.iterrows():
            if row[buy_or_sell.capitalize()] == 1: 
                target_date = row['Date'] + pd.Timedelta(days=no_of_days)
                future_rows = df[(df['Date'] > row['Date']) & (df['Date'] <= target_date)]

                if not future_rows.empty:
                    entry_price = row['Close']
                    exit_price = future_rows.iloc[-1]['Close']
                    profit_loss = (exit_price - entry_price) if buy_or_sell.lower() == 'buy' else (entry_price - exit_price)
                    profit_loss_list.append(profit_loss)
                else:
                    profit_loss_list.append(float('NaN'))  

        return jsonify({'profit_loss': profit_loss_list})

    except Exception as e:
        return jsonify({'error': 'Failed to process stock data', 'message': str(e)}), 500

@app.route('/get_tot_profit_loss', methods=['GET'])
def get_tot_profit_loss():
    global total_profit_loss
    try:

        total_profit_loss = sum(profit_loss_list)

        total_profit_loss = round(total_profit_loss, 2)
        return jsonify({'profit_loss': total_profit_loss})
    except Exception as e:

        return jsonify({'error': 'Failed to calculate total profit/loss', 'message': str(e)}), 500
#Not sure why it keeps generating empty chart        
@app.route('/get_chart_url')
def get_chart_url():

    all_var95 = []
    all_var99 = []


    for entry in results:
        all_var95.extend(entry["var95"])
        all_var99.extend(entry["var99"])


    avg_var95 = sum(all_var95) / len(all_var95) if all_var95 else 0
    avg_var99 = sum(all_var99) / len(all_var99) if all_var99 else 0


    avg_line_var95 = [avg_var95] * len(all_var95)
    avg_line_var99 = [avg_var99] * len(all_var99)


    chart_url = f'https://image-charts.com/chart?cht=lc&chs=900x500&chd=t:{",".join(map(str, all_var95))}|{",".join(map(str, all_var99))}|{",".join(map(str, avg_line_var95))}|{",".join(map(str, avg_line_var99))}&chco=FF6347,4682B4,FFD700,32CD32&chdl=95%25+Risk|99%25+Risk|Average+95%25+Risk|Average+99%25+Risk&chxt=x,y&chxr=1,-0.07,0&chg=20,50&chls=2|2|1|1&chma=40,20,20,30'

    return jsonify(chart_url=chart_url) 
       
# Retrives the total billable cost of analysis
@app.route('/get_time_cost', methods=['GET'])
def get_time_cost():
    global execution_time, scalable_service, cost

    if execution_time is None:
        return jsonify({'message': 'Analysis not yet complete. Call /analyse first.'}), 400

    
    if scalable_service == 'lambda':
        lambda_price_per_mb_per_second = 0.00001667
        lambda_duration = execution_time  
        lambda_cost = lambda_duration * lambda_price_per_mb_per_second
        cost += lambda_cost
    elif scalable_service == 'ec2':
        ec2_instance_type = 't2.micro'  
        ec2_price_per_hour = 0.0116  # USD per hour f
        analysis_hours = execution_time / 3600  
        cost += analysis_hours * ec2_price_per_hour
    else:
        return jsonify({'message': 'Invalid service selected. Choose between Lambda or EC2.'}), 400

    return jsonify({'time': execution_time, 'cost': cost})
    
# Here i am retriving all the global variables values and the audit is finally stored in an s3 bucket.
@app.route('/get_audit', methods=['GET'])
def get_audit():
    global s3_client, BUCKET_NAME
    audit_info = {
        's': scalable_service,
        'r': scale_out,
        'h': price_history,
        'd': shots,
        't': buy_or_sell,
        'p': no_of_days,
        'profit_loss': total_profit_loss,
        'av95': avg_results["var95"],
        'av99': avg_results["var99"],
        'time': execution_time,
        'cost': cost
    }
    

    log_key = f"audit_logs/log_{int(time.time())}.json"

    # Uploading the logs to S3
    try:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=log_key, Body=json.dumps(audit_info))
        return jsonify({'audit_info': audit_info})
    except Exception as e:
        return jsonify({'error': 'Failed to store audit log', 'message': str(e)}), 500
    return jsonify(audit_info) 
    
#The audit values stored in the s3 can be retrived so i have created an extra endpoint.
@app.route('/retrieve_audit', methods=['GET'])
def retrieve_audit():
    global s3_client, BUCKET_NAME
    try:
        # List all objects within the 'audit_logs' prefix in the S3 bucket
        object_listing = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='audit_logs/')

        audit_logs = []
        if 'Contents' in object_listing:
            # Retrieve each object
            for obj in object_listing['Contents']:
                # Fetch the object using its key
                log_object = s3_client.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                # Read the object's content
                log_data = log_object['Body'].read().decode('utf-8')
                # Convert from JSON string to Python dictionary
                audit_data = json.loads(log_data)
                audit_logs.append(audit_data)

        return jsonify(audit_logs)

    except Exception as e:
        # Log the error and return an error response
        print(f"Failed to retrieve audit logs: {str(e)}")
        return jsonify({'error': 'Failed to retrieve audit logs', 'message': str(e)}), 500  
        
@app.route('/reset', methods=['GET'])
def reset():
    global warmup_time, execution_time, cost, results_storage, results, avg_results
    global profit_loss_list, total_profit_loss, buy_or_sell, price_history, shots, no_of_days

    # Reset the specified global variables
    
    warmup_time = 0
    execution_time = 0
    cost = 0
    results_storage = {}
    results = []
    avg_results = {}
    profit_loss_list = []
    total_profit_loss = {}
    buy_or_sell = None
    price_history = 0
    shots = 0
    no_of_days = 0

    return jsonify({'result': 'ok'})
     
### STARTING THE CODE FOR EC2 ##
### code referenced from Lab 4 ###
def createEC2instance():
    global ec2_public_dns, ec2_instance_id
    ec2 = boto3.resource('ec2', region_name='us-east-1')

    user_data = """#!/bin/bash
            wget https://lsa-stock-market.appspot.com/cacheavoid/setup.bash
            bash setup.bash"""
    instances = ec2.create_instances(
    ImageId = 'ami-01ca8fc2cce55c0e4', 
    MinCount = scale_out, 
    MaxCount = scale_out, 
    InstanceType = 't2.micro', 
    KeyName = 'stock-market-kp', 
    SecurityGroups=['SSH'],
    UserData=user_data 
    )
    
    return instances
    
### code referenced from Lab 3 for parallel processing ###
def invoke_analyse_ec2(payload):
    global ec2_public_dns
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(getAnalyseEC2ParallelExecutionResult, idx, ec2_public_dns, payload) for idx in range(scale_out)]
        results = [future.result() for future in futures]
    return results

def getAnalyseEC2ParallelExecutionResult(id, ec2_public_dns, payload):
    try:
        conn = http.client.HTTPConnection(ec2_public_dns)
        
        json_data = json.dumps(payload)
        print(f"Thread {id} sending data to EC2: {json_data}")
        
        conn.request("POST", "/analyse", json_data, headers={'Content-type': 'application/json'})  

        response = conn.getresponse()
        data = response.read().decode('utf-8')
        print(f"Thread {id} received data: {data}")
        return data
    except IOError:
        print(f'Failed to connect to {ec2_public_dns} from Thread {id}')
        return f"unusual behaviour of Thread {id}"
        
 # # # # # # #  FOR TESTING EC2 CODE Locally before deploying in aws This works perfectly # # # #     

def send_data(payload):
    url = 'http://ec2-54-147-60-179.compute-1.amazonaws.com:5000/process_data'
     #Make sure to pass the dictionary directly to the `json` parameter.
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response Data:", response.text)

def ec2_invoke(payload):
    url = 'http://ec2-54-147-60-179.compute-1.amazonaws.com/cgi-bin/analyse.py'
    
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response Data:", response.text)
    


@app.route('/terminate', methods=['GET'])       
def terminate_EC2_Instances():
    global ec2_instance_id, scale_out
    scale_out = 0
    if len(ec2_instance_id)>0:
        ec2 = boto3.resource('ec2', region_name='us-east-1')
        for ids in ec2_instance_id:
            ec2.instances.filter(InstanceIds=[ids]).terminate()
        ec2_instance_id=[]
    return jsonify({"result": "ok"}), 200
        
@app.route('/scaled_terminated', methods=['GET'])
def scaled_terminate_Instances():

    if scale_out == 0:
        return jsonify({"terminated": 'true'}), 200
    else:
        return jsonify({"terminated": 'false'}), 200

if __name__ == '__main__': 
    app.run(host='127.0.0.1', port=8080, debug=True)
