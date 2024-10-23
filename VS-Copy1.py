# -*- coding: utf-8 -*-
"""
Created on Wed May 9 15:34:23 2024
This code works if you have weblog data. 
This code finds more than 1,000 consecutive IPs in one second at the same time with each responsesecode value of 404 or 200. 
Next, the generative ai model analyzes the payload of the hacker's vulnerability scan period 
and the payload of the attack after the vulnerability scan. It can even analyze API vulnerability attacks.
@author: Hui, Hong
"""

import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
from ipaddress import ip_address
import re
from collections import Counter
from dateutil import parser
import string

# Functions needed to automatically identify delimiters and column names for multiple txt files with different user input methods and convert them into a single panda data frame
def find_best_delimiter(lines):
    delimiters = [r'\s+', ',', ';', '\t']
    best_delimiter = None
    max_parts = 0

    for delim in delimiters:
        total_parts = sum(len(re.split(delim, line.strip())) for line in lines)
        if total_parts > max_parts:
            best_delimiter = delim
            max_parts = total_parts

    return best_delimiter

def auto_split(line, delimiter):
    parts = re.split(delimiter, line.strip())
    return parts

def infer_column_names(parsed_lines):
    if not parsed_lines:
        return []

    ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
    date_pattern = re.compile(r'\[?\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\]?')
    request_pattern = re.compile(r'\/[^\s]+')
    
    sample_line = parsed_lines[0]
    column_names = []

    for part in sample_line:
        if ip_pattern.match(part):
            column_names.append('ip_address')
        elif date_pattern.match(part):
            column_names.append('timestamp')
        elif request_pattern.match(part):
            column_names.append('request')
        elif part.isdigit() and len(part) == 3:
            column_names.append('status_code')
        elif part.isdigit() and 1 <= len(part) <= 4 or part == '-':
            column_names.append('size')
        else:
            column_names.append('unknown')
        
    counts = Counter(column_names)
    for i, name in enumerate(column_names):
        if counts[name] > 1:
            suffix = counts[name]
            column_names[i] = f'{name}_{suffix}'
            counts[name] -= 1

    return column_names

def standardize_date(date_str):
    # Convert YYYYMMDD format to YYYY-MM-DD format
    if re.match(r'^\d{8}$', date_str):
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str

def parse_uploaded_files(uploaded_files):
    parsed_data = []
    for uploaded_file in uploaded_files:
        try:
            file_contents = uploaded_file.getvalue().decode('utf-8')
            lines = file_contents.splitlines()
            delimiter = find_best_delimiter(lines)
            for line in lines:
                parts = auto_split(line, delimiter)
                parsed_data.append(parts)
        except Exception as e:
            st.error(f"An error occurred while parsing file {uploaded_file.name}: {str(e)}")
            return pd.DataFrame()

    if not parsed_data:
        st.warning("No data found in uploaded files.")
        return pd.DataFrame()

    columns = infer_column_names(parsed_data)

    # Check for missing column names.
    required_columns = ['ip_address', 'timestamp', 'request', 'status_code', 'size']
    if not all(col in columns for col in required_columns):
        st.write("Some columns were not automatically recognized. Please enter all column names in the correct order. Information about ip, date, payload, response status code, and payload size must be entered as follows: ip_address, timestamp, request, status_code, size. Feel free to name the other column names:")
        with st.form(key='column_input_form'):
            columns = []
            for i in range(len(parsed_data[0])):  # Determine the number of columns based on the length of the first row of parsed data
                col_name = st.text_input(f"Write name for column {i + 1}: ")
                columns.append(col_name)
            submitted = st.form_submit_button("Submit")
        if submitted:
            if len(columns) != len(parsed_data[0]):
                st.error(f"Number of columns entered ({len(columns)}) does not match the number of columns in the data ({len(parsed_data[0])}).")
                return pd.DataFrame()
            # Create DataFrame
            df = pd.DataFrame(parsed_data, columns=columns)
            # Apply standardization
            df['timestamp'] = df['timestamp'].apply(standardize_date)
            # Convert date to datetime format
            df['timestamp'] = df['timestamp'].apply(lambda x: parser.parse(x, fuzzy=True))
            # Sort by timestamp column in ascending order
            dataframe = df.sort_values(by='timestamp', ascending=True)
            return dataframe
        else:
            return pd.DataFrame()
    else:
        df = pd.DataFrame(parsed_data, columns=columns)
        df['timestamp'] = df['timestamp'].apply(standardize_date)
        df['timestamp'] = df['timestamp'].apply(lambda x: parser.parse(x, fuzzy=True))
        dataframe = df.sort_values(by='timestamp', ascending=True)
        return dataframe


# Identifying the hacker's vulnerability scanning period    
def filter_logs_within_1_minute(sorted_data):
    
    filtered_logs = []
    
    # Filtering while repeating data
    for i in range(len(sorted_data)):
        current_entry = sorted_data[i]
        current_datetime = current_entry['timestamp']
        current_ip = current_entry['ip_address']
        current_response_code = current_entry['status_code']
        
        # Determine if the current log has a '404' response code
        if current_response_code in ['404']:
            # Check the log within 1 second from the next log
            for j in range(i+1, len(sorted_data)):
                next_entry = sorted_data[j]
                next_datetime = next_entry['timestamp']
                next_ip = next_entry['ip_address']
                next_response_code = next_entry['status_code']
                
                # Filtering if logs within 1 second and has '404' and '200' response codes for the same IP
                if (next_datetime - current_datetime).total_seconds() <= 1 and next_ip == current_ip and next_response_code in ['404', '200']:
                    filtered_logs.append(current_entry)
                    break
    
    return filtered_logs

def get_public_ips(ip_list):
    public_ips = []
    
    for ip in ip_list:
        if not is_private_ip(ip):
            public_ips.append(ip)
    
    return public_ips
 
# Function that determines whether a private IP address exists
def is_private_ip(ip):
    private_ip_ranges = [
("10.0.0.0", "10.255.255.255"),
("172.16.0.0", "172.31.255.255"),
("192.168.0.0", "192.168.255.255")
    ]
    
    for start, end in private_ip_ranges:
        if ip_address(start) <= ip_address(ip) <= ip_address(end):
            return True
    return False
    
# Identifying web logs suspected of being used in attacks after the hacker has scanned vulnerabilities
def attacked_web_log(filtered_logs, dataframe):
    # Extracted an IP list with more than 1000 404 or 200 errors within 1 second
    filtered_logs = pd.DataFrame(filtered_logs).sort_values(by='ip_address')
    ip_counts = pd.DataFrame(filtered_logs['ip_address'].value_counts())
    ip_counts.columns = ['count']
    atk_ip = ip_counts[ip_counts['count'] >= 1000]
    atk_ip_lst = atk_ip.index.to_list()
    # Weblog data for those IPs
    atk_info = dataframe[dataframe['ip_address'].isin(atk_ip_lst)]
    # Filtering only web logs with important patterns
    ptn_plt = atk_info[atk_info['request'].str.contains("' OR '1'='1|%|passwd|cgi-bin")]
    # Filtering Weblogs Successful Attacks
    ptn_plt.loc[:, 'status_code'] = ptn_plt.loc[:, 'status_code'].astype(int)
    atk_scs = ptn_plt[ptn_plt['status_code'].apply(lambda x: eval(f"x {'==200'}"))]
    scan_frt_day = atk_scs['timestamp'].iloc[0]
    atk_scs_count = len(atk_scs)
    
    # Filter only web logs for new IPs from data values after scan and those found during scan
    scan_lst_day = atk_scs['timestamp'].iloc[-1]
    scan_b4_ip = dataframe[dataframe['timestamp'] <= scan_lst_day]['ip_address'].unique().tolist()
    for i in atk_ip_lst:
        scan_b4_ip.remove(i)
    afs = dataframe[dataframe['timestamp'] > scan_lst_day]
    new_ip_info = afs.loc[~(afs['ip_address'].isin(scan_b4_ip))]

    # Filtering Weblogs with ResponseCode == 200
    new_ip_info.loc[:, 'status_code'] = new_ip_info.loc[:, 'status_code'].astype(int)
    new_ip_info_200 = new_ip_info[new_ip_info['status_code'].apply(lambda x: eval(f"x {'==200'}"))]
   
    # Filtering web logs with payload size more than 10 times the average
    new_ip_info_200.loc[:, 'size'] = new_ip_info_200.loc[:, 'size'].astype(int)
    size_lst = new_ip_info_200['size'].tolist()
    size_avg = sum(size_lst)/len(size_lst)
    abnormal = new_ip_info_200[new_ip_info_200['size'] >= size_avg*10]

    atk_ip_count = len(atk_ip_lst)
    st.write(f"There are a total of {atk_ip_count} IPs used to scan vulnerabilities, and the IPs are as follows: {atk_ip_lst}")
    st.write(f"Vulnerability scan start point is {scan_frt_day} and end point is {scan_lst_day}. Hacker have scanned a total of {atk_scs_count} vulnerabilities.")
    
    return atk_scs, abnormal    
        
class ChatSession:
    def __init__(self, gemini_api_key, model_name="models/gemini-1.5-pro-latest"):
        # Configure the Gemini API key
        genai.configure(api_key=gemini_api_key)  # transport='' is available. you can pass a string to choose 'rest' or 'grpc' or 'grpc_asyncio'
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        self.history = []

    def send_message(self, question):
        # Create new messages, including previous conversation history
        #context = " ".join(self.history) + " " + question
        context = question
        # Generate response from the model
        response = self.model.generate_content(context)

        # Extract text from the response object
        if response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("No valid response received from the model.")
        # Append the question and response after the full response is received
        self.history.append((question, response_text))
        
        return response_text

def valid_api_key(api_key):
    if len(api_key) > 64:
        return False
    
    valid_chars = string.ascii_letters + string.digits + "!@#$%^&*()_-+={}|[]:;'<>,.?/~"
    
    for char in api_key:
        if char not in valid_chars:
            return False
    
    return True
    
def main():     
    # Start Streamlit app
    st.set_page_config(page_title="Vulnerability Analysis: Analyze your weblog data", page_icon="🔍")
    st.title("🔎 Vulnerability Analysis: Analyze your weblog data")
 
    uploaded_files = st.sidebar.file_uploader(label="Upload text files", type=["txt"], accept_multiple_files=True)

    # Check user inputs
    if not uploaded_files:
        st.info("Please upload text files to continue.")
    else:
        st.sidebar.success("Files uploaded successfully.")
        gemini_api_key = st.sidebar.text_input(label="Gemini API Key", type="password")
        # Check user inputs
        if not gemini_api_key:
            st.info("Please add your Gemini API key to continue.")
        else:
            if not valid_api_key(gemini_api_key):
                st.sidebar.error("Invalid Gemini API key. Please check and try again.")
            else:
                st.sidebar.success("Gemini API key added successfully.")

                # Process uploaded files
                dataframe = parse_uploaded_files(uploaded_files) 
                dic_df = dataframe.to_dict(orient='records')
    
                # Anomaly detected weblog
                filtered_logs = filter_logs_within_1_minute(dic_df)
    
                ### Analysis Results Output Steps ###
                #gemini_api_key = 'AIzaSyAij_2eLy-nzQVRDouKKur-1TfObHYi3E8' 
                session = ChatSession(gemini_api_key) 
                ### Default Output ###
                atk_scs, abnormal = attacked_web_log(filtered_logs, dataframe)
                ### Outputs payload analysis results during the hacker's vulnerability scan period ###
                st.write("The results of the payload analysis during the hacker's vulnerability scan are as follows.")
                lst = atk_scs['request'].tolist()
                response1 = session.send_message(f"{lst}: Please analyze those payloads") 
                st.markdown(response1)
                ### Output payload analysis results after vulnerability scan ###
                st.write("The following is the result of the payload analysis after the hacker's vulnerability scan.")
                pay_lst = abnormal['request'].tolist()
                response2 = session.send_message(f"{pay_lst}: Please analyze the payloads. If there's nothing special, let me know there isn't")
                st.markdown(response2)
                st.write("In addition, this is the result of an analysis of API security vulnerabilities after the vulnerability scan.")
                response3 = session.send_message(f"{pay_lst}: Please let me know if there are any vulnerabilities that correspond to OWASP API security TOP10. If there is nothing special, please let me know there are none")
                st.markdown(response3)

if __name__ == "__main__":
    main()