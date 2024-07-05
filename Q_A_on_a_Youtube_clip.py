# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:30:56 2024

@author: Hui, Hong
This code uses Hugging Face's Transformers library to answer questions about YouTube video clips using LangChain. 
Before running this code, you must install the Transformers library.
"""

!pip install google-api-python-client google-auth-oauthlib transformers
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from transformers import pipeline

# YouTube API settings
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

# YouTube API authentication
flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
credentials = flow.run_console()
youtube = build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

# Load the question-answering model using LangChain
qa_pipeline = pipeline("question-answering", model="mrm8488/bert-multi-cased-finetuned-xquadv1", tokenizer="mrm8488/bert-multi-cased-finetuned-xquadv1")

def get_video_subtitles(video_id):
    # Extract subtitles from the YouTube video clip using the YouTube API
    captions = youtube.captions().list(part="snippet", videoId=video_id).execute()
    if captions["items"]:
        caption_id = captions["items"][0]["id"]
        subtitle_text = youtube.captions().download(id=caption_id).execute()
        return subtitle_text["body"]
    else:
        return None

def main():
    # Input the YouTube video clip ID
    video_id = input("Enter YouTube video ID: ")
    
    # Extract subtitles from the YouTube video clip
    subtitle_text = get_video_subtitles(video_id)
    if subtitle_text:
        # Prompt the user to enter a question
        question = input("Enter your question: ")
        
        # Use LangChain to extract the answer to the question
        answer = qa_pipeline(question=question, context=subtitle_text)
        
        # Print the extracted answer
        print("Answer:", answer["answer"])
    else:
        print("Subtitles not available for this video.")

if __name__ == "__main__":
    main()
