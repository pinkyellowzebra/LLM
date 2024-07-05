# -*- coding: utf-8 -*-
'''
Created on Mon Jun 24 11:14:00 2024
This code defines a ChatSession class that interacts with the Gemini AI platform using the genai library. 
It allows you to initialize a session with an API key, send questions or messages to the AI model, 
and receive responses based on the conversation context stored in self.history. 
This setup enables building a simple conversational interface powered by AI, 
where each interaction updates the context for subsequent responses.
@author: Hui, Hong'''

import google.generativeai as genai

class ChatSession:
    def __init__(self, gemini_api_key, model_name="models/gemini-1.5-pro-latest"):
        # Configure the Gemini API key
        genai.configure(api_key=gemini_api_key)  # transport='' is available. you can pass a string to choose 'rest' or 'grpc' or 'grpc_asyncio'
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        self.history = []

    def send_message(self, question):
        # Create new messages, including previous conversation history
        context = " ".join(self.history) + " " + question
        
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

# Example usage
gemini_api_key = 'YOUR_GEMINI_API_KEY'
chat_session = ChatSession(gemini_api_key)
response1 = chat_session.send_message("Hello, how are you?")
print(response1)

response2 = chat_session.send_message("What do you think about today's weather?")
print(response2)

response3 = chat_session.send_message("Do you have any plans for the weekend?")
print(response3)
