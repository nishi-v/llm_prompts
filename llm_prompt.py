import os
import pandas as pd # type: ignore
from litellm import completion
from dotenv import load_dotenv
from pathlib import Path

# Loading environment variables
dir = Path(os.getcwd())
load_dotenv(dir/'.env')

COHERE_API_KEY = os.environ['COHERE_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']

# Take message content as user input
user_message = input("Enter your message: ")

messages = [{"content": user_message, "role": "user"}]

# Getting responses
response = completion(model="command-nightly", messages=messages)
cohere_response = response.choices[0].message.content

response = completion(model="gemini/gemini-pro", messages=messages)
gemini_response = response.choices[0].message.content

# Extract the message content
message_content = messages[0]['content']

# Data for DataFrame
data = {
    "Message": [message_content],
    "Cohere Response": [cohere_response],
    "Gemini Response": [gemini_response]
}

# Creating DataFrame
df = pd.DataFrame(data)

# CSV file path
csv_file_path = dir / 'output/responses.csv'

# Append to CSV file
if csv_file_path.exists():
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")
