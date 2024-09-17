import streamlit as st
import os
import pandas as pd
from pathlib import Path
import json
import asyncio
from dotenv import load_dotenv
import litellm
from litellm import completion
import datetime

# Set verbosity for litellm
litellm.set_verbose = True

st.title("LLM Prompting")

# Load environment variables
dir = Path(os.getcwd())
load_dotenv(dir / '.env')

COHERE_API_KEY = os.environ['COHERE_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GROQ_API_KEY = os.environ['GROQ_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

os.environ['LITELLM_LOG'] = 'DEBUG'

# Initialize session state for conversation history and end flag
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'conversation_ended' not in st.session_state:
    st.session_state.conversation_ended = False

def extract_streaming_content(response):
    content = ""
    buffer = ""
    try:
        for chunk in response:
            if isinstance(chunk, dict) and 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    buffer += delta['content']
            elif hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    buffer += delta.content
            
            if buffer.endswith((' ', '.', '!', '?', '\n')):
                content += buffer
                buffer = ""
    except Exception as e:
        st.error(f"Error extracting content: {e}")
    
    if buffer:
        content += buffer
    
    return content.strip()

def get_streaming_response_sync(model: str, messages: list):
    try:
        response = completion(model=model, messages=messages, stream=True)
        return extract_streaming_content(response)
    except litellm.ServiceUnavailableError as e:
        st.error(f"Service unavailable for model {model}: {e}")
        return f"Service unavailable for model {model}. Please try again later."
    except Exception as e:
        st.error(f"Error getting response from {model}: {e}")
        return "Error"

async def process_prompt(prompt):
    messages = [{"content": p['Prompt'], "role": "user"} for p in st.session_state.conversation_history] + [{"content": prompt, "role": "user"}]
    responses = {
        "Cohere Response": get_streaming_response_sync("command-nightly", messages),
        "Gemini Response": get_streaming_response_sync("gemini/gemini-pro", messages),
        "OpenAI Response": get_streaming_response_sync("gpt-4", messages),
        "Llama Response": get_streaming_response_sync("groq/llama-3.1-70b-versatile", messages),
        "Mixtral Response": get_streaming_response_sync("groq/mixtral-8x7b-32768", messages)
    }
    return responses

def display_responses(responses):
    st.write("Responses:")
    st.json(responses)

def save_conversation(conversation_history, timestamp):
    output_csv_path = dir / f'conversation_output_{timestamp}.csv'
    output_js_path = dir / f'conversation_output_{timestamp}.js'
    
    # Prepare DataFrame for CSV and JS
    df = pd.DataFrame(conversation_history)
    
    # Save as CSV
    df.to_csv(output_csv_path, index=False)
    st.subheader("Conversation CSV Content")
    st.dataframe(df)
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Conversation CSV",
        data=csv_data,
        file_name=f'conversation_output_{timestamp}.csv',
        mime='text/csv'
    )
    
    # Save as JS
    js_data = df.to_dict(orient='records')
    with open(output_js_path, 'w') as f:
        json.dump(js_data, f, indent=4)
    
    st.subheader("Conversation JS Content")
    st.text_area("JS Content", value=json.dumps(js_data, indent=4), height=400, max_chars=None)
    
    js_data_encoded = json.dumps(js_data, indent=4).encode('utf-8')
    st.download_button(
        label="Download Conversation JS",
        data=js_data_encoded,
        file_name=f'conversation_output_{timestamp}.js',
        mime='application/javascript'
    )

async def process_csv(file):
    try:
        df = pd.read_csv(file)
        if 'Prompt' not in df.columns:
            raise ValueError("CSV file must have a 'Prompt' column")
        
        if 'Prompt ID' not in df.columns:
            df['Prompt ID'] = [f"prompt{i+1}" for i in range(len(df))]

        st.subheader("Prompt Processing Results")
        for index, row in df.iterrows():
            prompt = row['Prompt']
            st.write("Processing Prompt: ", prompt)
            responses = await process_prompt(prompt)
            
            st.write("Responses:", responses)
            
            for key, value in responses.items():
                df.at[index, key] = value

        output_csv_path = dir / 'output/processed_prompts.csv'
        output_js_path = dir / 'output/processed_prompts.js'
        df.to_csv(output_csv_path, index=False)

        st.subheader("Processed CSV Content")
        st.dataframe(df)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name='processed_prompts.csv',
            mime='text/csv'
        )

        # Save as JS
        js_data = df.to_dict(orient='records')
        with open(output_js_path, 'w') as f:
            json.dump(js_data, f, indent=4)
        
        # Display JS Content
        st.subheader("Processed JS Content")
        st.text_area("JS Content", value=json.dumps(js_data, indent=4), height=400, max_chars=None)

        # Provide download button after displaying content
        js_data_encoded = json.dumps(js_data, indent=4).encode('utf-8')
        st.download_button(
            label="Download Processed JS",
            data=js_data_encoded,
            file_name='processed_prompts.js',
            mime='application/javascript'
        )

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")

async def process_js_file(file):
    try:
        js_data = json.load(file)
        if not isinstance(js_data, list):
            raise ValueError("JS file must contain a list of prompts")
        
        df = pd.DataFrame(js_data)
        if 'Prompt' not in df.columns:
            raise ValueError("JS file must have a 'Prompt' column")
        
        if 'Prompt ID' not in df.columns:
            df['Prompt ID'] = [f"prompt{i+1}" for i in range(len(df))]

        st.subheader("Prompt Processing Results")
        for index, row in df.iterrows():
            prompt = row['Prompt']
            st.write("Processing Prompt: ", prompt)
            responses = await process_prompt(prompt)
            
            st.write("Responses:", responses)
            
            for key, value in responses.items():
                df.at[index, key] = value

        output_csv_path = dir / 'output/processed_prompts_from_js.csv'
        output_js_path = dir / 'output/processed_prompts_from_js.js'
        df.to_csv(output_csv_path, index=False)

        st.subheader("Processed CSV Content")
        st.dataframe(df)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name='processed_prompts_from_js.csv',
            mime='text/csv'
        )

        # Save as JS
        js_data = df.to_dict(orient='records')
        with open(output_js_path, 'w') as f:
            json.dump(js_data, f, indent=4)
        
        # Display JS Content
        st.subheader("Processed JS Content")
        st.text_area("JS Content", value=json.dumps(js_data, indent=4), height=400, max_chars=None)

        # Provide download button after displaying content
        js_data_encoded = json.dumps(js_data, indent=4).encode('utf-8')
        st.download_button(
            label="Download Processed JS",
            data=js_data_encoded,
            file_name='processed_prompts_from_js.js',
            mime='application/javascript'
        )

    except Exception as e:
        st.error(f"Error processing JS file: {e}")

async def main():
    st.sidebar.title("Input Method")
    input_method = st.sidebar.radio("Choose an input method:", ['Prompt', 'Upload CSV', 'Upload JS'])
    
    if input_method == 'Prompt':
        prompt = st.text_area("Enter your prompt:")
        if st.button("Submit Prompt"):
            st.write("Processing prompt: ", prompt)
            responses = await process_prompt(prompt)
            display_responses(responses)
            st.session_state.conversation_history.append({"Prompt": prompt, **responses})
        
        if st.button("End Conversation"):
            if st.session_state.conversation_history:
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                save_conversation(st.session_state.conversation_history, timestamp)
            st.session_state.conversation_ended = True
            st.session_state.conversation_history = []
            st.success("Conversation ended and history cleared.")

    elif input_method == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            await process_csv(uploaded_file)

    elif input_method == 'Upload JS':
        uploaded_js_file = st.file_uploader("Upload JS file", type=["js"])
        if uploaded_js_file:
            await process_js_file(uploaded_js_file)

if __name__ == "__main__":
    asyncio.run(main())