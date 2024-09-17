import streamlit as st
import os
import pandas as pd
from pathlib import Path
import json
import asyncio
from dotenv import load_dotenv
import litellm
from litellm import completion

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

def extract_streaming_content(response):
    content = ""
    try:
        for chunk in response:
            if isinstance(chunk, dict) and 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    content += delta['content']
                    print(delta['content'])  # Display on Streamlit UI
            elif hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                    print(delta.content)  # Display on Streamlit UI
    except Exception as e:
        st.error(f"Error extracting content: {e}")
    return content.strip()

def get_streaming_response_sync(model: str, prompt: str):
    try:
        messages = [{"content": prompt, "role": "user"}]
        response = completion(model=model, messages=messages, stream=True)
        return extract_streaming_content(response)
    except litellm.ServiceUnavailableError as e:
        st.error(f"Service unavailable for model {model}: {e}")
        return f"Service unavailable for model {model}. Please try again later."
    except Exception as e:
        st.error(f"Error getting response from {model}: {e}")
        return "Error"

async def process_prompt(prompt):
    st.write(f"Processing prompt: {prompt}")
    responses = {
        "Cohere Response": get_streaming_response_sync("command-nightly", prompt),
        "Gemini Response": get_streaming_response_sync("gemini/gemini-pro", prompt),
        "OpenAI Response": get_streaming_response_sync("gpt-4", prompt),
        "Llama Response": get_streaming_response_sync("groq/llama-3.1-70b-versatile", prompt),
        "Mixtral Response": get_streaming_response_sync("groq/mixtral-8x7b-32768", prompt)
    }
    return responses

def get_next_prompt_id(csv_path: Path) -> str:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'Prompt ID' in df.columns:
            last_id = df['Prompt ID'].iloc[-1]
            next_id_num = int(last_id.replace('prompt', '')) + 1
            return f"prompt{next_id_num}"
    return "prompt1"

def save_single_prompt(prompt: str, responses: dict):
    output_csv_path = dir / 'single_prompt_response.csv'
    output_js_path = dir / 'single_prompt_response.js'
    
    if not dir.exists():
        os.makedirs(dir)
    
    new_df = pd.DataFrame([{"Prompt ID": get_next_prompt_id(output_csv_path), "Prompt": prompt, **responses}])

    if output_csv_path.exists():
        new_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    else:
        new_df.to_csv(output_csv_path, index=False)

    df = pd.read_csv(output_csv_path)
    st.subheader("CSV Content")
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name='single_prompt_response.csv',
        mime='text/csv'
    )

    # Save as JS
    js_data = df.to_dict(orient='records')
    with open(output_js_path, 'w') as f:
        json.dump(js_data, f, indent=4)
    
    # Check if JS file was created
    if output_js_path.exists():
        st.subheader("JS Content")
        with open(output_js_path, 'r') as f:
            js_content = f.read()
        st.text_area("JS Content", value=js_content, height=400, max_chars=None)  # Adjust height as needed

        js_data_encoded = json.dumps(js_data, indent=4).encode('utf-8')
        st.download_button(
            label="Download JS",
            data=js_data_encoded,
            file_name='single_prompt_response.js',
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
    st.sidebar.title("Choose Input Method")
    input_method = st.sidebar.radio("Input Method", ('Single Prompt', 'Upload CSV', 'Upload JS'))

    if input_method == 'Single Prompt':
        prompt = st.text_area("Enter your prompt:")
        if st.button("Submit"):
            if prompt:
                responses = await process_prompt(prompt)
                st.write("Responses:", responses)
                save_single_prompt(prompt, responses)
            else:
                st.error("Please enter a prompt.")

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
