import streamlit as st
import os
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
from pathlib import Path
import litellm

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

def extract_streaming_content(response):
    content = ""
    try:
        for chunk in response:
            if isinstance(chunk, dict):
                if 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        content += delta['content']
                        print(delta['content']) 
            elif hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                    print(delta.content)
    except Exception as e:
        st.error(f"Error extracting content: {e}")
    return content.strip()

def get_streaming_response(model: str, prompt: str):
    try:
        messages = [{"content": prompt, "role": "user"}]
        response = completion(model=model, messages=messages, stream=True)
        return extract_streaming_content(response)
    except Exception as e:
        st.error(f"Error getting response from {model}: {e}")
        return "Error"

def process_prompt(prompt):
    st.write(f"Processing prompt: {prompt}")
    responses = {
        "Cohere Response": get_streaming_response("command-nightly", prompt),
        "Gemini Response": get_streaming_response("gemini/gemini-pro", prompt),
        "OpenAI Response": get_streaming_response("gpt-4", prompt),
        "Llama Response": get_streaming_response("groq/llama-3.1-70b-versatile", prompt),
        "Mixtral Response": get_streaming_response("groq/mixtral-8x7b-32768", prompt)
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

def save_single_prompt(prompt, responses):
    output_csv_path = dir / 'output/single_prompt_response.csv'
    prompt_id = get_next_prompt_id(output_csv_path)
    new_df = pd.DataFrame([{"Prompt ID": prompt_id, "Prompt": prompt, **responses}])
    
    if output_csv_path.exists():
        new_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    else:
        new_df.to_csv(output_csv_path, index=False)
    
    st.success(f"Data saved to {output_csv_path}")

def process_csv(file):
    try:
        df = pd.read_csv(file)
        if 'Prompt' not in df.columns:
            raise ValueError("CSV file must have a 'Prompt' column")
        
        if 'Prompt ID' not in df.columns:
            df['Prompt ID'] = [f"prompt{i+1}" for i in range(len(df))]

        for index, row in df.iterrows():
            prompt = row['Prompt']
            responses = process_prompt(prompt)
            for key, value in responses.items():
                df.at[index, key] = value

        output_csv_path = dir / 'output/processed_prompts.csv'
        df.to_csv(output_csv_path, index=False)
        st.success(f"Data saved to {output_csv_path}")
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")

def main():
    st.sidebar.title("Choose Input Method")
    input_method = st.sidebar.radio("Input Method", ('Single Prompt', 'Upload CSV'))

    if input_method == 'Single Prompt':
        prompt = st.text_area("Enter your prompt:")
        if st.button("Submit"):
            if prompt:
                responses = process_prompt(prompt)
                st.write("Responses:", responses)
                save_single_prompt(prompt, responses)
            else:
                st.error("Please enter a prompt.")

    elif input_method == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            if st.button("Process CSV"):
                process_csv(uploaded_file)

if __name__ == "__main__":
    main()
