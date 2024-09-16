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

# Access API keys from Streamlit secrets
COHERE_API_KEY = st.secrets["api_keys"]["COHERE_API_KEY"]
GEMINI_API_KEY = st.secrets["api_keys"]["GEMINI_API_KEY"]
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
PERPLEXITYAI_API_KEY = st.secrets["api_keys"]["PERPLEXITYAI_API_KEY"]

# Ensure LITELLM_LOG is set for logging
os.environ['LITELLM_LOG'] = 'DEBUG'


def extract_streaming_content(response):
    content = ""
    try:
        for chunk in response:
            if isinstance(chunk, dict):
                if 'choices' in chunk and chunk['choices']:
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

def save_single_prompt(prompt: str, responses: dict):
    output_csv_path = dir / 'output/single_prompt_response.csv'
    prompt_id = get_next_prompt_id(output_csv_path)
    
    # Create DataFrame for the new prompt
    new_df = pd.DataFrame([{"Prompt ID": prompt_id, "Prompt": prompt, **responses}])

    # Save the DataFrame to CSV
    if output_csv_path.exists():
        new_df.to_csv(output_csv_path, mode='a', header=False, index=False)
    else:
        new_df.to_csv(output_csv_path, index=False)

    # Load and display the content of the CSV
    df = pd.read_csv(output_csv_path)
    st.subheader("CSV Content")
    st.dataframe(df)  # Display the DataFrame in the app

    # Provide download option for the CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name='single_prompt_response.csv',
        mime='text/csv'
    )

def process_csv(file):
    try:
        df = pd.read_csv(file)
        if 'Prompt' not in df.columns:
            raise ValueError("CSV file must have a 'Prompt' column")
        
        if 'Prompt ID' not in df.columns:
            df['Prompt ID'] = [f"prompt{i+1}" for i in range(len(df))]

        st.subheader("Prompt Processing Results")
        # Process each prompt
        for index, row in df.iterrows():
            prompt = row['Prompt']
            # st.write(f"Processing prompt {index + 1}: {prompt}")
            responses = process_prompt(prompt)
            
            # Write responses to the app
            st.write("Responses:", responses)
            
            # Update the DataFrame with the responses
            for key, value in responses.items():
                df.at[index, key] = value

        # Save processed responses to a CSV
        output_csv_path = dir / 'output/processed_prompts.csv'
        df.to_csv(output_csv_path, index=False)

        # Display the CSV content in the app
        st.subheader("Processed CSV Content")
        st.dataframe(df)  # Display the processed DataFrame in the app

        # Provide download option for the processed CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name='processed_prompts.csv',
            mime='text/csv'
        )

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
