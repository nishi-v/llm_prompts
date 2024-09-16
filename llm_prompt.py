import os
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
from pathlib import Path
import litellm
litellm.set_verbose=True


# Loading environment variables
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
                        print(delta['content'], end='', flush=True)
            elif hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                    print(delta.content, end='', flush=True)
    except Exception as e:
        print(f"\nError extracting content: {e}")
    print()  # New line after complete response
    return content.strip()

def get_streaming_response(model: str, prompt: str):
    try:
        messages = [{"content": prompt, "role": "user"}]
        response = completion(model=model, messages=messages, stream=True)
        return extract_streaming_content(response)
    except Exception as e:
        print(f"Error getting response from {model}: {e}")
        return "Error"

def process_prompt(prompt):
    print(f"\nProcessing prompt: {prompt}")
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
            # Extract the numeric part of the last ID and increment it
            next_id_num = int(last_id.replace('prompt', '')) + 1
            return f"prompt{next_id_num}"
    # Default to prompt1 if the file does not exist or no Prompt IDs are found
    return "prompt1"

def main():
    input_type = input("Enter '1' to input a single prompt, or '2' to upload a CSV file: ")

    if input_type == '1':
        prompt = input("Enter your prompt: ")
        responses = process_prompt(prompt)
        
        # Define the path to the CSV file
        output_csv_path = dir / 'output/single_prompt_response.csv'
        
        # Get the next available Prompt ID
        prompt_id = get_next_prompt_id(output_csv_path)
        
        # Create DataFrame for the new prompt
        new_df = pd.DataFrame([{"Prompt ID": prompt_id, "Prompt": prompt, **responses}])
        
        if output_csv_path.exists():
            # Append to the existing CSV file
            new_df.to_csv(output_csv_path, mode='a', header=False, index=False)
        else:
            # Create a new CSV file
            new_df.to_csv(output_csv_path, index=False)
        
        print(f"Data saved to {output_csv_path}")

    elif input_type == '2':
        csv_path = input("Enter the path to your CSV file: ")
        try:
            df = pd.read_csv(csv_path)
            if 'Prompt' not in df.columns:
                raise ValueError("CSV file must have a 'Prompt' column")
            
            # Add 'Prompt ID' column if it doesn't exist
            if 'Prompt ID' not in df.columns:
                df['Prompt ID'] = [f"prompt{i+1}" for i in range(len(df))]
            
            for index, row in df.iterrows():
                prompt = row['Prompt']
                responses = process_prompt(prompt)
                for key, value in responses.items():
                    df.at[index, key] = value
            
            output_csv_path = Path(csv_path)
            
            # Ensure 'Prompt ID' is the first column
            columns = df.columns.tolist()
            columns.insert(0, columns.pop(columns.index('Prompt ID')))
            df = df[columns]

            # Save the updated DataFrame back to the CSV file
            df.to_csv(output_csv_path, index=False)
            print(f"Data saved to {output_csv_path}")

        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return

    else:
        print("Invalid input. Please run the script again and enter '1' or '2'.")
        return

if __name__ == "__main__":
    main()
