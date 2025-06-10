import json
import os
import subprocess
import tempfile
import re
import sys
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from prepare_data.generate_tests import run_java_code
from prepare_data.generate_code import is_api_properly_used

from rich.console import Console

console = Console()

API_KEY = os.getenv("OPENAI_API_KEY", "")  # Prefer environment variable for safety
BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# Define available models
MODELS = [
    # "gpt-4o",
    # "o1-mini",
    "gpt-4.1",
    # "gemini-1.5-pro",
    # "claude-3-5-sonnet-20240620",
    # "qwen2.5-72b-instruct",
    # "Llama-3.1-70b",
    # "deepseek-v3",
    # "grok-3"
]

def call_LLM(prompt: str, model: str, api_key: str) -> str:
    """Call the LLM API and return the response."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )        
        code = response.choices[0].message.content.strip()
        return code
        
    except Exception as e:
        print(f"Error calling LLM {model}: {str(e)}")
        return ""

def extract_java_code(response: str) -> str:
    """Extract Java code from the LLM response."""
    java_pattern = r"```(?:java)?\s*([\s\S]*?)```"
    matches = re.findall(java_pattern, response)
    if not matches:
        return response
    return matches[0].strip()
    

def extract_api_info(query: str, api_info: Dict[str, Any], type_info: str = "none") -> Dict[str, Any]:
    api_name = api_info.get("name", "")
    api_module = api_info.get("module", "")
    api_signature = api_info.get("signature", "")
    api_documentation = api_info.get("documentation", "")
    from_version = api_info.get("from_version", "")
    to_version = api_info.get("to_version", "")
    source_code = api_info.get("source_code", "")

    if type_info == "none":
        return f"""
    Relevant API Information:
    - API Name: {api_name}
    - API Module: {api_module}
    """
    elif type_info == "api":
        return f"""
    Relevant API Information:
    - API Name: {api_name}
    - API Module: {api_module}
    - API Signature: {api_signature}
    - API Documentation: {api_documentation}
    - API Source Code: {source_code}
    - API Changed From Version: {from_version}
    - API Changed To Version: {to_version}
    """
    else: 
        raise ValueError(f"Invalid type_info: {type_info}")


def get_code_generation_prompt(query: str, api_info: Dict[str, Any], function_signature: str, type_info: str = "none") -> str:
    """Create a prompt for code generation with the specified information."""
    
    spring_version = api_info.get("to_version", "")
    api_name = api_info.get("name", "")
    
    extracted_api_info = extract_api_info(query, api_info, type_info)


    prompt = f"""
    You are an expert Java programmer. Write a Java function implementation for the following task:

    Task Description:
    {query}

    Required Function Signature:
    ```java
    {function_signature}
    ```

    {extracted_api_info}

    Requirements:
    1. Implement ONLY the function with the given signature, no additional functions.
    2. Your implementation MUST use the specified API: {api_name}
    3. Make sure your code is compatible with Spring version {spring_version}
    3. Do not include tests, main function, or any code outside the required function.
    4. Do not include additional comments or explanations.

    Respond with ONLY the Java function implementation, nothing else.

    ### Code Format ###
    ```java
    [All needed imports]

    public class ExampleSpringService {{
    [Your code here]
    }}
    ```
    """             

    return "Query:\n" + prompt

def gen_fix_prompt(previous_context: str, code: str, error_message: str) -> str:
    return f"""
    {previous_context}

    Your response:
    {code}

    Error message:
    {error_message}

    Please fix the error and return the corrected code.
    """

results = []

def process_task(pbar, task: Dict[str, Any], model: str, output_file: str, api_key: str, base_url: str, java_files_dir: str = None, iters : int = 10, type_info: str = "none") -> Dict[str, Any]:
    """Process a single task for a specific model."""
    result = task.copy()  # Start with a copy of the original task data
    

    # Extract required fields
    query = task.get("query", "")
    function_signature = task.get("function_signature", "")
    test_code = task.get("test_program", "")
    task_idx = task.get("task_idx", "")
    spring_version = task.get("to_version", "")
    api_name = task.get("name", "")
    
    # Generate code
    prompt = get_code_generation_prompt(query, task, function_signature, type_info)
    raw_response = call_LLM(prompt, model, api_key)
    code = extract_java_code(raw_response)
    
    # Check function signature
    if not is_api_properly_used(code, api_name):
        result[f"{model}_code"] = "INCORRECT SIG"
        result[f"{model}_test_result"] = "FAILED"
        return result

    success, error_message = run_java_code(code, test_code, spring_version, java_files_dir)
    
    previous_context = gen_fix_prompt(prompt, code, error_message)
    it_idx = 0
    while not success and it_idx < iters:
        it_idx += 1
        os.makedirs(os.path.join(java_files_dir, "prompts"), exist_ok=True)
        prompts_file_path = os.path.join(java_files_dir, "prompts", f"test_{task_idx}_{it_idx:02d}.prompt")
        with open(prompts_file_path, 'w', encoding='utf-8') as f:
            f.write(previous_context)
        raw_response = call_LLM(previous_context, model, api_key)
        code = extract_java_code(raw_response)
        success, error_message = run_java_code(code, test_code, spring_version, java_files_dir)
        previous_context = gen_fix_prompt(previous_context, code, error_message)

    result[f"{model}_code"] = code
    result[f"{model}_test_result"] = "SUCCESS" if success else "FAILED"

    if not success:
        os.makedirs(os.path.join(java_files_dir, "error_logs"), exist_ok=True)
        java_file_path = os.path.join(java_files_dir, "error_logs", f"test_{task_idx}.error")
        with open(java_file_path, 'w', encoding='utf-8') as f:
            f.write(error_message)
    
    pbar.update(1)
    results.append(result)
    if len(results) % 5 == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return result

def process_all_models(file_data: List[Dict[str, Any]], models: List[str], 
                      api_key: str, base_url: str, output_file: str, max_workers: int = 4, java_files_dir: str = None, type_info: str = "none"):
    """Process all tasks for all models in parallel."""

    print("processing all models")
    global results
    
    # Ensure each task has a unique 'task_idx' field
    remaining_tasks = []
    for idx, task in enumerate(file_data):
        task = task.copy()
        if 'task_idx' not in task:
            task['task_idx'] = str(idx)
        remaining_tasks.append(task)
    
    with tqdm(total=len(remaining_tasks) * len(models), desc="Processing tasks") as pbar:
        for model in models:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map each submitted Future to its originating task so we can
                # recover the original data when handling successes or errors.
                futures = {
                    executor.submit(
                        process_task,
                        pbar,
                        task,
                        model,
                        output_file,
                        api_key,
                        base_url,
                        java_files_dir,
                        type_info=type_info,
                    ): task
                    for task in remaining_tasks
                }
                
                for future in as_completed(futures):
                    task = futures[future]  # Retrieve the task associated with this future
                    task_result = task.copy()
                    try:
                        task_result = future.result()
                    except Exception as e:
                        task_result[f"{model}_code"] = f"ERROR: {str(e)}"
                        task_result[f"{model}_test_result"] = "FAILED"  
                        pbar.update(1)
                        results.append(task_result)
                        if len(results) % 5 == 0:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete. Results saved to: {output_file}")
    
    for model in models:
        success_count = sum(1 for result in results if result.get(f"{model}_test_result", "") == "SUCCESS")
        success_rate = (success_count / len(results)) * 100 if results else 0
        incorrect_sig = sum(1 for result in results if result.get(f"{model}_code", "") == "INCORRECT SIG")
        incorrect_api = sum(1 for result in results if result.get(f"{model}_code", "") == "INCORRECT API")
        
        print(f"\nModel: {model}")
        print(f"Success rate: {success_rate:.2f}% ({success_count}/{len(results)})")
        print(f"Incorrect signatures: {incorrect_sig}")
        print(f"Incorrect API usage: {incorrect_api}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM models on Java API evolution tasks")
    parser.add_argument("--input_file", required=True, help="Input JSON file with tasks and test programs")
    parser.add_argument("--output_file", required=True, help="Output JSON file for results")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of concurrent workers")
    parser.add_argument("--api_key", help="API key for LLM service", default=API_KEY)
    parser.add_argument("--base_url", help="Base URL for LLM service", default=BASE_URL)
    parser.add_argument("--java_files_dir", help="Directory to save Java test files")
    parser.add_argument("--type_info", help="Type of information to extract from API information", default="none")
    args = parser.parse_args()
    
    print(args)

    # Load the data from files
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            
        print(f"Loaded {len(file_data)} tasks from file")
    except Exception as e:
        print(f"Error loading input files: {str(e)}")
        sys.exit(1)
    
    # Process all models
    process_all_models(
        file_data, 
        args.models, 
        args.api_key, 
        args.base_url, 
        args.output_file, 
        args.max_workers, 
        args.java_files_dir,
        args.type_info
    )

if __name__ == "__main__":
    main()