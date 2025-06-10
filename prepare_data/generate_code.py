import re
import os
import time
from openai import OpenAI

from pathlib import Path
import tempfile
import subprocess
import json
from tqdm import tqdm
import concurrent.futures
import concurrent
from rich.console import Console

console = Console()

api_key = os.getenv("OPENAI_API_KEY", "")  # Prefer environment variable for safety
model = os.getenv("OPENAI_API_MODEL", "o1-mini")

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "queries" / "queries_spring_mini.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"  / "codes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "codes_spring_mini.json"

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs" / "generate_code"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load API tasks with generated queries and signatures
with open(INPUT_FILE, 'r', encoding='utf-8') as file:
    api_tasks = json.load(file)

def run_java_code(code, spring_version, timeout=30):
    
    with tempfile.TemporaryDirectory() as temp_dir:
        gradle_content = f"""
        plugins {{
            id 'java'
        }}

        repositories {{
            mavenCentral()
        }}

        dependencies {{
            implementation 'org.springframework:spring-context:{spring_version[1:]}'
            testImplementation 'org.junit.jupiter:junit-jupiter:5.10.2'
        }}

        test {{
            useJUnitPlatform()      // Enables JUnit 5
        }}
        """

        gradle_file_path = os.path.join(temp_dir, "build.gradle")
        with open(gradle_file_path, 'w', encoding='utf-8') as f:
            f.write(gradle_content)

        os.makedirs(os.path.join(temp_dir, "src", "main", "java"), exist_ok=True)
        java_file_path = os.path.join(temp_dir, "src", "main", "java", "ExampleSpringService.java")
        with open(java_file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        try:
            cmd = f'cd "{temp_dir}" && gradle build'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # 获取详细输出用于日志
            detailed_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            
            if result.returncode != 0:
                return False, detailed_output
            
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Timeout: Test took too long to execute"
        except Exception as e:
            return False, f"Error running tests: {str(e)}"


# Prompt template for code generation with error feedback
def create_code_prompt(query, signature, api_details, error_feedback=None):
    api_version = api_details.get('to_version', 'latest')
    # rust_version = '1.84.0'
    api_name = api_details.get('name', 'the specified API')
    api_detail = {
    'api_name': api_details.get('name', ''),
    # 'crate_version': api_details.get('to_version', ''),
    'signature': api_details.get('signature', ''),
    'documentation': api_details.get('documentation', ''),
    'source_code': api_details.get('source_code', ''),
}

    base_prompt = (f"Given the following Java programming task:\n"
                   f"{query}\n\n"
                   f"Implement the following function signature:\n"
                   f"{signature}\n\n"
                   f"### Important rules ###\n"
                   f"- Your implementation must compile and run correctly on Spring version '{api_version}'.\n"
                   f"- Your implementation MUST use '{api_name}'.\n"
                   f"- Provide a complete, concise, and correct Java implementation.\n"
                   f"- Relevant API Details:\n"
                   f"{api_detail}\n\n"
                   f"DON'T NEED EXPLANATION, JUST GIVE THE CODE.\n\n"
                   f"### Code Format ###\n"
                   f"<code>\n"
                   f"[All needed imports]\n"
                   f"\n"
                   f"""public class ExampleSpringService {{
                   [Your code here]
                   }}\n"""
                   f"</code>\n")

    if error_feedback:
        base_prompt += f"\n\nYour previous attempt resulted in the following error:\n{error_feedback}\nPlease correct the issues and ensure the API is used correctly."

    return base_prompt

def is_api_properly_used(code, s):

    api_name = None

    if '#' not in s:
        api_name = s.rsplit('.', 1)[-1]
    else:
        api_name = re.search(r'#([^(]+)\(', s).group(1)

    # if re.search(r'#([^(]+)\(', s) is None:
    #     console.print(f"  [red]No API name found in the signature: {s}[/]")
    console.print(f"  [blue]API name: {api_name}[/]")

    # Escape special regex characters in API name
    escaped_api = re.escape(api_name)
    
    # Find all comments in the code
    comments = re.findall(r'//.*$|/\*[\s\S]*?\*/', code, re.MULTILINE)
    
    # Remove comments from code for checking actual usage
    code_without_comments = code
    for comment in comments:
        code_without_comments = code_without_comments.replace(comment, '')
    
    # Check if API is mentioned outside of comments
    api_pattern = r'(?<![a-zA-Z0-9_])' + escaped_api + r'(?![a-zA-Z0-9_])'
    return bool(re.search(api_pattern, code_without_comments))

def static_analysis_java_code(code, api_name):
    
    api_used = is_api_properly_used(code, api_name)
    if not api_used:
        return False, f"Code does not properly use the required API: '{api_name}'"
    
    syntax_checks = [
        (r'\bpublic\b', "Missing function definition"),
        (r'[{]', "Missing opening braces"),
        (r'[}]', "Missing closing braces"),
    ]
    
    for pattern, error in syntax_checks:
        if not re.search(pattern, code):
            return False, error
    
    code_without_lifetimes = re.sub(r"<'[a-zA-Z_]+>|&'[a-zA-Z_]+", "<LIFETIME>", code)
    
    quotes = code_without_lifetimes.count('"') % 2
    single_quotes = code_without_lifetimes.count("'") % 2
    parentheses = code.count('(') - code.count(')')
    braces = code.count('{') - code.count('}')
    brackets = code.count('[') - code.count(']')
    
    if quotes != 0:
        return False, "Unclosed double quotes"
    if single_quotes != 0:
        return False, "Unclosed single quotes (not related to lifetimes)"
    if parentheses != 0:
        return False, "Mismatched parentheses"
    if braces != 0:
        return False, "Mismatched braces"
    if brackets != 0:
        return False, "Mismatched brackets"

    return True, "Static analysis passed"

def generate_and_validate_java_code(task_entry, max_retries=3):
    # Get necessary information from task entry
    api_version = task_entry.get('to_version', 'latest')
    
    query = task_entry.get('query', '')
    signature = task_entry.get('function_signature', '')
    api_name = task_entry.get('name', 'the specified API')
    
    # Skip if query or signature is missing or marked as error
    if not query or not signature or query.startswith("ERROR:") or signature.startswith("ERROR:"):
        task_entry['code'] = "ERROR: Missing or invalid query/signature"
        task_entry['validation_status'] = "skipped"
        task_entry['validation_output'] = "Invalid query/signature"
        return task_entry

    error_feedback = None
    for attempt in range(max_retries):
        console.print(f"Attempt {attempt+1}/{max_retries} for API: {api_name}")
        
        # Create prompt with any error feedback from previous attempts
        prompt = create_code_prompt(query, signature, task_entry, error_feedback)
        
        try:
            # Call the AI model to generate code
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Rust expert tasked with writing high-quality, correct Rust code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3 + (attempt * 0.1)  # Slightly increase temperature for diversity in retries
            )

            response = completion.choices[0].message.content.strip()
            
            task_entry['code'] = "CANNOT GENERATE CORRECT CODE"

            LOGS_CUR_DIR = LOGS_DIR / f"generate_code_spring_mini_{api_name}"
            LOGS_CUR_DIR.mkdir(parents=True, exist_ok=True)
            LOG_FILE = LOGS_CUR_DIR / f"attempt_{attempt+1}.log"

            # Extract code from response
            if "```" in response:
                # Extract code between triple backticks
                code_blocks = re.findall(r'```(?:java)?(.*?)```', response, re.DOTALL)
                if code_blocks:
                    java_code = code_blocks[0].strip()
                else:
                    error_feedback = "Please provide code within backticks or '<code> [Your code here] </code>' tags."
                    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
                        log_file.write(f"Failed to extract code from the response:\n{response}\n")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        task_entry['code'] = "CANNOT GENERATE CORRECT CODE"
                        task_entry['validation_status'] = "failed"
                        task_entry['validation_output'] = "Invalid response format"
                        return task_entry
            elif "<code>" in response and "</code>" in response:
                java_code = response.split('<code>')[1].split('</code>')[0].strip()
            else:
                error_feedback = "Please provide code within backticks or '<code> [Your code here] </code>' tags."
                console.print(f"  Missing code tags in response, retrying...")
                with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
                    log_file.write(f"Missing code tags in response:\n{response}\n")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    task_entry['code'] = "CANNOT GENERATE CORRECT CODE"
                    task_entry['validation_status'] = "failed"
                    task_entry['validation_output'] = "Invalid response format"
                    return task_entry
            
            # Check if code is too short/empty
            if len(java_code) < 20:  # Arbitrary minimum length for valid code
                error_feedback = "The generated code is too short or empty. Please provide a complete implementation."
                console.print(f"  Code too short ({len(java_code)} chars), retrying...")
                with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
                    log_file.write(f"Code too short ({len(java_code)} chars):\n{java_code}\n")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            # console.print the code for debugging
            # console.print(f" Rust code:\n {rust_code}")
            
            # 使用静态分析代替Docker运行
            # console.print(f"  Running static analysis on the generated code...")
            validation_passed, validation_message = static_analysis_java_code(java_code, api_name)

            run_passed, run_output = run_java_code(java_code, api_version)

            with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
                log_file.write(f"Java code:\n{java_code}\n")
                log_file.write(f"Validation passed:\n{validation_passed}\n")
                log_file.write(f"Validation message:\n{validation_message}\n")
                log_file.write(f"Run passed:\n{run_passed}\n")
                log_file.write(f"Run output:\n{run_output}\n")
            
            # console.print(f"  Validation passed: {validation_passed}")
            # console.print(f"  Run passed: {run_passed}")
            # console.print(f"  Validation message: {validation_message}")
            # console.print(f"  Run output: {run_output}")
            task_entry['code'] = java_code

            if validation_passed and run_passed:
                # 成功：代码通过静态分析
                console.print(f"  [green]Success![/] Code passed static analysis and uses the API correctly: {api_name}")
                task_entry['validation_output'] = validation_message
                task_entry['validation_status'] = "success"
                return task_entry
            elif not validation_passed:
                # 代码未通过静态分析
                error_feedback = f"Static analysis failed: {validation_message}"
                console.print(f"  Code failed static analysis: {validation_message}")
            elif not run_passed:
                # 代码未通过运行
                error_feedback = f"Code failed to run: {run_output}"
                console.print(f"  Code failed to run")
                # console.print(f"  {error_feedback}")
            
            # 继续下一次尝试
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except Exception as e:
            # Handle any exceptions in the API call
            error_feedback = f"Error generating code: {str(e)}"
            console.print(f"  Exception: {str(e)}, retrying...")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # All retries failed, return the last attempt with error status and CANNOT GENERATE CORRECT CODE message
    task_entry['validation_output'] = error_feedback
    task_entry['validation_status'] = "failed"
    console.print(f"  All {max_retries} attempts [red]failed[/] for API: {api_name}")
    # console.print(f"  Error feedback: {error_feedback}")
    return task_entry

def generate_all_validated_codes(api_tasks):
    updated_tasks = []
    progress_bar = tqdm(total=len(api_tasks), desc="Generating validated code")
    
    for idx in range(0, len(api_tasks), 50):
        batch = api_tasks[idx:idx+50]
        batch_results = [None] * len(batch)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {
                executor.submit(generate_and_validate_java_code, entry): i 
                for i, entry in enumerate(batch)
            }
            
            for future in concurrent.futures.as_completed(future_to_index.keys()):
                try:
                    original_index = future_to_index[future]
                    result = future.result()
                    batch_results[original_index] = result
                    
                    # 添加详细的失败信息输出
                    # if result.get('validation_status') == "failed":
                    #     console.print(f"\nFAILURE DETAILS for API {result.get('name')}:")
                    #     console.print(f"  Query: {result.get('query')}")
                    #     console.print(f"  Function signature: {result.get('function_signature')}")
                    #     console.print(f"  Generated code snippet:\n{result.get('code')[:200]}..." if len(result.get('code', '')) > 200 else f"  Generated code:\n{result.get('code')}")
                    #     console.print(f"  Error details: {result.get('validation_output')}\n")
                    
                    progress_bar.update(1)
                except Exception as e:
                    console.print(f"Error processing task: {str(e)}")
        
        # Add batch results in original order
        updated_tasks.extend(batch_results)
        
        # Save progress after each batch
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(updated_tasks, outfile, indent=2, ensure_ascii=False)
    
    progress_bar.close()
    console.print(f"Processing complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    console.print(f"Starting code generation for {len(api_tasks)} tasks...")
    console.print(f"Using static analysis instead of Docker for code validation")
    generate_all_validated_codes(api_tasks)