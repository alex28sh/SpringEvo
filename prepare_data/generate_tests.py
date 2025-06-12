
import json
import os
import subprocess
import tempfile
import time
import concurrent.futures
import re
from typing import Dict, List, Any, Tuple

from tqdm import tqdm
from openai import OpenAI

from pathlib import Path
from rich.console import Console

console = Console()

api_key = os.getenv("OPENAI_API_KEY", "")  # Prefer environment variable for safety
model = os.getenv("OPENAI_API_MODEL", "o1-mini")

INPUT_FILE = Path(__file__).resolve().parent.parent / "data_filtered" / "codes" / "codes_spring_success_mini.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data_filtered"  / "tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "tests_spring_mini.json"

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs" / "generate_tests"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_code(code: str, signature: str, query: str, max_retries: int = 3) -> str:
    """Generate test code using OpenAI model."""
    
    prompt = f"""
You are an expert Rust developer tasked with writing tests for a Java function.

Here is the function signature:
```java
{signature}
```

Here is the query describing the function:
{query}

Here is the function code:
```java
{code}
```

Your task is to write Java test code that thoroughly tests this function.
DO NOT include the original function code in your response.
ONLY provide the test code that would be placed in a test module.

IMPORTANT GUIDELINES FOR WRITING TESTS:
1. Make sure to include all necessary imports and crates at the top of your test
2. When testing generic functions, provide concrete type implementations
3. For functions with complex types, properly handle construction and error cases
4. Make sure your test references match the function signature exactly
5. Do not use non-deterministic functions like random number generation in tests without fixed seeds
6. Make sure you import all necessary test libraries (like `import static org.junit.jupiter.api.Assertions.assertEquals`, if needed)

Respond ONLY with the Java test code, nothing else.

### Code Format ###
```java
import org.junit.jupiter.api.Test;
[All needed imports]

public class ExampleSpringServiceTest {{
[Your code here]
}}
```
"""
    client = OpenAI(api_key=api_key)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            test_code = response.choices[0].message.content.strip()
            
            # Remove markdown code block delimiters if they exist
            if test_code.startswith("```"):
                # Extract content between code blocks
                pattern = r"```(?:java)?\s*([\s\S]*?)```"
                matches = re.findall(pattern, test_code)
                if matches:
                    test_code = "\n".join(matches)
                else:
                    # If regex didn't work, try a simple trim approach
                    test_code = test_code.replace("```java", "").replace("```", "").strip()
            
            return test_code
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting OpenAI response, retrying ({attempt+1}/{max_retries}): {e}")
                time.sleep(2)  # Adding a delay before retrying
            else:
                print(f"Failed to get response from OpenAI after {max_retries} attempts: {e}")
                return ""
    
    return ""

def run_java_code(code, test_code, spring_version, timeout=60):
    
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

        os.makedirs(os.path.join(temp_dir, "src", "test", "java"), exist_ok=True)
        test_file_path = os.path.join(temp_dir, "src", "test", "java", "ExampleSpringServiceTest.java")
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_code)

        try:
            cmd = f'cd "{temp_dir}" && gradle test'
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


def fix_test_with_feedback(code: str, test_code: str, error_message: str, signature: str, query: str) -> str:
    """Send error feedback to OpenAI to get improved test code."""
    
    prompt = f"""
You are an expert Java developer tasked with fixing failing test code.

Original function signature:
```java
{signature}
```

Function description:
{query}

Function code:
```java
{code}
```

Original test code:
```java
{test_code}
```

Compilation/test errors:
```
{error_message}
```

Please fix the test code to address the specific errors. Pay special attention to:
1. Making sure all imports are correct
2. Types match exactly with the function signature
3. Test values are valid for the expected types
4. Properly handling expected exceptions
5. Using correct syntax for the testing Spring and Java version

DO NOT include the original function implementation in your response.
ONLY provide the corrected test code.
Respond ONLY with the fixed Java test code, nothing else.
"""
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        fixed_code = response.choices[0].message.content.strip()
        
        # Remove markdown code block delimiters if they exist
        if fixed_code.startswith("```"):
            # Extract content between code blocks
            pattern = r"```(?:java)?\s*([\s\S]*?)```"
            matches = re.findall(pattern, fixed_code)
            if matches:
                fixed_code = "\n".join(matches)
            else:
                # If regex didn't work, try a simple trim approach
                fixed_code = fixed_code.replace("```java", "").replace("```", "").strip()
        
        return fixed_code
    except Exception as e:
        print(f"Error getting OpenAI fix response: {e}")
        return test_code  # Return original if we can't get a fix

def process_item(item: Dict[str, Any], item_index: int, rust_files_dir: str = None) -> Dict[str, Any]:
    """Process a single dataset item and return the updated item."""
    code = item.get("code", "")
    signature = item.get("function_signature", "")
    query = item.get("query", "")
    name = item.get("name", "")
    
    print(f"Processing item {item_index}: {name}...")
    
    spring_version = item.get("to_version", "")
    
    print(f"Item {item_index}: Using Spring version {spring_version}")
    
    # 生成初始测试代码
    test_code = generate_test_code(code, signature, query)
    
    # Try to run tests with retries
    max_attempts = 5
    success = False
    
    for attempt in range(max_attempts):
        print(f"Item {item_index}: Test attempt {attempt+1}/{max_attempts}")
        
        success, error_message = run_java_code(
            code, test_code, spring_version,
        )
        
        # Log the detailed output to a file for debugging
        if rust_files_dir:
            log_dir = os.path.join(rust_files_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"test_{item_index:05d}_attempt_{attempt+1}.log"), 'w', encoding='utf-8') as f:
                f.write(f"ATTEMPT: {attempt+1}\nTEST CODE: {test_code}\nSUCCESS: {success}\nERROR: {error_message}")
        
        if success:
            print(f"Item {item_index}: Test SUCCESSFUL on attempt {attempt+1}")
            break
        
        print(f"Item {item_index}: Test FAILED on attempt {attempt+1}: {error_message}")
        
        if attempt < max_attempts - 1:
            # Get improved test code with error feedback
            test_code = fix_test_with_feedback(code, test_code, error_message, signature, query)
    
    # Store the result
    if success:
        item["test_program"] = test_code
    else:
        item["test_program"] = "INCORRECT TEST"
    
    return item

def process_dataset_concurrent(input_file: str, output_file: str, rust_files_dir: str = None, max_workers: int = 4) -> None:
    """Process the dataset with concurrent workers while maintaining order."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    results = [None] * total_items  # Pre-allocate result list to maintain order
    
    # Create progress bar
    progress_bar = tqdm(total=total_items, desc="Processing Java code samples")
    
    # Counter for tracking progress
    completed_count = 0
    incorrect_code_count = 0
    incorrect_test_count = 0
    
    # Process items concurrently while preserving order
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store futures with their original indices
        future_to_idx = {
            executor.submit(process_item, item, idx, rust_files_dir): idx 
            for idx, item in enumerate(data)
        }
        
        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                
                # Update counters
                completed_count += 1
                if result.get("test_program") == "INCORRECT CODE":
                    incorrect_code_count += 1
                elif result.get("test_program") == "INCORRECT TEST":
                    incorrect_test_count += 1
                
                # Update progress bar
                progress_bar.update(1)
                
                # Save checkpoint every 40 items
                if completed_count % 20 == 0:
                    # Copy over processed results to data
                    for i, res in enumerate(results):
                        if res is not None:
                            data[i] = res
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"\nCheckpoint saved at {completed_count}/{total_items} items")
                
            except Exception as e:
                print(f"\nError processing item {idx}: {e}")
                # In case of error, keep the original item
                results[idx] = data[idx]
                results[idx]["test_program"] = "ERROR: " + str(e)
                progress_bar.update(1)
    
    progress_bar.close()
    
    # Update data with all results
    for i, result in enumerate(results):
        if result is not None:
            data[i] = result
    
    # Calculate success rate
    valid_items = total_items - incorrect_code_count
    success_rate = 0
    if valid_items > 0:
        success_rate = (valid_items - incorrect_test_count) / valid_items * 100
    
    print(f"\nTest Results:")
    print(f"Total items: {total_items}")
    print(f"Incorrect code (function not found): {incorrect_code_count}")
    print(f"Tests that failed after retries: {incorrect_test_count}")
    print(f"Success rate: {success_rate:.2f}%")
    
    # Save the final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Rust code using OpenAI-generated test cases")
    parser.add_argument("--input_file", help="Input JSON file with Rust code samples", default=INPUT_FILE)
    parser.add_argument("--output_file", help="Output JSON file for results", default=OUTPUT_FILE)
    parser.add_argument("--rust_files_dir", help="Directory to save Rust test files (optional)", default=LOGS_DIR)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    process_dataset_concurrent(args.input_file, args.output_file, args.rust_files_dir, args.max_workers)