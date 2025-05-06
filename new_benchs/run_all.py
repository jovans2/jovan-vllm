import subprocess
import os

# Define models and TP values
models = [
    "meta-llama/Meta-Llama-3-8b-Instruct",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "ibm/PowerMoE-3b",
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-llm-67b-base",
]

models = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-405B-FP8",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-R1",
    "state-spaces/mamba-790m",
    "state-spaces/mamba-2.8b",
    "nvidia/mamba2-hybrid-8b-3t-128k"
]

tensor_parallel_sizes = [1, 2, 4, 8]

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

# Open a single master log file
with open("logs/master_log.txt", "w") as master_log:
    for model in models:
        for tp in tensor_parallel_sizes:
            cmd = [
                "python3",
                "jovan_bench.py",
                "--model", model,
                "--dtype", "half",
                "--enforce-eager",
                "--tensor-parallel-size", str(tp)
            ]

            # Write header for this run
            master_log.write(f"\n\n=== Running: {cmd} ===\n")
            master_log.flush()  # Immediately write to disk
            
            print(f"Running: {cmd}")
            
            try:
                # Run the command, wait until it finishes
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False  # Don't throw exception if command fails
                )
                
                # Write output to log
                master_log.write(process.stdout)
                master_log.flush()
                
                # Check return code
                if process.returncode != 0:
                    master_log.write(f"\n[ERROR] Process exited with return code {process.returncode}\n")
                    master_log.flush()
                    
            except Exception as e:
                # Log the exception and continue
                master_log.write(f"\n[EXCEPTION] {str(e)}\n")
                master_log.flush()

print("\nAll runs finished. Full logs are saved in logs/master_log.txt")

