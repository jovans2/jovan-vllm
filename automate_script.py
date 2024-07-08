import subprocess
import time

cmd1 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-70b-chat-hf',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=8',
        '--batch-size=256']

cmd2 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-70b-chat-hf',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=4',
        '--batch-size=256']

cmd3 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-70b-chat-hf',
        '--swap-space',     
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=2',
        '--batch-size=256']

cmd4 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-7b-chat-hf',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=8',
        '--batch-size=256']

cmd5 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-13b-chat-hf',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=8',
        '--batch-size=256']

cmd6 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-70b-chat-hf',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=8',
        '--batch-size=64']

cmd7 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-70b-chat-hf',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=8',
        '--batch-size=16']

cmd8 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'meta-llama/Llama-2-70b-chat-hf',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=8',
        '--batch-size=1']

cmd9 = ['python3',
        '-m',
        'vllm.entrypoints.api_server',
        '--model',
        'Llama-2-70b-chat-hf-awq',
        '--swap-space',
        '16',
        '--disable-log-requests',
        '--tensor-parallel-size=8',
        '--batch-size=256',
        '--quantization=awq']

commands = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9]

def start_server(command):
    
    # Start the process in the background
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return process

def main():

    for cmd in commands:
        # Start the server
        server_process = start_serverc(cmd)
    
        print("Server started with PID:", server_process.pid)
    
        # Sleep for 5 minutes
        print("Sleeping for 5 minutes before killing the server...")
        time.sleep(300)  # 300 seconds = 5 minutes
    
        

        # Kill the server process
        print("Killing the server process...")
        server_process.terminate()

if __name__ == "__main__":
    main()

