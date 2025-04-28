import sys
import re

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    expected_input_lengths = [128, 256, 512, 1024, 2048, 4096]

    latency_pattern = re.compile(r'P50 latency\s*=\s*([0-9.]+)')
    input_len_pattern = re.compile(r'Input len\s*=\s*(\d+)')
    
    current_input_index = 0  # Index inside expected_input_lengths
    last_seen_input_len = None

    with open(input_file, 'r') as f:
        for line in f:
            if "torch.OutOfMemoryError: CUDA out of memory" in line:
                for _ in range(3):
                    print(0)
                current_input_index = 0
                last_seen_input_len = None
                continue

            input_len_match = input_len_pattern.search(line)
            if input_len_match:
                current_input_len = int(input_len_match.group(1))
                if current_input_len in expected_input_lengths:
                    current_input_index = expected_input_lengths.index(current_input_len)
                    last_seen_input_len = current_input_len
                continue

            if "Exception: Invalid prefix encountered" in line:
                #if last_seen_input_len is not None:
                #    print(f"Invalid prefix after Input len = {last_seen_input_len}")
                #else:
                #    print("Invalid prefix encountered before any valid Input len")
                
                # Adjusted the remaining calculation
                remaining = len(expected_input_lengths) - current_input_index
                for _ in range(remaining):
                    print(0)
                current_input_index = 0
                last_seen_input_len = None
                continue

            match = latency_pattern.search(line)
            if match:
                seconds = float(match.group(1))
                milliseconds = seconds * 1000
                print(milliseconds)
                
                # After successful latency extraction, move to next expected input
                current_input_index += 1

if __name__ == "__main__":
    main()

