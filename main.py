
# main.py -- clean version (no wiki trigger, with timeout)
import sys
import traceback
import concurrent.futures
from agent import run_agent

def main_loop():
    print("Agnikul assistant (agent). Type 'q' to quit.")

    while True:
        try:
            question = input("Ask your question (q to quit): ").strip()
        except EOFError:
            print("\nEOF received, exiting.")
            break

        if not question:
            continue
        if question.lower() == "q":
            print("Goodbye.")
            break

        # ------------------------------
        # Run the agent with a timeout
        # ------------------------------
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_agent, question)
                # ⬇ Set timeout here (in seconds)
                response = future.result(timeout=100)

            print("\nRESPONSE:\n")
            print(response)

        except concurrent.futures.TimeoutError:
            print("\n⚠ Agent timed out after 100 seconds.\nTry a shorter query.")
        except Exception:
            print("Agent error (traceback):", file=sys.stderr)
            traceback.print_exc()

def main_loop():
    print("Agnikul assistant (agent). Type 'q' to quit.")

    while True:
        try:
            question = input("Ask your question (q to quit): ").strip()
        except EOFError:
            print("\nEOF received, exiting.")
            break

        if not question:
            continue
        if question.lower() == "q":
            print("Goodbye.")
            break

        # ------------------------------
        # Run the agent with a timeout
        # ------------------------------
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_agent, question)
                # ⬇ Set timeout here (in seconds)
                response = future.result(timeout=1000)

            print("\nRESPONSE:\n")
            print(response)

        except concurrent.futures.TimeoutError:
            print("\n⚠ Agent timed out after 1000 seconds.\nTry a shorter query.")
        except Exception:
            print("Agent error (traceback):", file=sys.stderr)
            traceback.print_exc()

if __name__ == "__main__":
    main_loop()
