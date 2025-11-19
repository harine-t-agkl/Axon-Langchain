# main.py  -- explicit Wikipedia trigger + agent fallback
import sys, traceback
from agent import run_agent
# import direct wiki_search for explicit requests
from tools.wiki_tool import wiki_search

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

        # ---------- explicit wiki trigger ----------
        lowered = question.lower().strip()
        if ("wikipedia" in lowered) or ("wiki" in lowered) or lowered.startswith("wiki:") or lowered.startswith("w:") or ("use wikipedia" in lowered):
            print("Explicit Wikipedia request detected — calling Wikipedia tool...\n")
            try:
                wiki_res = wiki_search(question)
                # if wiki_search returns empty or an error message, show fallback message
                if not wiki_res:
                    print("Wikipedia returned no results.")
                else:
                    # Print the wiki result (it's already cleaned by the wrapper)
                    print("\nWIKIPEDIA RESULT:\n")
                    print(wiki_res)
            except Exception:
                print("Wikipedia lookup failed (traceback):", file=sys.stderr)
                traceback.print_exc()
            # after wiki result, continue loop
            continue

        # ---------- otherwise use the agent ----------
        try:
            response = run_agent(question)
            print("\nRESPONSE:\n")
            print(response)
        except Exception:
            print("Agent error (traceback):", file=sys.stderr)
            traceback.print_exc()

if __name__ == "__main__":
    main_loop()
