import os
import argparse
import subprocess
from openai import OpenAI


def get_git_diff():
    try:
        # For simplicity, we get the diff of the latest commit
        # In a real Action, you might compare PR branches
        result = subprocess.run(
            ["git", "show", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        print(f"Failed to get git diff: {e}")
        return ""


def generate_docs(diff_content: str, api_key: str):
    client = OpenAI(api_key=api_key)
    prompt = f"""
    You are an AI repository maintainer.
    Review the following git diff and output a summary of the architectural changes,
    and provide recommendations for documentation updates.

    Diff:
    {diff_content}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful software engineering assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="AI Documentation Agent")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without making API calls"
    )
    args = parser.parse_args()

    print("AI Documentation Agent initialized.")
    print("Reviewing repository changes...")

    diff = get_git_diff()

    if args.dry_run:
        print("[DRY RUN] Would fetch latest PR diff.")
        print(f"[DRY RUN] Diff length: {len(diff)} characters.")
        print("[DRY RUN] Would send diff to LLM for review.")
        print(
            "[DRY RUN] Would generate architecture summaries and documentation updates."
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. Cannot run full AI analysis.")
            return

        print("Analyzing changes using AI...")
        if not diff:
            print("No diff found or failed to read diff. Exiting.")
            return

        summary = generate_docs(diff, api_key)
        if summary:
            print("\n--- AI Review Summary ---")
            print(summary)
            print("-------------------------")

            # Optionally write this to a file or post as a PR comment
            with open("ai_review_summary.md", "w") as f:
                f.write(summary)

            print("AI analysis complete. Documentation review successfully saved.")
        else:
            print("Failed to generate AI summary.")


if __name__ == "__main__":
    main()
