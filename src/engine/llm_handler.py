import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
from src.engine.retriever import Retriever

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class LLMHandler:
    """
    LLM Handler for reranking assessment search results.

    Uses Google's Gemini LLM to rerank assessment results based on relevance
    to the user query, ensuring a balance of technical and soft skill assessments.
    """
    def __init__(self):
        """Initialize the LLM handler with Gemini model."""
        if not GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not found in environment variables.")
            print("LLM Reranking will be skipped (returning raw vector results).")
            self.model = None
            return

        # Configure the Gemini API
        genai.configure(api_key=GEMINI_API_KEY)

        # Initialize the Gemini model
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        print("Connected to Gemini LLM (gemini-2.5-flash) for reranking.")

    def rerank(self, query, results):
        """
        Rerank assessment results using LLM.

        Args:
            query (str): The user query
            results (list): List of assessment dictionaries from vector search

        Returns:
            list: Reranked list of assessment dictionaries
        """
        # Return raw results if LLM is not available or results are empty
        if not self.model or not results:
            return results[:10]

        # Format assessment data for the prompt
        result_texts = ""
        for idx, res in enumerate(results):
            result_texts += f"{idx + 1}. Assessment Name: {res['name']}\n"
            result_texts += f"   Type: {res['test_type']}\n"
            result_texts += f"   Description: {str(res.get('description', ''))[:500]}\n\n"

        # Create prompt for the LLM
        prompt = f"""
                You are an expert SHL Assessment Recruiter.

                USER QUERY: "{query}"

                TASK:
                Select the best 10 assessments from the CANDIDATE LIST below that match the User Query.
                Try to take as many relevant assignments as possible.

                CRITICAL RULES:
                1. BALANCE: If the query asks for both hard skills (e.g., coding, analysis) and soft skills (e.g., leadership, personality), you MUST pick a mix of 'Knowledge & Skills' and 'Personality & Behavior'/'Competencies' tests.
                2. ACCURACY: Only choose assessments that are genuinely relevant to the query.
                3. OUTPUT FORMAT: Return ONLY a valid JSON array of the integer IDs of your selected choices (e.g. [0, 2, 4]). Do not write any other text.

                CANDIDATE LIST:
                {result_texts}
                """

        try:
            # Generate response from LLM
            response = self.model.generate_content(prompt)

            # Parse the LLM response to extract the JSON array
            text = response.text
            # Use regex to find the list part [ ... ]
            match = re.search(r'\[.*\]', text, re.DOTALL)

            if match:
                clean_text = match.group(0)
                selected_indices = json.loads(clean_text)
            else:
                # Fallback cleaning if regex fails
                clean_text = text.strip().replace("```json", "").replace("```", "")
                selected_indices = json.loads(clean_text)

            # Process the selected indices to create the final results
            final_results = []
            already_added = set()
            for idx in selected_indices:
                if isinstance(idx, int) and 0 <= idx < len(results):
                    final_results.append(results[idx])
                    already_added.add(idx)

            # Fallback to vector search if LLM returns no valid matches
            if not final_results:
                print("LLM returned no matches, falling back to vector search.")
                return results[:10]

            # Ensure we have at least 5 results by adding from vector search
            if len(final_results) < 5:
                for idx, res in enumerate(results):
                    if idx not in already_added:
                        final_results.append(res)
                    if len(final_results) >= 10:
                        break

            return final_results

        except Exception as e:
            print(f"LLM Error: {e}")
            # Fallback to vector search results on error
            return results[:10]


if __name__ == "__main__":
    # Test the LLM handler with a sample query
    primary_matches = Retriever().search("I need a Java developer who is good at teamwork")

    print("Testing LLM Handler...")
    handler = LLMHandler()

    # Test query combining technical and soft skills
    test_query = "I need a Java developer who is good at teamwork"

    results = handler.rerank(test_query, primary_matches)

    print(f"\nSelected {len(results)} assessments for query: '{test_query}'")
    for item in results:
        print(f" - {item['name']} ({item['test_type']})")
