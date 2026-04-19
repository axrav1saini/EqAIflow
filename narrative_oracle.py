import os
import json
from google import genai
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_plain_english_report(metrics_json: str) -> str:
    """
    Converts statistical parity metrics into a plain-English severity assessment using Gemini.
    """
    
    prompt = f"""
    You are the EquiLens Narrative Engine, a world-class AI fairness auditor.
    I am providing you with raw statistical bias metrics calculated from a dataset.
    
    Metrics:
    {metrics_json}
    
    Your task is to translate these raw statistics into a human-readable narrative report.
    
    Provide ONLY a Markdown string with a Severity Assessment (Low, Medium, High, Critical) and a plain-English explanation of who is being harmed or underrepresented. (Typically Demographic Parity Ratio < 0.8 is problematic).
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating narrative report: {str(e)}"

def generate_recommendations(metrics_json: str) -> str:
    """
    Generates recommended next steps for auto-mitigation based on bias metrics.
    """
    prompt = f"""
    You are the EquiLens Narrative Engine, a world-class AI fairness auditor.
    I am providing you with raw statistical bias metrics calculated from a dataset.
    
    Metrics:
    {metrics_json}
    
    Your task is to provide recommended next steps for auto-mitigation or data reweighting.
    Respond ONLY with a Markdown string outlining these recommendations.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"