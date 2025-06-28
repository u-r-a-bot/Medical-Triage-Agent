#!/usr/bin/env python3
"""
Test script to verify JSON parsing fixes in the triage agent.
"""

import os
import sys
from triage_agent import process_triage_request

def test_triage_agent():
    """Test the triage agent with various inputs."""
    
    # Test cases
    test_cases = [
        "I have a severe headache and fever",
        "Chest pain with shortness of breath",
        "Mild cough and runny nose",
        "Abdominal pain with nausea and vomiting"
    ]
    
    print("Testing Triage Agent LLM Output")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case}")
        print("-" * 30)
        
        try:
            result = process_triage_request(test_case)
            
            if result["success"]:
                print("✅ Success!")
                print(f"   LLM Output: {result['raw_response']}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    # Set API key if not already set
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDtkhXwbXJ4G34igeM8z44MJieWANsRpWM"
    
    test_triage_agent() 