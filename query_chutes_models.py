#!/usr/bin/env python3
import requests
import json

def query_chutes_models():
    """Query Chutes.ai models and filter by criteria"""
    url = "https://llm.chutes.ai/v1/models"
    
    try:
        response = requests.get(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        models = data.get("data", [])
        
        # Filter models with text output_modalities and context_length > 200k
        filtered_models = []
        for model in models:
            # Check if output_modalities includes "text"
            output_modalities = model.get("output_modalities", [])
            if "text" not in output_modalities:
                continue
                
            # Check context length > 200k
            context_length = model.get("context_length", 0)
            if context_length <= 200000:
                continue
            
            # Get supported features
            supported_features = model.get("supported_features", [])
            
            filtered_models.append({
                "id": model.get("id"),
                "context_length": context_length,
                "has_tools": "tools" in supported_features,
                "has_reasoning": "reasoning" in supported_features,
                "has_structured_outputs": "structured_outputs" in supported_features
            })
        
        # Print results
        print("Models with text output, >200k context length:")
        print("=" * 60)
        for model in filtered_models:
            features = []
            if model["has_tools"]:
                features.append("tools")
            if model["has_reasoning"]:
                features.append("reasoning")
            if model["has_structured_outputs"]:
                features.append("structured_outputs")
            
            features_str = ", ".join(features) if features else "none"
            print(f"[{model['id']}, {model['context_length']}] - {features_str}")
        
        return filtered_models
        
    except requests.exceptions.RequestException as e:
        print(f"Error querying API: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return []

if __name__ == "__main__":
    query_chutes_models()