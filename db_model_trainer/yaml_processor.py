import yaml

def process_yaml(yaml_string: str) -> str:
    """
    Process a YAML string by capitalizing all values.
    
    Args:
        yaml_string (str): Input YAML string
        
    Returns:
        str: Processed YAML string with capitalized values
    """
    try:
        # Parse YAML
        data = yaml.safe_load(yaml_string)
        if data is None:  # Handle empty YAML
            return ""
        
        # Capitalize all string values recursively
        def capitalize_values(obj):
            if isinstance(obj, dict):
                return {k: capitalize_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [capitalize_values(item) for item in obj]
            elif isinstance(obj, str):
                return obj.upper()
            return obj
        
        # Process the data
        processed_data = capitalize_values(data)
        
        # Convert back to YAML string with consistent formatting
        return yaml.dump(processed_data, default_flow_style=False, sort_keys=False)
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML string: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing YAML: {str(e)}") 