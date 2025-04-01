import click
import sys
from typing import Optional, TextIO, Union
from .yaml_processor import process_yaml

def process_yaml_string(yaml_content: str) -> str:
    """Process a YAML string and return the result.
    
    This function can be called directly from Python code, including Databricks notebooks.
    
    Args:
        yaml_content: The YAML string to process
        
    Returns:
        The processed YAML string with capitalized values
        
    Raises:
        ValueError: If the YAML is invalid
    """
    return process_yaml(yaml_content)

@click.command()
@click.argument('input_file', type=click.File('r'), required=False)
@click.option('--yaml', '-y', help='YAML string to process directly')
@click.option('--output', '-o', type=click.File('w'), default='-', 
              help='Output file (default: stdout)')
def main(input_file: Optional[TextIO], yaml: Optional[str], output: TextIO) -> None:
    """Process YAML by capitalizing all string values.
    
    There are three ways to provide input:
    1. INPUT_FILE: Path to a YAML file
    2. --yaml/-y: Direct YAML string (for simple YAML)
    3. Standard input (for any YAML, including complex structures)
    
    If no input method is specified, reads from stdin.
    
    Examples:
        # Process a file:
        python -m db_model_trainer.cli input.yaml
        
        # Process a simple YAML string:
        python -m db_model_trainer.cli --yaml "name: test"
        
        # Process complex YAML from stdin:
        echo "name: test
        config:
          env: dev" | python -m db_model_trainer.cli
        
        # Process and save to file:
        python -m db_model_trainer.cli input.yaml -o output.yaml
    """
    try:
        # Determine input source
        if yaml is not None:
            yaml_content = yaml
        elif input_file is not None:
            yaml_content = input_file.read()
        else:
            yaml_content = click.get_text_stream('stdin').read()
        
        # Process the YAML
        result = process_yaml(yaml_content)
        
        # Write the result
        output.write(result)
            
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)

def run_cli():
    """Entry point for the CLI when run as a module."""
    try:
        main()
    except SystemExit as e:
        if e.code != 0:
            raise
        # Suppress SystemExit(0) when running in Databricks
        return

if __name__ == '__main__':
    run_cli() 