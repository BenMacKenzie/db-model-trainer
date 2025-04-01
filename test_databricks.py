from db_model_trainer import process_yaml_string
from db_model_trainer.cli import main
import click

def test_function_call():
    yaml_content = """
    name: test
    environment: development
    """
    result = process_yaml_string(yaml_content)
    print("Function call result:")
    print(result)

def test_cli_call():
    result = main.callback(
        input_file=None,
        yaml="name: test\nenvironment: development",
        output=click.get_text_stream('stdout'),
        standalone_mode=False
    )
    print("\nCLI call result:")
    print(result)

if __name__ == '__main__':
    test_function_call()
    test_cli_call() 