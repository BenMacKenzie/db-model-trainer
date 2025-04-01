# DB Model Trainer

A Python package for processing YAML in Databricks environments.

## Installation

```bash
pip install db-model-trainer
```

## Usage

### As a Python Package

```python
from db_model_trainer import process_yaml

# Example YAML string
yaml_string = """
name: test
version: 1.0
"""

# Process the YAML
result = process_yaml(yaml_string)
print(result)
```

### Command Line Interface

The package provides a command-line tool `yaml-processor` that can process YAML files or input from stdin:

```bash
# Process a file
yaml-processor input.yaml -o output.yaml

# Process from stdin
echo "name: test" | yaml-processor

# Get help
yaml-processor --help
```

## Development

To install in development mode:

```bash
pip install -e .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.



python -m build
python -m twine upload dist/*


