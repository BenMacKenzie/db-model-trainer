# DB Model Trainer

A Python package for processing YAML in Databricks environments.

## Installation

```bash
pip install db-model-trainer
```

## Usage

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

## Development

To install in development mode:

```bash
pip install -e .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 



python -m build
python -m twine upload dist/*


