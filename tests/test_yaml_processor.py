import pytest
from db_model_trainer import process_yaml

def test_basic_yaml_processing():
    input_yaml = """
    name: test
    version: 1.0
    """
    expected = "name: TEST\nversion: 1.0\n"
    assert process_yaml(input_yaml) == expected

def test_nested_yaml_processing():
    input_yaml = """
    name: test
    config:
      environment: development
      features:
        - feature1
        - feature2
    """
    expected = "name: TEST\nconfig:\n  environment: DEVELOPMENT\n  features:\n  - FEATURE1\n  - FEATURE2\n"
    assert process_yaml(input_yaml) == expected

def test_invalid_yaml():
    input_yaml = """
    name: test: invalid: format
    - [broken, yaml: here
    """
    with pytest.raises(ValueError):
        process_yaml(input_yaml) 