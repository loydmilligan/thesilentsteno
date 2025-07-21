# Claude Configuration for Python Project

## Bash Commands
- python -m venv venv: Create virtual environment
- source venv/bin/activate: Activate virtual environment (Unix)
- pip install -r requirements.txt: Install dependencies
- python -m pytest: Run tests
- python -m black .: Format code
- python -m flake8: Run linter

## Code Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Use docstrings for functions and classes
- Use Black for code formatting

## Project Structure
- Place modules in appropriate packages
- Use __init__.py files for packages
- Separate tests in tests/ directory

## Testing
- Use pytest for testing
- Write unit tests for all functions
- Use fixtures for test data
- Aim for high test coverage

## Environment
- Always use virtual environments
- Pin dependency versions in requirements.txt
- Use .env files for environment variables