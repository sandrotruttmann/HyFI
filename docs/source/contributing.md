# Contributing

We welcome contributions to HyFI! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/your-username/HyFI.git
   cd HyFI
   ```

3. Create a new branch for your feature:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. Install the package in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

### Testing

- Write tests for new features using pytest
- Ensure all tests pass before submitting a pull request:

  ```bash
  pytest
  ```

- Aim for good test coverage of new code

### Documentation

- Update documentation for any changed functionality
- Add examples for new features
- Keep the README up to date

## Submitting Changes

1. Commit your changes:

   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

2. Push to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub
4. Wait for review and address any feedback

## Questions?

If you have questions about contributing, feel free to open an issue on GitHub.

## License

By contributing to HyFI, you agree that your contributions will be licensed under the GNU General Public License v3.0.
