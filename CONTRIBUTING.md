# AutoInsight Contributing Guide

## Submitting Changes

1. **Fork the repository** and create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Create a new file or modify existing files** following the project's coding style.

3. **Run tests, linting, and type checks locally**:
   ```bash
   pip install -r requirements-dev.txt
   ruff check .
   ruff format --check .
   mypy modules/ app.py
   pytest --cov=modules
   ```

4. **Commit your changes** with a clear commit message:
   ```bash
   git add .
   git commit -m "feat: add new feature" or "fix: resolve issue"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request** targeting the `main` branch.

## Development Guidelines

- Add type hints to all function signatures
- Add docstrings to all public functions
- Keep functions under ~50 lines; break up long logic into focused helpers
- All plotting functions should return a matplotlib Figure or list of (name, Figure) tuples
- Pinned version ranges are enforced in `requirements.txt` (`>=X.Y,<Z.W`)
- Maintain $\ge 80\%$ test coverage across `modules/`
- Test your changes with different CSV file types (delimiters, encodings, edge cases)
- Ensure the app runs cleanly with `streamlit run app.py`

## Report Issues

- Use the issue tracker to report bugs or request features
- Include a sample CSV file that demonstrates the issue
- Mention your Python version and operating system

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.