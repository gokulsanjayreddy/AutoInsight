# AutoInsight Contributing Guide

## Submitting Changes

1. **Fork the repository** and create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Create a new file or modify existing files** following the project's coding style.

3. **Run tests locally** to ensure your changes don't break existing functionality:
   ```bash
   pip install -r requirements.txt
   # Run the app and test manually
   streamlit run app.py
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
- Keep functions under ~50 lines; break up long logic into helpers
- All plotting functions should return a matplotlib Figure object
- Include a `requirements.txt` with pinned versions
- Test your changes with different CSV file types
- Ensure the app runs locally with `streamlit run app.py`

## Report Issues

- Use the issue tracker to report bugs or request features
- Include a sample CSV file that demonstrates the issue
- Mention your Python version and operating system

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.