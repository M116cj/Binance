# Contributing to Crypto Surge Prediction System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 📝 Documentation Updates

### Single Source of Truth: `docs/project_manifest.yaml`

All project metadata (features, architecture, tech stack, etc.) is centralized in `docs/project_manifest.yaml`. This ensures consistency across all documentation.

**Before making documentation changes:**

1. **Update the manifest**: Edit `docs/project_manifest.yaml` with your changes
2. **Regenerate README**: Run `python scripts/update_readme.py`
3. **Update replit.md**: Manually sync any relevant changes to `replit.md`

### Automatic README Updates

The README.md is **automatically generated** from the manifest. Do not edit README.md directly!

#### Local Updates

```bash
# After editing docs/project_manifest.yaml
python scripts/update_readme.py
```

#### Automatic Updates (GitHub Actions)

The README is automatically updated when you:
- Push changes to `docs/project_manifest.yaml`
- Push changes to `docs/README_template.md`
- Push changes to `scripts/update_readme.py`
- Trigger the workflow manually from GitHub Actions

The workflow runs on:
- Every push to `main` branch (affecting documentation files)
- Manual dispatch
- Weekly schedule (Sundays at 00:00 UTC)

## 🔧 Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/crypto-surge-prediction.git
cd crypto-surge-prediction
```

### 2. Set Up Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Set up database
# (instructions specific to your setup)
```

### 3. Make Changes

- Follow existing code style and patterns
- Add tests for new features
- Update documentation (manifest file)
- Ensure all tests pass

### 4. Test Your Changes

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=backend --cov-report=html

# Test README generation
python scripts/update_readme.py
```

### 5. Commit and Push

```bash
git add .
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

### 6. Open a Pull Request

- Provide a clear description of your changes
- Reference any related issues
- Ensure CI checks pass

## 📋 Commit Message Guidelines

Follow conventional commits format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add multi-exchange support
fix: correct probability calculation in OFI
docs: update installation instructions
```

## 🎯 Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

## 🧪 Testing Requirements

All new features should include:

- Unit tests for core logic
- Integration tests for API endpoints
- Documentation updates

## 📚 Documentation Standards

When adding features, document:

1. **In code**: Docstrings and comments
2. **In manifest**: Update `docs/project_manifest.yaml`
3. **In replit.md**: Add to "Recent Changes" section

## 🐛 Bug Reports

When reporting bugs, include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

## 💡 Feature Requests

For feature requests, describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered
- How this fits with the project's goals

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Join discussions for questions and ideas
- Check existing issues before creating new ones

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
