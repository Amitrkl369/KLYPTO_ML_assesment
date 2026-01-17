# Contributing to Quantitative Trading System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/Klypto_ML_assignmant.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests and verify: `python setup_verify.py`
6. Commit: `git commit -m "Add your descriptive message"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise

Example:
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """
    Calculate Sharpe Ratio for a return series
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 6.5%)
        
    Returns:
        Sharpe ratio value
    """
    # Implementation
    pass
```

### Testing

- Add unit tests for new functionality
- Ensure all existing tests pass
- Test with different data scenarios

### Documentation

- Update README.md if adding new features
- Add comments for complex logic
- Update docstrings when modifying functions

## Areas for Contribution

### 🐛 Bug Fixes
- Report bugs through GitHub issues
- Include error messages and steps to reproduce
- Submit fixes with clear descriptions

### ✨ New Features
- **Data Sources**: Add support for new APIs
- **Indicators**: Implement additional technical indicators
- **Models**: Add new ML models or architectures
- **Strategies**: Implement alternative trading strategies
- **Visualizations**: Create new plots and dashboards

### 📚 Documentation
- Improve README clarity
- Add usage examples
- Create tutorials
- Translate documentation

### 🧪 Testing
- Add unit tests
- Improve test coverage
- Add integration tests

## Specific Contribution Ideas

### Easy (Good First Issues)
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Improve error messages
- Add data validation checks
- Create additional visualizations

### Medium
- Implement portfolio optimization
- Add risk management features
- Create interactive dashboards with Dash/Streamlit
- Add multi-timeframe analysis

### Advanced
- Implement reinforcement learning for strategy optimization
- Add deep learning architectures (Transformer, GRU)
- Create ensemble models
- Implement genetic algorithm for parameter optimization

## Code Review Process

1. All submissions require review
2. Reviews focus on:
   - Code quality and style
   - Test coverage
   - Documentation
   - Performance implications
3. Address review feedback promptly
4. Maintain professional and respectful communication

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: Add MACD indicator to feature engineering
fix: Correct futures rollover calculation
docs: Update installation instructions
test: Add unit tests for Greeks calculation
refactor: Optimize regime detection performance
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Testing
- `refactor`: Code refactoring
- `style`: Formatting
- `perf`: Performance improvement

## Questions?

- Open an issue for questions
- Check existing issues and PRs first
- Be patient and respectful

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
