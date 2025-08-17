# Contributing to GraphVerse

Thank you for your interest in contributing to GraphVerse! This document provides guidelines for contributing to this research codebase.

## 🎯 Ways to Contribute

### Research Contributions
- **New rule types**: Implement additional graph traversal constraints
- **Analysis methods**: Add novel ways to analyze rule learning dynamics  
- **Architectures**: Test rule learning in different model architectures
- **Evaluation metrics**: Develop new ways to measure rule internalization

### Code Contributions  
- **Bug fixes**: Fix issues in existing code
- **Performance improvements**: Optimize training or analysis pipelines
- **Documentation**: Improve documentation and examples
- **Visualization**: Add new plotting functions or improve existing ones

### Experimental Contributions
- **Replication studies**: Verify results across different setups
- **Ablation studies**: Test the impact of different components
- **Scaling studies**: Test behavior with larger models/datasets
- **Cross-domain studies**: Apply framework to non-graph domains

## 🔧 Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/GraphVerse.git
cd GraphVerse
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests
cd tests
python run_tests.py

# Run a small experiment
cd ../scripts  
python complete_rule_analysis_demo.py --config ../configs/small_demo_config.json
```

## 📝 Code Style

### Python Code Standards
- **Black** for code formatting: `black .`
- **Flake8** for linting: `flake8 graphverse/`  
- **Type hints** encouraged for new functions
- **Docstrings** required for all public functions (Google style)

### Example Function

```python
def analyze_rule_violations(
    token_data: List[Dict[str, Any]], 
    rule_types: List[str]
) -> Dict[str, float]:
    """
    Analyze rule violation patterns in token-level data.
    
    Args:
        token_data: List of token-level analysis dictionaries
        rule_types: Types of rules to analyze
        
    Returns:
        Dictionary mapping rule types to violation rates
        
    Raises:
        ValueError: If token_data is empty or malformed
    """
    if not token_data:
        raise ValueError("token_data cannot be empty")
    
    # Implementation here...
    return violation_rates
```

### File Organization
- New analysis scripts go in `scripts/`
- Core functionality goes in `graphverse/` submodules
- Configuration files go in `configs/`
- Documentation goes in `docs/`
- Test files go in `tests/`

## 🧪 Testing

### Writing Tests

Create test files in `tests/` with descriptive names:

```python
# tests/test_rule_analysis.py
import unittest
from graphverse.llm.evaluation import analyze_rule_violations_for_token

class TestRuleAnalysis(unittest.TestCase):
    def test_violation_detection(self):
        # Test your functionality
        pass
        
    def test_edge_cases(self):
        # Test edge cases
        pass
```

### Running Tests

```bash
cd tests
python run_tests.py

# Or run specific test
python -m unittest test_rule_analysis.TestRuleAnalysis.test_violation_detection
```

## 📊 Adding New Rule Types

### 1. Implement Rule Class

```python
# In graphverse/graph/rules.py
class MyCustomRule(Rule):
    def __init__(self, custom_params):
        self.custom_params = custom_params
        self.is_my_custom_rule = True  # For identification
    
    def is_satisfied_by(self, walk, graph):
        """Check if walk satisfies your custom rule."""
        # Implement rule logic
        return True
    
    def get_violation_position(self, walk):
        """Return position where rule is first violated, or None."""
        # Optional: implement for detailed analysis
        return None
```

### 2. Add to Evaluation

```python  
# In graphverse/llm/evaluation.py, update analyze_rule_violations_for_token:

elif hasattr(rule, 'is_my_custom_rule') and rule.is_my_custom_rule:
    rule_analysis = analyze_my_custom_rule_violation(predicted_walk, rule, predicted_node)
```

### 3. Add Analysis Function

```python
def analyze_my_custom_rule_violation(walk, rule, predicted_node):
    """Analyze my custom rule violations."""
    analysis = {
        'violates': False, 
        'violation_position': None,
        'context_nodes': [],
        'expected_behavior': ''
    }
    
    # Implement analysis logic
    return analysis
```

## 📈 Adding New Visualizations

### 1. Create Visualization Function

```python
# In scripts/learning_progression_viz.py or new file
def plot_my_custom_analysis(data, output_path=None, figsize=(12, 8)):
    """
    Create a custom visualization for rule learning analysis.
    
    Args:
        data: Analysis data dictionary
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create your plot
    ax.plot(...)
    ax.set_xlabel(...)
    ax.set_ylabel(...)
    ax.set_title(...)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig, ax
```

### 2. Integrate into Report Generation

```python
# In scripts/learning_progression_viz.py, update create_learning_progression_report:

try:
    plot_my_custom_analysis(checkpoint_data,
                           os.path.join(viz_folder, "my_custom_analysis.png"))
    plots_created.append("my_custom_analysis.png")
except Exception as e:
    print(f"Error creating custom analysis plot: {e}")
```

## 🔬 Research Guidelines

### Experimental Reproducibility
- **Always set random seeds** in experiments
- **Document all hyperparameters** in config files
- **Save intermediate results** for replication
- **Include error bars/confidence intervals** in plots

### Data and Results
- **Never commit large data files** to git (use git-lfs if needed)
- **Save raw results** in JSON format for reanalysis  
- **Include statistical significance tests** when comparing methods
- **Document any preprocessing steps** clearly

### Documentation Requirements
- **Update README** with new features
- **Add examples** for new functionality  
- **Document any new dependencies**
- **Include references** for novel methods

## 📋 Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation as needed
- Run tests and ensure they pass

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add custom rule analysis functionality

- Implement MyCustomRule class
- Add visualization for custom rule violations  
- Update evaluation pipeline to handle new rule type
- Add tests for new functionality"
```

### 4. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Create pull request on GitHub with:
- Clear description of changes
- Link to any relevant issues
- Screenshots of new visualizations (if applicable)
- Results from any new experiments

### 5. Code Review

- Address reviewer feedback
- Update documentation if requested  
- Ensure all tests pass
- Squash commits if requested

## 🏷️ Issue Guidelines

### Bug Reports
Include:
- Exact error message
- Steps to reproduce
- Python version and dependencies
- Configuration used
- Expected vs actual behavior

### Feature Requests  
Include:
- Clear description of desired functionality
- Use case/motivation
- Example of how it would work
- Rough implementation ideas (if any)

### Research Questions
Include:
- Specific research question
- Relevant background/references
- Proposed experimental approach  
- Expected outcomes

## 🤝 Community Guidelines

### Be Respectful
- Use inclusive language
- Be patient with beginners
- Provide constructive feedback
- Credit others' contributions

### Research Ethics
- Properly cite related work
- Share results openly
- Acknowledge limitations
- Follow academic integrity standards

### Communication
- Use clear, descriptive titles for issues/PRs
- Respond promptly to feedback
- Ask questions if anything is unclear
- Help others when you can

## 🎯 Priority Areas

We're especially interested in contributions in:

1. **Novel rule types** that test different aspects of model learning
2. **Scaling studies** with larger models and datasets
3. **Cross-architecture comparisons** (CNNs, RNNs, etc.)
4. **Real-world applications** beyond synthetic graphs
5. **Theoretical analysis** of learning dynamics
6. **Performance optimizations** for large-scale experiments

## 📚 Resources

- **Research Papers**: See references in main README
- **Code Style**: [Black documentation](https://black.readthedocs.io/)
- **Testing**: [Python unittest docs](https://docs.python.org/3/library/unittest.html)
- **Git**: [GitHub flow](https://guides.github.com/introduction/flow/)

---

Thank you for contributing to GraphVerse! Your contributions help advance our understanding of how language models learn implicit rules and constraints.