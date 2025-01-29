# Feature Selection Optimization with Competitive Coevolution PSO 

This repository contains implementations of a Competitive Coevolution Particle Swarm Optimization (PSO) algorithm along with an email processing system. The project combines feature selection optimization with natural language processing capabilities.

## Project Structure

```
├── coevolution_pso/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── fitness.py
│   ├── tournament.py
│   └── breeding.py
├── email_processor/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── nlp_setup.py
│   ├── visualization.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Features

### Competitive Coevolution PSO
- Feature selection optimization using competitive coevolution
- Single elimination tournament selection
- Adaptive mutation rates
- Elitism-based population management
- Balanced fitness function with complexity penalties

### Email Processing
- Google Drive integration for data loading
- Natural Language Processing (NLP) capabilities
- Text visualization with WordCloud
- Duplicate removal and preprocessing
- Configurable data processing pipeline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/coevolution-pso-email.git
cd coevolution-pso-email
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- wordcloud
- google.colab (for Google Colab integration)

## Usage

### Coevolution PSO

```python
from coevolution_pso import CompetitiveCoevolutionPSO

# Initialize optimizer
optimizer = CompetitiveCoevolutionPSO(n_particles=32, dim=100)

# Run optimization
best_solution, best_fitness = optimizer.optimize(
    X_train=your_training_data,
    X_test=your_test_data,
    y_train=your_training_labels,
    y_test=your_test_labels,
    max_generations=40
)
```

### Email Processing

```python
from email_processor.main import main

# Run the complete email processing pipeline
emails, labels = main()
```

## Configuration

### Coevolution PSO Parameters
- `n_particles`: Number of particles in the population (will be rounded to next power of 2)
- `dim`: Dimensionality of the search space
- `max_generations`: Maximum number of generations for evolution

### Email Processing Configuration
Edit `email_processor/config.py` to modify:
- Data file paths
- NLTK resources
- Column names
- Other processing parameters

## Implementation Details

### Coevolution PSO
The implementation uses a competitive coevolution approach with:
- Tournament-based fitness evaluation
- Adaptive mutation rates based on fitness scores
- Elitism for population preservation
- Balanced fitness function considering:
  - Classification accuracy
  - Feature selection complexity
  - Solution diversity

### Email Processing
The email processing pipeline includes:
- Automated Google Drive mounting
- NLTK resource management
- Text preprocessing
- Visualization capabilities
- Duplicate handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by various implementations of Particle Swarm Optimization
- Built upon NLTK and scikit-learn libraries
- Visualization components powered by matplotlib and WordCloud

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Future Work

- Implementation of parallel tournament evaluation
- Addition of more sophisticated NLP features
- Enhanced visualization capabilities
- Support for more data sources
- Integration with deep learning models

## Citation

If you use this code in your research, please cite:

```bibtex
@software{coevolution_pso_email,
  author = {Your Name},
  title = {Competitive Coevolution PSO with Email Processing},
  year = {2025},
  url = {https://github.com/yourusername/coevolution-pso-email}
}
```
