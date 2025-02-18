# Feature Selection Optimization with Competitive Coevolution GA 

This repository contains implementations of a Competitive Coevolution Particle Swarm Optimization (PSO) algorithm along with an email processing system. The project combines feature selection optimization with natural language processing capabilities.

## Project Structure

```
├── coevolution_ga/
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

### Competitive Coevolution GA
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


## Implementation Details

### Coevolution 
The implementation uses a competitive coevolution approach with:
- Tournament-based fitness evaluation
- Adaptive mutation rates based on fitness scores
- Elitism for population preservation
- Balanced fitness function considering:
  - Classification accuracy
  - Feature selection complexity
  - Solution diversity






