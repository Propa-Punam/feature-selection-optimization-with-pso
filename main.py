from .data_loader import mount_drive, load_dataset, preprocess_dataset, get_features_and_labels
from .nlp_setup import setup_nltk
from .visualization import setup_matplotlib

def main():
    # Mount Google Drive
    mount_drive()
    
    # Setup NLTK and visualization
    setup_nltk()
    setup_matplotlib()
    
    # Load and preprocess data
    dataset = load_dataset()
    df = preprocess_dataset(dataset)
    
    # Get features and labels
    emails, labels = get_features_and_labels(df)
    
    # Print dataset info
    print("\nDataset Information:")
    dataset.info()
    
    return emails, labels

if __name__ == "__main__":
    emails, labels = main()
