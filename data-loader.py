import pandas as pd
from google.colab import drive
from .config import DATA_PATH, SUBJECT_COLUMN

def mount_drive():
    """Mount Google Drive."""
    drive.mount('/content/drive')

def load_dataset():
    """Load the dataset from CSV file."""
    return pd.read_csv(DATA_PATH)

def preprocess_dataset(df):
    """Perform initial preprocessing on the dataset."""
    # Remove subject column
    df = df.drop(SUBJECT_COLUMN, axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates(keep='first')
    
    return df

def get_features_and_labels(df):
    """Extract features and labels from the preprocessed dataset."""
    from .config import MESSAGE_COLUMN, LABEL_COLUMN
    
    emails = df[MESSAGE_COLUMN]
    labels = df[LABEL_COLUMN]
    
    return emails, labels
