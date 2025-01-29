import nltk
from .config import NLTK_DOWNLOADS

def setup_nltk():
    """Download required NLTK resources."""
    for resource in NLTK_DOWNLOADS:
        nltk.download(resource)

def get_stopwords():
    """Get stopwords from NLTK."""
    from nltk.corpus import stopwords
    return stopwords.words('english')
