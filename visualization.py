import matplotlib.pyplot as plt
from wordcloud import WordCloud

def setup_matplotlib():
    """Configure matplotlib for inline display."""
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except:
        plt.switch_backend('agg')

def create_wordcloud(text):
    """Create and return a WordCloud object."""
    return WordCloud(width=800, height=400, 
                    background_color='white',
                    min_font_size=10).generate(text)
