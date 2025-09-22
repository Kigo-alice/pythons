# cord19_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CORD19Analyzer:
    def __init__(self, file_path='metadata.csv'):
        """Initialize the analyzer with the dataset path"""
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        """Load the metadata CSV file"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found. Please download from Kaggle.")
            return False
    
    def basic_exploration(self):
        """Perform basic data exploration"""
        if self.df is None:
            print("Please load data first")
            return
        
        print("\n=== BASIC DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData types:")
        print(self.df.dtypes)
        
        print("\nMissing values by column:")
        missing_data = self.df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        print("\nBasic statistics for numerical columns:")
        print(self.df.describe())
        
    def clean_data(self):
        """Clean and prepare the data for analysis"""
        if self.df is None:
            print("Please load data first")
            return
        
        print("\n=== DATA CLEANING ===")
        self.df_cleaned = self.df.copy()
        
        # Handle publication date
        self.df_cleaned['publish_time'] = pd.to_datetime(
            self.df_cleaned['publish_time'], errors='coerce'
        )
        self.df_cleaned['year'] = self.df_cleaned['publish_time'].dt.year
        
        # Fill missing years with mode
        mode_year = self.df_cleaned['year'].mode()[0]
        self.df_cleaned['year'] = self.df_cleaned['year'].fillna(mode_year)
        
        # Create abstract word count
        self.df_cleaned['abstract_word_count'] = self.df_cleaned['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        # Create title word count
        self.df_cleaned['title_word_count'] = self.df_cleaned['title'].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        # Clean journal names
        self.df_cleaned['journal_clean'] = self.df_cleaned['journal'].fillna('Unknown')
        self.df_cleaned['journal_clean'] = self.df_cleaned['journal_clean'].str.title()
        
        print(f"Data cleaning completed. Cleaned dataset shape: {self.df_cleaned.shape}")
        
    def analyze_publications_over_time(self):
        """Analyze publication trends over time"""
        if self.df_cleaned is None:
            print("Please clean data first")
            return
        
        yearly_counts = self.df_cleaned['year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        yearly_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of COVID-19 Publications by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('publications_by_year.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return yearly_counts
    
    def analyze_top_journals(self, top_n=15):
        """Analyze top publishing journals"""
        if self.df_cleaned is None:
            print("Please clean data first")
            return
        
        journal_counts = self.df_cleaned['journal_clean'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        journal_counts.sort_values().plot(kind='barh', color='lightcoral')
        plt.title(f'Top {top_n} Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Publications')
        plt.tight_layout()
        plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return journal_counts
    
    def create_title_wordcloud(self):
        """Create a word cloud from paper titles"""
        if self.df_cleaned is None:
            print("Please clean data first")
            return
        
        # Combine all titles
        titles = ' '.join(self.df_cleaned['title'].dropna().astype(str))
        
        # Clean the text
        titles_clean = re.sub(r'[^\w\s]', '', titles.lower())
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(titles_clean)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('title_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_word_frequency(self, top_n=20):
        """Analyze most frequent words in titles"""
        if self.df_cleaned is None:
            print("Please clean data first")
            return
        
        titles = ' '.join(self.df_cleaned['title'].dropna().astype(str))
        words = re.findall(r'\b[a-zA-Z]{4,}\b', titles.lower())
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'from', 'have', 'were', 'been', 'they', 'their', 
                     'what', 'when', 'which', 'than', 'into', 'such', 'more', 'these', 'those'}
        words_filtered = [word for word in words if word not in stop_words]
        
        word_freq = Counter(words_filtered).most_common(top_n)
        
        words, counts = zip(*word_freq)
        
        plt.figure(figsize=(12, 8))
        plt.barh(words, counts, color='lightgreen')
        plt.title(f'Top {top_n} Most Frequent Words in Titles', fontsize=16, fontweight='bold')
        plt.xlabel('Frequency')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('word_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return word_freq
    
    def generate_report(self):
        """Generate a summary report of findings"""
        if self.df_cleaned is None:
            print("Please clean data first")
            return
        
        print("\n" + "="*50)
        print("CORD-19 DATA ANALYSIS REPORT")
        print("="*50)
        
        print(f"\nTotal number of papers: {len(self.df_cleaned):,}")
        print(f"Date range: {int(self.df_cleaned['year'].min())} - {int(self.df_cleaned['year'].max())}")
        
        # Papers with abstracts
        has_abstract = self.df_cleaned['abstract'].notna().sum()
        print(f"Papers with abstracts: {has_abstract:,} ({has_abstract/len(self.df_cleaned)*100:.1f}%)")
        
        # Average word counts
        avg_title_words = self.df_cleaned['title_word_count'].mean()
        avg_abstract_words = self.df_cleaned[self.df_cleaned['abstract_word_count'] > 0]['abstract_word_count'].mean()
        print(f"Average title length: {avg_title_words:.1f} words")
        print(f"Average abstract length: {avg_abstract_words:.1f} words")
        
        # Top journals
        top_journals = self.df_cleaned['journal_clean'].value_counts().head(5)
        print("\nTop 5 journals:")
        for journal, count in top_journals.items():
            print(f"  - {journal}: {count:,} papers")

# Main execution
if __name__ == "__main__":
    analyzer = CORD19Analyzer('metadata.csv')
    
    if analyzer.load_data():
        analyzer.basic_exploration()
        analyzer.clean_data()
        analyzer.analyze_publications_over_time()
        analyzer.analyze_top_journals()
        analyzer.create_title_wordcloud()
        analyzer.analyze_word_frequency()
        analyzer.generate_report()