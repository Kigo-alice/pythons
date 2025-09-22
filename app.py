# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitCORD19App:
    def __init__(self):
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the data"""
        try:
            self.df = pd.read_csv('metadata.csv')
            
            # Data cleaning
            self.df['publish_time'] = pd.to_datetime(self.df['publish_time'], errors='coerce')
            self.df['year'] = self.df['publish_time'].dt.year
            self.df['year'] = self.df['year'].fillna(self.df['year'].mode()[0] if not self.df['year'].mode().empty else 2020)
            
            self.df['abstract_word_count'] = self.df['abstract'].apply(
                lambda x: len(str(x).split()) if pd.notnull(x) else 0
            )
            self.df['title_word_count'] = self.df['title'].apply(
                lambda x: len(str(x).split()) if pd.notnull(x) else 0
            )
            self.df['journal_clean'] = self.df['journal'].fillna('Unknown').str.title()
            
        except FileNotFoundError:
            st.error(" metadata.csv file not found. Please make sure it's in the same directory.")
            st.info(" Download the dataset from: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title(" Controls")
        
        st.sidebar.subheader("Data Filters")
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=int(self.df['year'].min()),
            max_value=int(self.df['year'].max()),
            value=(2020, 2021)
        )
        
        min_abstract_words = st.sidebar.slider(
            "Minimum Abstract Words",
            min_value=0,
            max_value=500,
            value=0
        )
        
        top_n_journals = st.sidebar.slider(
            "Number of Top Journals to Show",
            min_value=5,
            max_value=20,
            value=10
        )
        
        return year_range, min_abstract_words, top_n_journals
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<div class="main-header">📚 CORD-19 COVID-19 Research Explorer</div>', 
                   unsafe_allow_html=True)
        
        st.write("""
        This interactive dashboard explores the CORD-19 dataset, containing metadata about COVID-19 research papers. 
        Use the controls in the sidebar to filter the data and explore different aspects of the research landscape.
        """)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Papers", f"{len(self.df):,}")
        with col2:
            st.metric("Date Range", f"{int(self.df['year'].min())}-{int(self.df['year'].max())}")
        with col3:
            has_abstract = self.df['abstract'].notna().sum()
            st.metric("Papers with Abstracts", f"{has_abstract:,}")
        with col4:
            avg_words = self.df['title_word_count'].mean()
            st.metric("Avg Title Words", f"{avg_words:.1f}")
    
    def render_publications_over_time(self, year_range):
        """Render publications over time chart"""
        st.markdown('<div class="section-header">Publications Over Time</div>', 
                   unsafe_allow_html=True)
        
        filtered_df = self.df[
            (self.df['year'] >= year_range[0]) & 
            (self.df['year'] <= year_range[1])
        ]
        
        yearly_counts = filtered_df['year'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        yearly_counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Number of Publications by Year', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    def render_top_journals(self, top_n):
        """Render top journals chart"""
        st.markdown('<div class="section-header">Top Publishing Journals</div>', 
                   unsafe_allow_html=True)
        
        journal_counts = self.df['journal_clean'].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        journal_counts.sort_values().plot(kind='barh', color='lightcoral', ax=ax)
        ax.set_title(f'Top {top_n} Journals Publishing COVID-19 Research', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Publications')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    def render_word_analysis(self):
        """Render word frequency analysis"""
        st.markdown('<div class="section-header">🔤 Word Frequency Analysis</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Word Cloud - Paper Titles")
            titles = ' '.join(self.df['title'].dropna().astype(str))
            titles_clean = re.sub(r'[^\w\s]', '', titles.lower())
            
            wordcloud = WordCloud(width=400, height=300, background_color='white',
                                max_words=50, colormap='viridis').generate(titles_clean)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Most Frequent Words in Titles', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top 15 Words in Titles")
            titles = ' '.join(self.df['title'].dropna().astype(str))
            words = re.findall(r'\b[a-zA-Z]{4,}\b', titles.lower())
            
            stop_words = {'this', 'that', 'with', 'from', 'have', 'were', 'been', 
                         'they', 'their', 'what', 'when', 'which', 'than', 'into'}
            words_filtered = [word for word in words if word not in stop_words]
            
            word_freq = Counter(words_filtered).most_common(15)
            words, counts = zip(*word_freq)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(words, counts, color='lightgreen')
            ax.set_xlabel('Frequency')
            ax.set_title('Top 15 Words in Paper Titles', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            st.pyplot(fig)
    
    def render_data_sample(self):
        """Render a sample of the data"""
        st.markdown('<div class="section-header">📋 Data Sample</div>', 
                   unsafe_allow_html=True)
        
        st.dataframe(
            self.df[['title', 'journal_clean', 'year', 'abstract_word_count']].head(100),
            use_container_width=True
        )
    
    def run(self):
        """Main method to run the app"""
        if self.df is None:
            return
        
        self.render_header()
        
        # Get sidebar controls
        year_range, min_abstract_words, top_n_journals = self.render_sidebar()
        
        # Apply filters
        filtered_df = self.df[
            (self.df['year'] >= year_range[0]) & 
            (self.df['year'] <= year_range[1]) &
            (self.df['abstract_word_count'] >= min_abstract_words)
        ]
        
        # Update dataframe with filters
        original_len = len(self.df)
        self.df = filtered_df
        
        # Render visualizations
        self.render_publications_over_time(year_range)
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_top_journals(top_n_journals)
        
        with col2:
            st.markdown('<div class="section-header">📊 Abstract Length Distribution</div>', 
                       unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            has_abstract = self.df[self.df['abstract_word_count'] > 0]
            ax.hist(has_abstract['abstract_word_count'], bins=30, color='orange', alpha=0.7)
            ax.set_xlabel('Abstract Word Count')
            ax.set_ylabel('Number of Papers')
            ax.set_title('Distribution of Abstract Lengths', fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        self.render_word_analysis()
        self.render_data_sample()
        
        # Reset to original data for next render
        self.load_data()

# Run the app
if __name__ == "__main__":
    app = StreamlitCORD19App()
    app.run()