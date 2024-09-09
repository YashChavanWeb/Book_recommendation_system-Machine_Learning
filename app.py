import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_name = pickle.load(open('artifacts/book_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

# Function to fetch book posters
def fetch_poster(suggestion):
    book_names = []
    ids_index = []
    poster_urls = []

    for book_id in suggestion:
        book_names.append(book_pivot.index[book_id])

    for name in book_names[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['img_url']
        print(f"Fetched URL: {url}")  # Debug print statement
        if pd.notna(url):  # Ensure URL is not NaN
            poster_urls.append(url)
        else:
            poster_urls.append(None)  # Handle missing images

    return poster_urls

# Function to recommend books with rating filter
def recommend_books(book_name, n_recommendations=20):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n_recommendations+1)
    
    poster_urls = fetch_poster(suggestion)
    
    for i in range(len(suggestion[0])):
        books = book_pivot.index[suggestion[0][i]]
        book_list.append(books)

    # Filter books with rating > 3 and with available images
    filtered_books = []
    filtered_posters = []
    for book, poster in zip(book_list, poster_urls):
        if poster is not None:  # Only include books with an image
            book_details = final_rating[final_rating['title'] == book].iloc[0]
            if book_details['rating'] > 3:
                filtered_books.append(book)
                filtered_posters.append(poster)

    return filtered_books, filtered_posters

# UI enhancements
st.set_page_config(page_title="Book Recommendation System", page_icon=":books:", layout="wide")

# Custom CSS
st.markdown(
    """
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" />
    <style>
    .stButton>button {
        font-family: 'Poppins', Arial;
    }
    </style>
    """, unsafe_allow_html=True
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css('styles.css')


# Main Layout
st.markdown('<h1 class="title-style">Book Recommendation System<span class="material-symbols-rounded">book_5</span></h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subheader-style">Using Machine Learning to Suggest the Best Books for You</h3>', unsafe_allow_html=True)

# Sidebar for book selection
st.sidebar.header('Select a Book')
search_query = st.sidebar.text_input("Search for a book")
if search_query:
    filtered_books = [book for book in book_name if search_query.lower() in book.lower()]
    selected_book = st.sidebar.selectbox("Type or select a book", filtered_books)
else:
    selected_book = st.sidebar.selectbox("Type or select a book", book_name)

# Sidebar button
if st.sidebar.button('Show Recommendations'):
    recommended_books, poster_urls = recommend_books(selected_book, n_recommendations=20)
    
    if recommended_books:
        st.markdown('<h4 class="bks">Recommended Books :)</h4>', unsafe_allow_html=True)
        
        st.markdown('<div class="recommendation-section">', unsafe_allow_html=True)
        for i in range(len(recommended_books)):
            book_title = recommended_books[i]
            book_details = final_rating[final_rating['title'] == book_title].iloc[0]
            
            st.markdown(
                f"""
                <div class="recommendation-item">
                    <div class="card" style="background-image: url('{poster_urls[i]}')">
                        <img src="{poster_urls[i]}" class="image"/>
                        <section class="details">
                            <h2>{book_title}</h2>
                                <section>
                                    <p><strong>Author:</strong><i> {book_details['author']}</i></p>
                                    <p><strong>Rating:</strong> {book_details['rating']}<span class="material-symbols-rounded">star</span></p>
                                    <p class="bleh" ><strong>Year: </strong>{book_details['year']}</p>
                                </section>
                        </section>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("No recommendations found with a rating greater than 3 and with available images.", unsafe_allow_html=True)