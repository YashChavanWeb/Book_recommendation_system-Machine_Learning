### 📘 README: Book Recommendation System

---

#### 📖 Overview

This Book Recommendation System suggests similar books based on a user's selection using collaborative filtering and a trained KNN model. It enhances user experience with a visually engaging Streamlit interface, only recommending books with ratings above 3 and available cover images.

---

#### 🧠 Features

* ML-based book similarity search using K-Nearest Neighbors (KNN)
* Interactive UI built with Streamlit
* Searchable sidebar for selecting books
* Displays cover image, author, rating, and publication year
* CSS-enhanced styling for better visual presentation
* Filters recommendations by rating and image availability

---

#### 🔧 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yashcahvanweb/Book_recommendation_system-Machine_Learning.git
   cd Book_recommendation_system-Machine_Learning
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**

   ```bash
   streamlit run app.py
   ```

---

#### 📁 Project Structure

```
├── artifacts/
│   ├── model.pkl
│   ├── book_name.pkl
│   ├── final_rating.pkl
│   └── book_pivot.pkl
├── styles.css
├── app.py
├── README.md
└── requirements.txt
```

---

#### 🧪 Requirements

* Python 3.7+
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Pickle

---

#### 📸 Sample Output

* Book cards with titles, authors, rating stars, and background images
* Clean, responsive layout for recommendations
* Sidebar for user search and selection

---

#### 🚀 Future Improvements

* Add user-based collaborative filtering
* Include genre filters or personalized profiles
* Integrate external book APIs for real-time metadata
