import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import random

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('ecommerce.db')
    conn.row_factory = sqlite3.Row
    return conn

# Check if database exists, if not, create it
if not os.path.exists('ecommerce.db'):
    st.warning("Database not found. Please run create_dummy_data.py first.")
    st.stop()

# Load BPR model if it exists
@st.cache_resource
def load_bpr_model():
    if os.path.exists('bpr_model.pkl'):
        with open('bpr_model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

bpr_model = load_bpr_model()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_user_id' not in st.session_state:
    # Randomly select a user for this session
    conn = get_db_connection()
    users = conn.execute('SELECT id FROM users').fetchall()
    conn.close()
    if users:
        st.session_state.current_user_id = random.choice(users)['id']
    else:
        st.session_state.current_user_id = 1  # Default user ID

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to get product recommendations based on query
def get_content_based_recommendations(query, top_n=5):
    conn = get_db_connection()
    
    # Get all in-stock products
    cursor = conn.execute('''
        SELECT id, name, category, color, occasion, price, description, rating, in_stock
        FROM products
        WHERE in_stock = 1
    ''')
    
    # Get column names from cursor description
    columns = [col[0] for col in cursor.description]
    
    # Fetch all rows and create DataFrame with explicit column names
    rows = cursor.fetchall()
    products_df = pd.DataFrame(rows, columns=columns)
    
    conn.close()
    
    if products_df.empty:
        return pd.DataFrame()
    
    # Create product features for similarity calculation
    products_df['features'] = products_df['name'] + ' ' + products_df['category'] + ' ' + \
                             products_df['color'] + ' ' + products_df['occasion'] + ' ' + \
                             products_df['description']
    
    # Preprocess the query and product features
    processed_query = preprocess_text(query)
    processed_features = products_df['features'].apply(preprocess_text)
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(processed_features)
    
    # Transform the query into TF-IDF vector
    query_vec = tfidf.transform([processed_query])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top N recommendations
    indices = cosine_sim.argsort()[-top_n:][::-1]
    
    # Filter recommendations with similarity score > 0
    filtered_indices = [idx for idx in indices if cosine_sim[idx] > 0]
    
    if not filtered_indices:
        # If no relevant products found, return empty DataFrame
        return pd.DataFrame()
    
    # Return the recommended products
    return products_df.iloc[filtered_indices].copy()

# Function to get personalized recommendations using BPR model
def get_personalized_recommendations(user_id, top_n=5):
    if bpr_model is None:
        return pd.DataFrame()
    
    # Get recommendations from BPR model
    try:
        # Adjust user_id for zero-based indexing in the model
        adjusted_user_id = user_id - 1
        if adjusted_user_id < 0 or adjusted_user_id >= bpr_model.n_users:
            return pd.DataFrame()
        
        recommended_item_ids = bpr_model.recommend(adjusted_user_id, n_recommendations=top_n)
        
        # Adjust item_ids back to one-based indexing for the database
        recommended_item_ids = [item_id + 1 for item_id in recommended_item_ids]
        
        if not recommended_item_ids:
            return pd.DataFrame()
        
        # Get product details from database
        conn = get_db_connection()
        placeholders = ', '.join(['?'] * len(recommended_item_ids))
        query = f'''
            SELECT id, name, category, color, occasion, price, description, rating, in_stock
            FROM products
            WHERE id IN ({placeholders}) AND in_stock = 1
        '''
        cursor = conn.execute(query, recommended_item_ids)
        
        # Get column names from cursor description
        columns = [col[0] for col in cursor.description]
        
        # Fetch all rows and create DataFrame with explicit column names
        rows = cursor.fetchall()
        products_df = pd.DataFrame(rows, columns=columns)
        
        conn.close()
        
        return products_df
    except Exception as e:
        st.error(f"Error getting personalized recommendations: {e}")
        return pd.DataFrame()

# Function to generate bot response
def generate_response(user_query, user_id):
    # Check for general questions
    general_questions = {
        r'difference between (.*) and (.*)': handle_difference_question,
        r'what (is|are) (.*)': handle_what_is_question,
        r'how (to|do) (.*)': handle_how_to_question,
        r'recommend|suggest|show': handle_recommendation_question
    }
    
    for pattern, handler in general_questions.items():
        if re.search(pattern, user_query.lower()):
            return handler(user_query, user_id)
    
    # Default to product recommendations
    return handle_recommendation_question(user_query, user_id)

def handle_difference_question(query, user_id):
    # Extract the items being compared
    match = re.search(r'difference between (.*) and (.*)', query.lower())
    if match:
        item1, item2 = match.groups()
        
        # Predefined differences for common comparisons
        differences = {
            ('sneakers', 'running shoes'): "Sneakers are general-purpose casual shoes, while running shoes are specifically designed for performance, with extra cushioning and support for running. Would you like recommendations for either category?",
            ('jeans', 'trousers'): "Jeans are typically made from denim and are more casual, while trousers are usually made from lighter fabrics and are more formal. Would you like to see some options?",
            ('t-shirt', 'shirt'): "T-shirts are casual, short-sleeved tops typically made of cotton, while shirts usually have buttons, collars, and can be more formal. Would you like recommendations for either?",
        }
        
        # Check both orders of the comparison
        if (item1.strip(), item2.strip()) in differences:
            return differences[(item1.strip(), item2.strip())]
        elif (item2.strip(), item1.strip()) in differences:
            return differences[(item2.strip(), item1.strip())]
        
        # Generic response if specific comparison not found
        return f"The main differences between {item1} and {item2} relate to their design, materials, and typical use cases. Would you like to see some options for either?"
    
    return "I'm not sure about that comparison. Could you be more specific or ask about our products?"

def handle_what_is_question(query, user_id):
    # Extract what the user is asking about
    match = re.search(r'what (is|are) (.*)', query.lower())
    if match:
        subject = match.group(2).strip()
        
        # Predefined explanations for common questions
        explanations = {
            'dress': "A dress is a one-piece garment typically worn by women or girls, consisting of a skirt with an attached bodice. Would you like to see our dress collection?",
            'jeans': "Jeans are casual pants made from denim or dungaree cloth, known for their durability and comfort. Would you like to see our jeans collection?",
            't-shirt': "A T-shirt is a casual top with short sleeves and no collar, typically made of cotton. Would you like to see our T-shirt collection?",
            'sneakers': "Sneakers are casual athletic shoes with rubber soles, designed for comfort and everyday wear. Would you like to see our sneaker collection?",
        }
        
        if subject in explanations:
            return explanations[subject]
        
        # Check if the subject is a category in our product data
        conn = get_db_connection()
        categories = [row[0].lower() for row in conn.execute('SELECT DISTINCT category FROM products').fetchall()]
        conn.close()
        
        for category in categories:
            if category.lower() in subject:
                return f"{category.capitalize()} is a type of clothing in our collection. Would you like to see our {category} options?"
        
        # Generic response if specific explanation not found
        return f"I don't have specific information about {subject}. Would you like to browse our product categories instead?"
    
    return "I'm not sure what you're asking about. Could you rephrase your question?"

def handle_how_to_question(query, user_id):
    # Extract what the user is asking about
    match = re.search(r'how (to|do) (.*)', query.lower())
    if match:
        subject = match.group(2).strip()
        
        # Predefined explanations for common questions
        explanations = {
            'style jeans': "Jeans can be styled in numerous ways! For casual looks, pair with t-shirts and sneakers. For a more dressed-up look, try a button-up shirt and boots or heels. Would you like to see our jeans collection?",
            'choose a dress': "When choosing a dress, consider the occasion, your body type, and personal style. For formal events, look at our elegant gowns. For casual outings, sundresses or shirt dresses work well. Would you like to see our dress collection?",
            'find my size': "To find your size, refer to our size guide which provides measurements for different body parts. If you're between sizes, we recommend sizing up for comfort. Would you like me to show you specific products?",
        }
        
        for key, explanation in explanations.items():
            if key in subject:
                return explanation
        
        # Generic response if specific explanation not found
        return f"I don't have specific guidance on how to {subject}. Would you like to browse our products instead?"
    
    return "I'm not sure what you're asking about. Could you rephrase your question?"

def handle_recommendation_question(query, user_id):
    # First try content-based recommendations based on the query
    content_recommendations = get_content_based_recommendations(query)
    
    # If no content-based recommendations, try personalized recommendations
    if content_recommendations.empty:
        personalized_recommendations = get_personalized_recommendations(user_id)
        
        if personalized_recommendations.empty:
            return "I couldn't find any products matching your request. Would you like to browse our categories or try a different search?"
        
        # Format the personalized recommendations
        response = "I couldn't find exact matches for your query, but based on your preferences, you might like these items:\n\n"
        for _, product in personalized_recommendations.iterrows():
            response += f"• **{product['name']}** - {product['color']} {product['category']} for {product['occasion']} occasions, ${product['price']:.2f} (Rating: {product['rating']}⭐)\n"
        
        response += "\nWould you like more specific recommendations or information about any of these products?"
        
        return response
    
    # Format the content-based recommendations
    response = "Here are some recommendations based on your request:\n\n"
    for _, product in content_recommendations.iterrows():
        response += f"• **{product['name']}** - {product['color']} {product['category']} for {product['occasion']} occasions, ${product['price']:.2f} (Rating: {product['rating']}⭐)\n"
    
    response += "\nWould you like more specific recommendations or information about any of these products?"
    
    return response

# Function to record user interaction
def record_interaction(user_id, product_id, interaction_type):
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO interactions (user_id, product_id, interaction_type)
            VALUES (?, ?, ?)
        ''', (user_id, product_id, interaction_type))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error recording interaction: {e}")
        return False

# Function to add new product
def add_new_product(name, category, color, occasion, price, description, in_stock, rating):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO products (name, category, color, occasion, price, description, in_stock, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, category, color, occasion, price, description, in_stock, rating))
        conn.commit()
        product_id = cursor.lastrowid
        conn.close()
        return f"Added new product: {name}", product_id
    except Exception as e:
        return f"Error adding product: {e}", None

# Main app layout
st.title("🛍️ AI Shopping Assistant")

# Display current user
st.sidebar.info(f"Current User ID: {st.session_state.current_user_id}")

# Sidebar for product management
with st.sidebar:
    st.header("Product Management")
    
    # Add new product form
    with st.expander("Add New Product"):
        with st.form("new_product_form"):
            name = st.text_input("Product Name")
            category = st.text_input("Category")
            color = st.text_input("Color")
            occasion = st.text_input("Occasion")
            price = st.number_input("Price", min_value=0.0, format="%.2f")
            description = st.text_area("Description")
            in_stock = st.checkbox("In Stock", value=True)
            rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
            
            submit_button = st.form_submit_button("Add Product")
            
            if submit_button and name and category:
                result, product_id = add_new_product(name, category, color, occasion, price, description, in_stock, rating)
                st.success(result)
    
    # View product database
    with st.expander("View Product Database"):
        conn = get_db_connection()
        products_df = pd.DataFrame(conn.execute('''
            SELECT id, name, category, color, price, in_stock
            FROM products
            ORDER BY id
        ''').fetchall())
        conn.close()
        
        st.dataframe(products_df)
    
    # Train BPR model
    if st.button("Train Recommendation Model"):
        with st.spinner("Training BPR model..."):
            try:
                from bpr_model import train_bpr_model
                train_bpr_model()
                st.success("BPR model trained successfully! Please refresh the page to load the new model.")
            except Exception as e:
                st.error(f"Error training model: {e}")

# Main chat interface
st.subheader("Chat with our AI Shopping Assistant")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask about products or recommendations...")

if user_query:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = generate_response(user_query, st.session_state.current_user_id)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display product categories for browsing
st.subheader("Browse by Category")
conn = get_db_connection()
categories = [row[0] for row in conn.execute('SELECT DISTINCT category FROM products').fetchall()]
conn.close()

# Create columns for categories
cols = st.columns(min(4, len(categories)))
for i, category in enumerate(categories):
    if cols[i % len(cols)].button(category):
        # Add category selection to chat history
        query = f"Show me {category}"
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Generate response
        response = generate_response(query, st.session_state.current_user_id)
        
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update the chat display
        st.rerun()

# Add information about the recommendation system
with st.expander("About the AI Shopping Assistant"):
    st.markdown("""
    ### How it works
    
    This AI Shopping Assistant uses two recommendation approaches:
    
    1. **Content-Based Filtering**: Recommends products based on your specific query and product attributes.
    
    2. **Bayesian Personalized Ranking (BPR)**: Provides personalized recommendations based on user interaction history.
    
    The system also uses natural language processing to understand your questions about products, styles, and shopping advice.
    
    ### Getting Started
    
    - Ask for product recommendations like "Show me black dresses for a party"
    - Ask questions like "What's the difference between sneakers and running shoes?"
    - Browse categories by clicking on the category buttons below
    
    Your interactions are recorded to improve future recommendations.
    """)

# Update requirements.txt to include scipy
# scipy is needed for the BPR model

# Function to display product recommendations
def display_recommendations(recommendations, msg_idx):
    if recommendations.empty:
        return
    
    cols = st.columns(3)
    
    for i, (_, product) in enumerate(recommendations.iterrows()):
        col = cols[i % 3]
        with col:
            with st.container():
                # Use proper DataFrame column access
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-title">{product.name}</div>
                    <div class="product-details">
                        <div>{product.color} {product.category}</div>
                        <div>For {product.occasion} occasions</div>
                        <div class="product-price">${product.price:.2f}</div>
                        <div>Rating: {product.rating}⭐</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add buttons for interaction
                col1, col2 = st.columns(2)
                with col1:
                    unique_like_key = f"like_{msg_idx}_{i}_{product.id}_{st.session_state.user_id}"
                    if st.button(f"👍 Like", key=unique_like_key):
                        record_feedback(product.id, 'like')
                with col2:
                    unique_cart_key = f"cart_{msg_idx}_{i}_{product.id}_{st.session_state.user_id}"
                    st.button(f"🛒 Add to Cart", key=unique_cart_key)
