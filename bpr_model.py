import numpy as np
import sqlite3
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import pickle
import os

class BPR:
    """
    Bayesian Personalized Ranking for implicit feedback data
    """
    def __init__(self, learning_rate=0.01, n_factors=20, n_iterations=100, reg=0.01):
        self.learning_rate = learning_rate
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg  # Regularization term
        
    def fit(self, user_item_matrix):
        """
        Train the BPR model
        
        Parameters:
        -----------
        user_item_matrix: scipy.sparse.csr_matrix
            User-item interaction matrix
        """
        self.n_users, self.n_items = user_item_matrix.shape
        
        # Initialize latent factors
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.item_biases = np.zeros(self.n_items)
        
        # Get all user-item interactions
        self.user_item_matrix = user_item_matrix
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Sample triplets (user, positive_item, negative_item)
            for u in range(self.n_users):
                # Get positive items for user u
                pos_items = user_item_matrix[u].indices
                
                if len(pos_items) == 0:
                    continue
                
                # Sample a positive item
                i = np.random.choice(pos_items)
                
                # Sample a negative item (not interacted with by user u)
                j = np.random.randint(0, self.n_items)
                while j in pos_items:
                    j = np.random.randint(0, self.n_items)
                
                # Compute prediction for positive and negative items
                x_ui = self.predict(u, i)
                x_uj = self.predict(u, j)
                
                # Compute BPR loss
                x_uij = x_ui - x_uj
                sigmoid = 1.0 / (1.0 + np.exp(-x_uij))
                
                # Update parameters with gradient descent
                # User latent factors
                grad_u = (1.0 - sigmoid) * (self.item_factors[i] - self.item_factors[j]) - self.reg * self.user_factors[u]
                self.user_factors[u] += self.learning_rate * grad_u
                
                # Positive item latent factors
                grad_i = (1.0 - sigmoid) * self.user_factors[u] - self.reg * self.item_factors[i]
                self.item_factors[i] += self.learning_rate * grad_i
                
                # Negative item latent factors
                grad_j = -(1.0 - sigmoid) * self.user_factors[u] - self.reg * self.item_factors[j]
                self.item_factors[j] += self.learning_rate * grad_j
                
                # Item biases
                self.item_biases[i] += self.learning_rate * ((1.0 - sigmoid) - self.reg * self.item_biases[i])
                self.item_biases[j] += self.learning_rate * (-(1.0 - sigmoid) - self.reg * self.item_biases[j])
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} completed")
    
    def predict(self, user_id, item_id):
        """
        Predict the preference of a user for an item
        
        Parameters:
        -----------
        user_id: int
            User ID
        item_id: int
            Item ID
            
        Returns:
        --------
        float
            Predicted preference score
        """
        return np.dot(self.user_factors[user_id], self.item_factors[item_id]) + self.item_biases[item_id]
    
    def recommend(self, user_id, n_recommendations=5, exclude_seen=True):
        """
        Recommend items for a user
        
        Parameters:
        -----------
        user_id: int
            User ID
        n_recommendations: int
            Number of recommendations to return
        exclude_seen: bool
            Whether to exclude items the user has already interacted with
            
        Returns:
        --------
        list
            List of recommended item IDs
        """
        # Get all items the user has interacted with
        if exclude_seen:
            seen_items = self.user_item_matrix[user_id].indices
        else:
            seen_items = []
        
        # Compute predictions for all items
        predictions = []
        for item_id in range(self.n_items):
            if item_id not in seen_items:
                predictions.append((item_id, self.predict(user_id, item_id)))
        
        # Sort predictions by score
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        return [item_id for item_id, _ in predictions[:n_recommendations]]

def load_data_from_db():
    """
    Load interaction data from SQLite database
    
    Returns:
    --------
    pandas.DataFrame
        Interaction data
    """
    conn = sqlite3.connect('ecommerce.db')
    
    # Load interactions
    interactions_df = pd.read_sql_query('''
    SELECT user_id, product_id, interaction_type
    FROM interactions
    ''', conn)
    
    # Load products
    products_df = pd.read_sql_query('''
    SELECT id, name, category, color, occasion, price, in_stock
    FROM products
    ''', conn)
    
    # Load users
    users_df = pd.read_sql_query('''
    SELECT id, username
    FROM users
    ''', conn)
    
    conn.close()
    
    return interactions_df, products_df, users_df

def create_user_item_matrix(interactions_df, n_users, n_items):
    """
    Create a user-item interaction matrix
    
    Parameters:
    -----------
    interactions_df: pandas.DataFrame
        Interaction data
    n_users: int
        Number of users
    n_items: int
        Number of items
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        User-item interaction matrix
    """
    # Create a binary interaction matrix (1 if user interacted with item, 0 otherwise)
    # We'll consider all interaction types as positive feedback
    user_item_df = interactions_df.groupby(['user_id', 'product_id']).size().reset_index(name='count')
    
    # Convert to sparse matrix
    user_item_matrix = csr_matrix((user_item_df['count'], 
                                  (user_item_df['user_id'] - 1, user_item_df['product_id'] - 1)),
                                 shape=(n_users, n_items))
    
    # Convert to binary matrix (1 if interaction exists, 0 otherwise)
    user_item_matrix.data = np.ones_like(user_item_matrix.data)
    
    return user_item_matrix

def train_bpr_model():
    """
    Train the BPR model and save it to disk
    """
    # Load data
    interactions_df, products_df, users_df = load_data_from_db()
    
    # Get number of users and items
    n_users = users_df['id'].max()
    n_items = products_df['id'].max()
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(interactions_df, n_users, n_items)
    
    # Initialize and train BPR model
    model = BPR(learning_rate=0.01, n_factors=20, n_iterations=100, reg=0.01)
    model.fit(user_item_matrix)
    
    # Save model to disk
    with open('bpr_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("BPR model trained and saved successfully")

if __name__ == "__main__":
    train_bpr_model() 
