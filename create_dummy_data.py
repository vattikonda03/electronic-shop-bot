import sqlite3
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Create a connection to the SQLite database
conn = sqlite3.connect('ecommerce.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    color TEXT NOT NULL,
    occasion TEXT NOT NULL,
    price REAL NOT NULL,
    description TEXT NOT NULL,
    in_stock BOOLEAN NOT NULL,
    rating REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    interaction_type TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (product_id) REFERENCES products (id)
)
''')

# Generate product data
product_data = {
    'name': [
        'Elegant Black Party Dress', 'Casual Black Dress', 'Formal Black Gown',
        'Blue Denim Jeans', 'Slim Fit Black Jeans', 'Distressed Blue Jeans',
        'White Cotton T-shirt', 'Black Graphic T-shirt', 'Striped Polo T-shirt',
        'Red Running Shoes', 'Black Leather Sneakers', 'White Canvas Sneakers',
        'Floral Summer Dress', 'Pastel Pink Sundress', 'Pastel Blue Beach Outfit',
        'Khaki Chino Pants', 'Black Formal Trousers', 'Grey Sweatpants',
        'Navy Blue Blazer', 'Black Leather Jacket', 'Denim Jacket',
        'White Linen Shirt', 'Blue Oxford Shirt', 'Checkered Flannel Shirt',
        'Brown Leather Boots', 'Black Ankle Boots', 'Hiking Boots',
        'Floral Print Blouse', 'White Silk Blouse', 'Striped Button-up Blouse'
    ],
    'category': [
        'Dress', 'Dress', 'Dress',
        'Jeans', 'Jeans', 'Jeans',
        'T-shirt', 'T-shirt', 'T-shirt',
        'Shoes', 'Shoes', 'Shoes',
        'Dress', 'Dress', 'Outfit',
        'Pants', 'Pants', 'Pants',
        'Outerwear', 'Outerwear', 'Outerwear',
        'Shirt', 'Shirt', 'Shirt',
        'Boots', 'Boots', 'Boots',
        'Blouse', 'Blouse', 'Blouse'
    ],
    'color': [
        'Black', 'Black', 'Black',
        'Blue', 'Black', 'Blue',
        'White', 'Black', 'Striped',
        'Red', 'Black', 'White',
        'Floral', 'Pink', 'Blue',
        'Khaki', 'Black', 'Grey',
        'Navy', 'Black', 'Blue',
        'White', 'Blue', 'Checkered',
        'Brown', 'Black', 'Brown',
        'Floral', 'White', 'Striped'
    ],
    'occasion': [
        'Party', 'Casual', 'Formal',
        'Casual', 'Casual', 'Casual',
        'Casual', 'Casual', 'Casual',
        'Sports', 'Casual', 'Casual',
        'Summer', 'Summer', 'Vacation',
        'Casual', 'Formal', 'Casual',
        'Formal', 'Casual', 'Casual',
        'Casual', 'Casual', 'Casual',
        'Casual', 'Casual', 'Outdoor',
        'Casual', 'Formal', 'Casual'
    ],
    'price': [
        89.99, 49.99, 129.99,
        59.99, 69.99, 79.99,
        19.99, 24.99, 29.99,
        89.99, 79.99, 49.99,
        69.99, 59.99, 79.99,
        49.99, 69.99, 39.99,
        99.99, 129.99, 89.99,
        59.99, 69.99, 49.99,
        119.99, 99.99, 129.99,
        49.99, 69.99, 59.99
    ],
    'description': [
        'A stunning black dress perfect for parties and special occasions.',
        'A comfortable black dress for everyday casual wear.',
        'An elegant black gown for formal events and galas.',
        'Classic blue denim jeans for everyday casual wear.',
        'Stylish black jeans with a slim fit design.',
        'Trendy distressed blue jeans for a casual look.',
        'Essential white cotton t-shirt for everyday wear.',
        'Cool black t-shirt with graphic design.',
        'Classic striped polo t-shirt for a smart casual look.',
        'High-performance red running shoes with cushioned soles.',
        'Sleek black leather sneakers for a stylish casual look.',
        'Classic white canvas sneakers for a clean, casual style.',
        'Beautiful floral dress perfect for summer days.',
        'Cute pastel pink sundress ideal for warm weather.',
        'Comfortable pastel blue outfit perfect for beach vacations.',
        'Versatile khaki chino pants for casual or smart casual occasions.',
        'Essential black trousers for formal and business settings.',
        'Comfortable grey sweatpants for lounging and casual wear.',
        'Classic navy blue blazer for formal and business occasions.',
        'Stylish black leather jacket for a cool, edgy look.',
        'Versatile denim jacket that pairs well with many outfits.',
        'Crisp white linen shirt perfect for warm weather.',
        'Classic blue Oxford shirt for a timeless, smart look.',
        'Cozy checkered flannel shirt for casual, relaxed days.',
        'Durable brown leather boots with classic styling.',
        'Fashionable black ankle boots that complement many outfits.',
        'Rugged hiking boots designed for outdoor adventures.',
        'Feminine floral print blouse for a pretty, casual look.',
        'Elegant white silk blouse for formal and business settings.',
        'Classic striped button-up blouse for a smart casual style.'
    ],
    'in_stock': [
        True, True, False,
        True, True, True,
        True, False, True,
        True, True, True,
        False, True, True,
        True, True, True,
        False, True, True,
        True, True, False,
        True, True, True,
        False, True, True
    ],
    'rating': [
        4.5, 4.2, 4.8,
        4.0, 4.3, 3.9,
        4.1, 4.4, 3.8,
        4.6, 4.7, 4.2,
        4.3, 4.1, 4.5,
        3.9, 4.2, 4.0,
        4.8, 4.6, 4.3,
        4.1, 4.4, 4.2,
        4.5, 4.3, 4.7,
        3.9, 4.6, 4.2
    ]
}

# Insert product data
for i in range(len(product_data['name'])):
    cursor.execute('''
    INSERT INTO products (name, category, color, occasion, price, description, in_stock, rating)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        product_data['name'][i],
        product_data['category'][i],
        product_data['color'][i],
        product_data['occasion'][i],
        product_data['price'][i],
        product_data['description'][i],
        product_data['in_stock'][i],
        product_data['rating'][i]
    ))

# Generate user data
usernames = [
    'user1', 'user2', 'user3', 'user4', 'user5',
    'user6', 'user7', 'user8', 'user9', 'user10',
    'user11', 'user12', 'user13', 'user14', 'user15',
    'user16', 'user17', 'user18', 'user19', 'user20'
]

emails = [f"{username}@example.com" for username in usernames]

# Insert user data
for i in range(len(usernames)):
    cursor.execute('''
    INSERT INTO users (username, email)
    VALUES (?, ?)
    ''', (usernames[i], emails[i]))

# Generate interaction data
interaction_types = ['view', 'click', 'add_to_cart', 'purchase', 'rate']
weights = [0.5, 0.2, 0.15, 0.1, 0.05]  # Probability weights for interaction types

# Generate random timestamps within the last 30 days
now = datetime.now()
start_date = now - timedelta(days=30)

# Generate 500 random interactions
for _ in range(500):
    user_id = random.randint(1, len(usernames))
    product_id = random.randint(1, len(product_data['name']))
    interaction_type = random.choices(interaction_types, weights=weights, k=1)[0]
    
    # Generate random timestamp
    random_days = random.randint(0, 30)
    random_seconds = random.randint(0, 86400)  # Seconds in a day
    timestamp = start_date + timedelta(days=random_days, seconds=random_seconds)
    
    cursor.execute('''
    INSERT INTO interactions (user_id, product_id, interaction_type, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (user_id, product_id, interaction_type, timestamp))

# Commit changes and close connection
conn.commit()
conn.close()

print("Dummy data created successfully in ecommerce.db") 
