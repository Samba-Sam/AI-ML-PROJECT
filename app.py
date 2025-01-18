from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# load data
popular_df = pd.read_pickle('popular.pkl')
pt = pd.read_pickle('pt.pkl')
items = pd.read_pickle('books.pkl')
similarity_scores = pd.read_pickle('similarity_scores.pkl')

# flask 
app = Flask(__name__)
app.secret_key = ''  

# Chatbot  https://www.youtube.com/watch?v=d0o89z134CQ
template = """
Answer the question below.

Here is the conversation history: {context}

Question : {question}

Answer:
"""

model = OllamaLLM(model="gemma2:2b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


@app.route('/')
def index():
    
    images = popular_df['Image-URL-M'].apply(lambda x: x if isinstance(x, str) and x.strip() != "" else "1.jpeg").values
    return render_template('index.html',
                           product_name=list(popular_df['Grocery-Title'].values),
                           author=list(popular_df['Book-Price'].values),
                           image=list(images),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values))

# Chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    context = request.json.get("context", "")

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Process 
    result = chain.invoke({"context": context, "question": user_input})
    
    return jsonify({"response": result})

# Recommendation UI route
@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

# Recommendation processing route
@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_inputt = request.form.get('user_input').lower()
    ppp = np.where(pt.index == user_inputt)[0]
    if len(ppp) > 0:
        user_input = request.form.get('user_input').lower()
        index = np.where(pt.index == user_input)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[0:11]

        data = []
        for i in similar_items:
            item = []
            temp_df = items[items['Grocery-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Grocery-Title')['Grocery-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Grocery-Title')['Book-Price'].values))
            # Fallback to "1.jpeg" if image URL is invalid
            image_url = temp_df.drop_duplicates('Grocery-Title')['Image-URL-M'].values
            item.append(image_url[0] if len(image_url) > 0 and isinstance(image_url[0], str) and image_url[0].strip() != "" else "1.jpeg")

            data.append(item)

        return render_template('recommend.html', data=data)
    else:
        return render_template('recommend.html', data=None)

'''
@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    item_id = request.form.get('item_id')  # Item ID from the form
    item = items[items['Grocery-Title'] == item_id].iloc[0]

    # Check if cart exists in session
    if 'cart' not in session:
        session['cart'] = []

    # Add item to cart
    cart_item = {
        'title': item['Grocery-Title'],
        'price': item['Book-Price'],
        'image': item['Image-URL-M'] if isinstance(item['Image-URL-M'], str) else "1.jpeg"
    }
    session['cart'].append(cart_item)

    session.modified = True
    return redirect(url_for('index'))

# View the shopping cart
@app.route('/cart')
def view_cart():
    if 'cart' not in session or len(session['cart']) == 0:
        return render_template('cart.html', cart_items=[], total_price=0)

    total_price = sum([float(item['price']) for item in session['cart']])
    return render_template('cart.html', cart_items=session['cart'], total_price=total_price)

# Remove item from cart
@app.route('/remove_from_cart/<int:item_index>')
def remove_from_cart(item_index):
    if 'cart' in session:
        session['cart'].pop(item_index)
        session.modified = True
    return redirect(url_for('view_cart'))'''

# Run
if __name__ == '__main__':
    app.run(debug=True)