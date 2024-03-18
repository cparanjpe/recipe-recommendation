from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model
loaded_model = load_model('recipe_model.h5')

# Recipe data and MultiLabelBinarizer
recipes = {
    "Spaghetti Carbonara": ["spaghetti", "eggs", "pecorino cheese", "guanciale", "black pepper", "salt"],
    "Chicken Tikka Masala": ["chicken", "yogurt", "tomato puree", "onion", "garlic", "ginger", "garam masala", "cumin", "coriander", "paprika", "chili powder", "cream", "butter", "salt"],
    "Caesar Salad": ["romaine lettuce", "parmesan cheese", "croutons", "olive oil", "lemon juice", "garlic", "dijon mustard", "worcestershire sauce", "anchovy paste", "salt", "black pepper"],
    "Beef Bourguignon": ["beef", "bacon", "onion", "carrot", "garlic", "red wine", "beef broth", "tomato paste", "bay leaf", "thyme", "mushrooms", "butter", "flour", "salt", "black pepper"],
    "Sushi": ["sushi rice", "nori", "fish (e.g., tuna, salmon)", "avocado", "cucumber", "soy sauce", "wasabi", "pickled ginger"],
    "Margherita Pizza": ["pizza dough", "tomato sauce", "mozzarella cheese", "fresh basil leaves", "olive oil", "salt", "black pepper"],
    "Mushroom Risotto": ["arborio rice", "mushrooms", "onion", "garlic", "white wine", "vegetable broth", "butter", "parmesan cheese", "salt", "black pepper"],
    "Chocolate Cake": ["flour", "sugar", "cocoa powder", "baking powder", "baking soda", "salt", "eggs", "milk", "vegetable oil", "vanilla extract", "boiling water"],
    "Beef Stroganoff": ["beef", "onion", "mushrooms", "butter", "flour", "beef broth", "sour cream", "dijon mustard", "worcestershire sauce", "salt", "black pepper"],
    "Pad Thai": ["rice noodles", "shrimp", "tofu", "eggs", "bean sprouts", "garlic chives", "peanuts", "vegetable oil", "fish sauce", "soy sauce", "sugar", "lime", "chili flakes"],
    "Hamburger": ["ground beef", "hamburger buns", "lettuce", "tomato", "onion", "pickle slices", "cheese", "ketchup", "mustard", "mayonnaise", "salt", "black pepper"],
    "Chicken Curry": ["chicken", "onion", "garlic", "ginger", "tomato", "coconut milk", "curry powder", "turmeric", "cumin", "coriander", "cayenne pepper", "salt", "black pepper"],
    "Pasta Primavera": ["pasta", "olive oil", "garlic", "bell peppers", "broccoli", "carrots", "zucchini", "tomato", "cream", "parmesan cheese", "salt", "black pepper"],
    "Lasagna": ["lasagna noodles", "ricotta cheese", "mozzarella cheese", "parmesan cheese", "meat sauce (beef, tomato sauce, onion, garlic, Italian seasoning)", "egg", "salt", "black pepper"],
    "Gazpacho": ["tomatoes", "cucumber", "red bell pepper", "red onion", "garlic", "olive oil", "red wine vinegar", "tomato juice", "salt", "black pepper"],
    "Ratatouille": ["eggplant", "zucchini", "bell peppers", "tomatoes", "onion", "garlic", "olive oil", "thyme", "basil", "oregano", "salt", "black pepper"],
    "Beef Tacos": ["beef", "taco seasoning", "tortillas", "lettuce", "tomato", "onion", "cheese", "sour cream", "salsa"],
    "Shakshuka": ["tomatoes", "bell peppers", "onion", "garlic", "eggs", "paprika", "cumin", "cayenne pepper", "salt", "black pepper", "olive oil", "parsley"],
    "Ramen": ["ramen noodles", "broth (chicken, pork, or vegetable)", "egg", "green onions", "bamboo shoots", "seaweed", "mushrooms", "tofu", "soy sauce", "sesame oil"],
    "Biryani": ["basmati rice", "chicken", "yogurt", "onion", "garlic", "ginger", "garam masala", "cumin", "coriander", "turmeric", "cayenne pepper", "bay leaf", "cardamom", "cloves", "raisins", "cashews", "saffron", "salt"],
    "Goulash": ["beef", "onion", "bell pepper", "tomato", "paprika", "caraway seeds", "garlic", "beef broth", "potatoes", "salt", "black pepper"],
    "Pho": ["beef broth", "rice noodles", "beef slices", "onion", "ginger", "star anise", "cinnamon", "cloves", "fish sauce", "lime", "bean sprouts", "basil", "mint leaves", "cilantro", "jalapeno", "hoisin sauce", "sriracha"],
    "Pancakes": ["flour", "sugar", "baking powder", "salt", "milk", "egg", "butter", "vanilla extract", "maple syrup"],
    "Margarita Cocktail": ["tequila", "triple sec", "lime juice", "simple syrup", "salt (for rimming)"]
}  # Your recipes dictionary here
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(recipes.values())
y = np.arange(len(recipes))

def get_recipe(ingredients):
    # Transform input ingredients into binary format
    input_vec = mlb.transform([ingredients])

    # Predict the recipe index
    recipe_index = np.argmax(loaded_model.predict(input_vec))

    # Get the recipe name from the index
    recipe_name = list(recipes.keys())[recipe_index]
    
    return recipe_name

@app.route('/recommend_recipe', methods=['POST'])
def recommend_recipe():
    ingredients = request.json.get('ingredients', [])
    
    # Validate input
    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400
    
    # Get recommended recipe
    recommended_recipe = get_recipe(ingredients)
    
    return jsonify({"recipe": recommended_recipe})

@app.route('/home')
def home():
    return jsonify({"msg": "hello world !"})
if __name__ == '__main__':
    app.run(debug=True)
