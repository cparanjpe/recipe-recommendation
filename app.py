from flask import Flask, request, jsonify
from flask_cors import CORS 
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  
# Load the saved model
loaded_model = load_model('new_recipe_model.h5')
recipes = {

 "Spaghetti Carbonara": {
"spaghetti": 10,
 "eggs": 8,
 "cheese": 9,
 "black pepper": 6,
 "salt": 5
 },

 "Chicken Tikka Masala": {
 "chicken": 10,
"yogurt": 6,
 "tomatoes": 7,
"onion": 7,
"garlic": 7,
 "ginger": 6,
"garam masala": 8,
"cumin": 7,
 "coriander": 7,
"paprika": 6,
"chilli powder": 6,
"cream": 5,
"butter": 5,
"salt": 5
},

"Vegetable Stir Fry": {
        "bell peppers": 8,
        "broccoli": 7,
        "carrots": 7,
        "peas": 6,
        "onion": 6,
        "garlic": 6,
        "ginger": 5,
        "soy sauce": 8,
        "salt": 4,
        "black pepper": 4
    },

    "Caprese Salad": {
        "tomatoes": 9,
        "cheese": 9,
        "basil leaves": 8,
        "olive oil": 7,
        "vinegar": 6,
        "salt": 4,
        "black pepper": 4
    },

    "Vegetable Curry": {
        "potatoes": 8,
        "cauliflower": 7,
        "carrots": 7,
        "peas": 6,
        "onion": 6,
        "garlic": 6,
        "ginger": 5,
        "coconut milk": 4,
        "tomatoes": 7,
        "curry powder": 7,
        "cumin": 6,
        "coriander": 6,
        "turmeric": 6,
        "chilli powder": 4,
        "salt": 4
    },

"Vegetable Noodles": {
        "noodles": 10,
        "carrots": 8,
        "bell peppers": 7,
        "cabbage": 7,
        "onion": 7,
        "garlic": 6,
        "ginger": 6,
        "soy sauce": 8,
        "vegetable oil": 6,
        "green onions": 6,
        "bean sprouts": 5,
        "salt": 4,
        "black pepper": 4,
        "chilli flakes": 4
    },

"Chicken Noodles": {
        "noodles": 10,
        "carrots": 8,
        "bell peppers": 7,
        "cabbage": 7,
        "onion": 7,
        "garlic": 6,
        "ginger": 6,
        "soy sauce": 8,
        "chicken": 10,
        "vegetable oil": 6,
        "green onions": 6,
        "salt": 4,
        "black pepper": 4,
        "chilli flakes": 4
    },

    "Chicken Biryani": {
        "chicken": 10,
        "basmati rice": 10,
        "onion": 8,
        "tomatoes": 7,
        "yogurt": 9,
        "green chilli": 6,
        "mint leaves": 6,
        "coriander leaves": 6,
        "cinnamon": 6,
        "cardamom": 6,
        "cloves": 6,
        "bay leaves": 5,
        "turmeric powder": 5,
        "cumin powder": 5,
        "coriander powder": 5,
        "red chilli powder": 5,
        "ghee": 8,
        "vegetable oil": 6,
        "saffron": 7,
        "salt": 5,
        "black pepper": 4
    },

"Vegetable Biryani": {
        "basmati rice": 10,
        "carrots": 8,
        "potatoes": 8,
        "cauliflower": 7,
        "beans": 7,
        "bell peppers": 7,
        "onion": 7,
        "tomatoes": 6,
        "yogurt": 6,
        "green chilli": 5,
        "mint leaves": 5,
        "coriander leaves": 5,
        "cinnamon": 5,
        "cardamom": 5,
        "cloves": 5,
        "bay leaves": 4,
        "turmeric powder": 4,
        "cumin powder": 4,
        "coriander powder": 4,
        "red chilli powder": 4,
        "ghee": 7,
        "vegetable oil": 6,
        "salt": 5,
        "black pepper": 3
    },

"Pav Bhaji": {
        "potatoes": 9,
        "tomatoes": 8,
        "onion": 8,
        "peas": 7,
        "bell peppers": 7,
        "cauliflower": 6,
        "carrots": 6,
        "green beans": 6,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "butter": 8,
        "pav bhaji masala": 9,
        "cumin powder": 6,
        "coriander powder": 6,
        "turmeric powder": 5,
        "red chilli powder": 5,
        "lemon juice": 7,
        "coriander leaves": 7,
        "salt": 5,
        "black pepper": 4
    },

    "Chicken Pasta": {
        "chicken": 10,
        "pasta": 10,
        "bell peppers": 7,
        "onion": 7,
        "garlic": 7,
        "mushrooms": 6,
        "tomatoes": 6,
        "olive oil": 8,
        "butter": 6,
        "cream": 6,
        "cheese": 7,
        "basil": 6,
        "oregano": 5,
        "salt": 5,
        "black pepper": 5
    },

"Vegetable Pasta": {
        "pasta": 10,
        "bell peppers": 9,
        "mushrooms": 8,
        "tomatoes": 9,
        "onion": 7,
        "garlic": 6,
        "olive oil": 7,
        "butter": 5,
        "cream": 5,
        "cheese": 6,
        "basil": 6,
        "oregano": 5,
        "salt": 5,
        "black pepper": 5
    },

"Chicken Tacos": {
        "chicken": 10,
        "taco shells": 10,
        "lettuce": 8,
        "tomatoes": 9,
        "onion": 8,
        "cream": 6,
        "lime": 6,
        "garlic": 5,
        "cumin": 5,
        "chilli powder": 5,
        "salt": 5,
        "black pepper": 5
    },

"Vegetable Tacos": {
        "taco shells": 10,
        "bell peppers": 9,
        "mushrooms": 8,
        "onion": 9,
        "tomatoes": 9,
        "lettuce": 8,
        "cheese": 6,
        "cream": 5,
        "lime": 5,
        "garlic": 5,
        "cumin": 4,
        "chilli powder": 4,
        "paprika": 4,
        "salt": 4,
        "black pepper": 4
    },


"Chocolate Cake": {
        "flour": 10,
        "sugar": 10,
        "cocoa powder": 10,
        "baking powder": 7,
        "baking soda": 7,
        "eggs": 10,
        "milk": 10,
        "vegetable oil": 8,
        "vanilla extract": 8,
        "boiling water": 7,
        "salt": 6,
        "butter": 9,
        "cream": 9,
        "dark chocolate": 10
    },

"Matar Paneer": {
        "paneer": 10,
        "peas": 10,
        "onion": 8,
        "tomatoes": 9,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cashew nuts": 5,
        "cumin seeds": 5,
        "coriander powder": 6,
        "turmeric powder": 5,
        "garam masala": 9,
        "cream": 10,
        "vegetable oil": 6,
        "butter": 10,
        "salt": 5,
        "black pepper": 4
    },

    "Chicken Manchurian": {
        "chicken": 10,
        "ginger-garlic paste": 8,
        "green onions": 9,
        "bell peppers": 9,
        "onion": 8,
        "soy sauce": 7,
        "cornstarch": 6,
        "flour": 6,
        "eggs": 9,
        "tomatoes": 8,
        "vinegar": 6,
        "red chilli sauce": 6,
        "green chilli sauce": 6,
        "sugar": 5,
        "vegetable oil": 7,
        "salt": 5,
        "black pepper": 5,
        "cabbage": 9,
    },

    "Vegetable Manchurian": {
        "cabbage": 10,
        "carrots": 8,
        "bell peppers": 9,
        "onion": 9,
        "garlic": 6,
        "ginger": 6,
        "green chilli": 5,
        "spring onions": 6,
        "soy sauce": 6,
        "cornstarch": 6,
        "flour": 9,
        "green chilli sauce": 6,
        "tomatoes": 6,
        "vinegar": 5,
        "sugar": 5,
        "vegetable oil": 6,
        "salt": 5,
        "black pepper": 4
    },

"Aloo Paratha": {
        "wheat flour": 10,
        "potatoes":10,
        "onion": 9,
        "green chilli": 6,
        "ginger": 6,
        "coriander leaves": 6,
        "cumin seeds": 5,
        "garam masala": 5,
        "turmeric powder": 5,
        "red chilli powder": 5,
        "vegetable oil": 6,
        "ghee": 6,
        "salt": 5,
        "black pepper": 4
    },

"Corn Flour Onion Rings": {
        "onion": 10,
        "corn flour": 10,
        "flour": 10,
        "salt": 6,
        "black pepper": 6,
        "paprika": 5,
        "garlic powder": 5,
        "vegetable oil": 8
    },

"Buttered Corn": {
        "corn": 10,
        "butter": 10,
        "salt": 9,
        "black pepper": 6,
        "parsley": 5,
        "garlic powder": 5,
        "lemon juice": 5
    },

"Apple Pie": {
        "apple": 10,
        "flour": 8,
        "sugar": 8,
        "butter": 8,
        "lemon juice": 6,
        "cinnamon": 6,
        "nutmeg": 5,
        "salt": 5,
        "water": 5,
        "pie crust": 8,
        "eggs": 6
    },


"Chicken Sandwich": {
        "chicken": 10,
        "bread": 10,
        "lettuce": 9,
        "tomatoes": 9,
        "mayonnaise": 8,
        "mustard": 6,
        "cheese": 9,
        "onion": 5,
        "salt": 5,
        "black pepper": 5,
        "vegetable oil": 5
    },

"Vegetable Sandwich": {
        "bread": 10,
        "tomatoes": 9,
        "cucumber": 9,
        "lettuce": 7,
        "bell peppers": 7,
        "onion": 10,
        "cheese": 9,
        "mayonnaise": 8,
        "mustard": 5,
        "salt": 5,
        "black pepper": 5,
        "butter": 7
    },

"Vegetable Chilli": {
        "kidney beans": 9,
        "black beans": 8,
        "bell peppers": 8,
        "onion": 7,
        "garlic": 7,
        "carrots": 6,
        "corn": 6,
        "tomatoes": 6,
        "chilli powder": 7,
        "cumin": 6,
        "paprika": 5,
        "oregano": 5,
        "salt": 5,
        "black pepper": 5,
        "olive oil": 5
    },

"Chicken Chilli": {
        "chicken": 10,
        "bell peppers": 9,
        "onion": 8,
        "garlic": 7,
        "ginger": 6,
        "green chilli": 7,
        "soy sauce": 6,
        "tomatoes": 6,
        "cornstarch": 5,
        "vegetable oil": 5,
        "salt": 5,
        "black pepper": 5,
        "spring onions": 6
    },

"Paneer Chilli": {
        "paneer": 10,
        "bell peppers": 8,
        "onion": 7,
        "garlic": 7,
        "ginger": 6,
        "green chilli": 7,
        "soy sauce": 6,
        "tomatoes": 6,
        "cornstarch": 5,
        "vegetable oil": 5,
        "salt": 5,
        "black pepper": 5,
        "spring onions": 6
    },

"French Fries": {
        "potatoes": 10,
        "vegetable oil": 8,
        "salt": 7,
        "black pepper": 6,
        "paprika": 5,
        "garlic powder": 5,
        "onion powder": 5
    },

"Dal": {
        "moong dal": 10,
        "onion": 8,
        "tomatoes": 7,
        "garlic": 7,
        "ginger": 6,
        "green chilli": 6,
        "cumin seeds": 5,
        "turmeric powder": 5,
        "coriander powder": 5,
        "red chilli powder": 5,
        "garam masala": 5,
        "vegetable oil": 5,
        "salt": 5,
        "coriander leaves": 6
    },

"Fruit Tart": {
        "pie crust": 9,
        "cream": 8,
        "strawberry": 8,
        "mango": 8,
        "sugar": 9,
    },

"Chickpea Curry": {
        "chickpeas": 9,
        "onion": 8,
        "tomatoes": 8,
        "garlic": 7,
        "ginger": 6,
        "green chilli": 6,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 9,
        "coconut milk": 6,
        "vegetable oil": 5,
        "salt": 5,
        "coriander leaves": 6
    },

"Vegetable Fried Rice": {
        "basmati rice": 9,
        "carrots": 8,
        "bell peppers": 7,
        "onion": 7,
        "garlic": 6,
        "ginger": 6,
        "green onions": 6,
        "soy sauce": 6,
        "vegetable oil": 5,
        "salt": 5,
        "black pepper": 5,
    },

"Chicken Fried Rice": {
        "basmati rice": 10,
        "chicken": 10,
        "carrots": 7,
        "peas": 7,
        "bell peppers": 7,
        "onion": 7,
        "garlic": 6,
        "ginger": 6,
        "green onions": 6,
        "soy sauce": 6,
        "sesame oil": 5,
        "vegetable oil": 5,
        "salt": 5,
        "black pepper": 5,
        "eggs": 8
    },

"Chicken Tikka Pizza": {
        "flour": 10,
        "chicken": 10,
        "onion": 7,
        "bell peppers": 7,
        "cheese": 9,
        "tomatoes": 10,
        "olive oil": 6,
        "garlic": 5,
        "salt": 5,
        "black pepper": 5,
        "red chilli flakes": 4
    },

"Paneer Tikka Pizza": {
        "flour": 10,
        "paneer": 10,
        "onion": 7,
        "bell peppers": 7,
        "cheese": 7,
        "tomatoes": 6,
        "olive oil": 6,
        "garlic": 5,
        "salt": 5,
        "black pepper": 5,
        "red chilli flakes": 4
    },

"Garlic Bread": {
        "bread": 9,
        "butter": 9,
        "garlic": 10,
        "parsley": 9,

        "salt": 5,
    },

    "Nimbu Pani": {
        "water": 9,
        "lemon": 10,
        "sugar": 7,
        "salt": 6,
        "black salt": 6,
        "mint leaves": 5,
    },

"Chickoo Milkshake": {
        "chickoo": 9,
        "milk": 8,
        "sugar": 7,
    },

"mango Milkshake": {
        "mango": 9,
        "milk": 8,
        "sugar": 7,
    },

"Pancakes": {
        "flour": 9,
        "milk": 8,
        "eggs": 7,
        "baking powder": 6,
        "sugar": 5,
        "salt": 5,
        "butter": 6,
        "vanilla extract": 5
    },

"Mint Chutney": {
        "mint leaves": 10,
        "coriander leaves": 10,
        "green chilli": 7,
        "ginger": 6,
        "garlic": 6,
        "lemon juice": 6,
        "salt": 5,
        "sugar": 5,
        "cumin powder": 5,
        "water": 5
    },

"Butter Cashew Biscuits": {
        "flour": 9,
        "butter": 10,
        "powdered sugar": 7,
        "cashew nuts": 10,
        "vanilla extract": 5,
        "baking powder": 8,
        "salt": 4
    },

"Rajma": {
        "rajma": 10,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "vegetable oil": 5,
        "salt": 5,
        "coriander leaves": 6
    },

"Bread Pakora": {
        "bread": 10,
        "potatoes": 8,
        "flour": 8,
        "rice flour": 6,
        "ginger": 5,
        "green chilli": 5,
        "coriander leaves": 5,
        "salt": 5,
        "turmeric powder": 4,
        "cumin seeds": 4,
        "garam masala": 4,
        "vegetable": 6
    },

"Onion Pakora": {
        "onion": 10,
        "flour": 8,
        "rice flour": 7,
        "green chilli": 6,
        "ginger": 6,
        "coriander leaves": 6,
        "salt": 5,
        "turmeric powder": 4,
        "cumin seeds": 4,
        "vegetable oil": 6
    },

"Tomato Soup": {
        "tomatoes": 10,
        "onion": 8,
        "garlic": 7,
        "water": 7,
        "butter": 6,
        "olive oil": 6,
        "sugar": 5,
        "salt": 5,
        "black pepper": 5,
    },

"Chicken Soup": {
        "chicken": 10,
        "carrots": 7,
        "onion": 7,
        "garlic": 6,
        "salt": 5,
        "black pepper": 5,
        "parsley": 5,
        "butter": 4,
        "olive oil": 4
    },

"Paneer Tikka": {
        "paneer":10,
        "yogurt": 8,
        "ginger-garlic paste": 7,
        "lemon juice": 6,
        "red chilli powder": 6,
        "turmeric powder": 5,
        "garam masala": 8,
        "coriander powder": 5,
        "cumin powder": 5,
        "salt": 5,
        "bell peppers": 6,
        "onion": 7,
    },

"Samosa": {
        "flour": 9,
        "potatoes": 10,
        "onion": 7,
        "ginger": 6,
        "green chilli": 6,
        "turmeric powder": 5,
        "garam masala": 5,
        "salt": 5,
        "vegetable oil": 6
    },

"Masala Chai": {
        "water": 9,
        "milk": 9,
        "tea leaves": 10,
        "ginger": 8,
        "cinnamon": 5,
        "cardamom": 5,
        "cloves": 8,
        "sugar": 10
    },

"Palak Paneer": {
        "spinach": 9,
        "paneer": 10,
        "onion": 7,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "cream": 4,
        "salt": 5,
        "vegetable oil": 5
    },
"Baingan Bharta": {
        "eggplant": 9,
        "tomatoes": 8,
        "onion": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "fresh cilantro (coriander leaves)": 4,
        "vegetable oil": 5,
        "salt": 5
    },

"Chana Masala": {
        "chickpeas": 9,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "dry mango powder": 4,
        "coriander leaves": 4,
        "vegetable oil": 5,
        "salt": 5
    },

"Dal Makhani": {
        "urad dal": 9,
        "rajma": 8,
        "onion": 7,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "cream": 4,
        "butter": 4,
        "coriander leaves": 4,
        "vegetable oil": 5,
        "salt": 5
    },

"Butter Chicken": {
        "chicken": 10,
        "tomatoes": 9,
        "butter": 10,
        "cream": 10,
        "onion": 6,
        "garlic": 6,
        "ginger": 6,
        "green chilli": 5,
        "red chilli powder": 5,
        "garam masala": 5,
        "fenugreek leaves": 5,
        "lemon juice": 4,
        "vegetable oil": 5,
        "salt": 5
    },

"Bhindi Masala": {
        "lady finger": 10,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 7,
        "dry mango powder": 4,
        "coriander leaves": 4,
        "vegetable oil": 5,
        "salt": 5
    },

"Tandoori Chicken": {
        "chicken": 10,
        "yogurt": 8,
        "ginger-garlic paste": 7,
        "lemon juice": 7,
        "paprika": 6,
        "cumin powder": 6,
        "coriander powder": 6,
        "garam masala": 9,
        "turmeric powder": 5,
        "red chilli powder": 5,
        "salt": 5,
        "vegetable oil": 5
    },

"Shahi Paneer": {
        "paneer":10,
        "onion": 8,
        "tomatoes": 7,
        "cashew nuts": 6,
        "cream": 6,
        "butter": 8,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "garam masala": 5,
        "red chilli powder": 5,
        "salt": 5,
        "vegetable oil": 5
    },


    "Vegetable Pulao": {
        "rice": 10,
        "carrots": 9,
        "onion": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "cinnamon": 5,
        "peas": 9,
        "cloves": 5,
        "bay leaf": 5,
        "turmeric powder": 4,
        "French beans": 9,
        "ghee": 6,
        "salt": 5,
        "coriander leaves": 4
    },

"Malai Kofta": {
        "chicken": 10,
        "cashew nuts": 7,
        "cream": 7,
        "onion": 6,
        "tomatoes": 6,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "garam masala": 5,
        "red chilli powder": 5,
        "saffron": 4,
        "salt": 5,
        "vegetable oil": 5
    },

"Masala Dosa": {
        "dosa batter": 9,
        "potatoes": 8,
        "onion": 7,
        "green chilli": 6,
        "ginger": 6,
        "mustard seeds": 5,
        "cumin seeds": 5,
        "turmeric powder": 5,
        "curry leaves": 5,
        "vegetable oil": 5,
        "salt": 5
    },

"Aloo Tikki": {
        "potatoes": 9,
        "breadcrumbs": 8,
        "onion": 6,
        "ginger": 5,
        "green chilli": 5,
        "coriander leaves": 5,
        "cumin powder": 5,
        "coriander powder": 5,
        "garam masala": 5,
        "turmeric powder": 4,
        "salt": 5,
        "vegetable oil": 6
    },

    "Chicken Patty": {
        "chicken": 10,
        "breadcrumbs": 8,
        "onion": 7,
        "garlic": 6,
        "green chilli": 5,
        "ginger": 5,
        "coriander leaves": 5,
        "cumin powder": 5,
        "coriander powder": 5,
        "paprika": 5,
        "salt": 5,
        "black pepper": 5,
        "eggs": 6,
        "vegetable oil": 6
    },

"Vegetable Cutlet": {
        "potatoes": 8,
        "carrots": 7,
        "onion": 6,
        "ginger": 5,
        "green chilli": 5,
        "coriander leaves": 5,
        "cumin powder": 5,
        "coriander powder": 5,
        "garam masala": 5,
        "turmeric powder": 4,
        "breadcrumbs": 7,
        "salt": 5,
        "black pepper": 5,
        "vegetable oil": 6
    },

"Vada Pav": {
        "potatoes": 8,
        "flour": 7,
        "green chilli": 6,
        "ginger": 5,
        "garlic": 5,
        "curry leaves": 5,
        "turmeric powder": 5,
        "coriander leaves": 5,
        "cumin powder": 4,
        "salt": 5,
        "vegetable oil": 6,
        "bread": 8,
    },

"Idli": {
        "rice flour": 9,
        "urad dal": 8,
        "fenugreek seeds": 5,
        "salt": 5
    },

"Paneer Butter Masala": {
        "paneer": 10,
        "onion": 8,
        "tomatoes": 7,
        "cashew nuts": 9,
        "cream": 9,
        "butter": 10,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
        "cumin powder": 5,
        "coriander powder": 5,
        "garam masala": 8,
        "red chilli powder": 5,
        "sugar": 4,
        "salt": 5,
        "vegetable oil": 5
    },

"Chole Bhature": {
        "chickpeas": 10,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "dry mango powder": 4,
        "coriander leaves": 4,
        "baking soda": 4,
        "flour": 9,
        "yogurt": 7,
        "vegetable oil": 6,
        "sugar": 5,
        "salt": 5
    },

"Rasmalai": {
        "milk": 9,
        "sugar": 8,
        "lemon juice": 7,
        "saffron": 6,
        "cardamom pods": 6,
        "pistachios": 6,
        "almonds": 6,
        "rose water": 4
    },

    "Jalebi": {
        "flour": 9,
        "sugar": 7,
        "water": 7,
        "saffron": 6,
        "cardamom pods": 6,
        "baking powder": 5,
        "vegetable oil": 6
    },

    "Rasgulla": {
        "milk": 9,
        "lemon juice": 8,
        "sugar": 7,
        "water": 7,
        "cardamom": 6,
        "rose water": 6,
        "saffron": 5
    },

"Tandoori Roti": {
        "wheat flour": 9,
        "yogurt": 8,
        "water": 7,
        "salt": 6,
        "baking powder": 5,
        "vegetable oil": 5,
        "ghee": 4
    },

"Chicken 65": {
        "chicken": 10,
        "curry leaves": 8,
        "ginger-garlic paste": 7,
        "corn flour": 6,
        "rice flour": 6,
        "chilli powder": 6,
        "turmeric powder": 5,
        "garam masala": 5,
        "yogurt": 5,
        "salt": 5,
        "vegetable oil": 7
    },

"Vegetable Frankie": {
        "wheat flour": 9,
        "cabbage": 8,
        "potatoes": 7,
        "onion": 6,
        "ginger-garlic paste": 6,
        "green chilli": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "coriander powder": 5,
        "chaat masala": 5,
        "salt": 5,
        "vegetable oil": 6,
    },

    "Chicken Frankie": {
        "chicken": 10,
        "cabbage":8,
        "onion": 7,
        "ginger-garlic paste": 6,
        "green chilli": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "coriander powder": 5,
        "chaat masala": 5,
        "lemon juice": 5,
        "salt": 5,
        "vegetable oil": 6,
    },

    "Vegetable Roll": {
        "wheat flour": 10,
        "cabbage": 8,
        "potatoes": 7,
        "onion": 6,
        "ginger-garlic paste": 6,
        "green chilli": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "coriander powder": 5,
        "chaat masala": 5,
        "salt": 5,
    },

    "Chicken Roll": {
        "chicken": 10,
        "cabbage": 8,
        "onion": 7,
        "ginger-garlic paste": 6,
        "green chilli": 5,
        "turmeric powder": 5,
        "garam masala": 5,
        "coriander powder": 5,
        "chaat masala": 5,
        "salt": 5,
    },

"Masala Aloo": {
        "potatoes": 9,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cumin seeds": 5,
        "coriander powder": 5,
        "turmeric powder": 5,
        "garam masala": 9,
        "dry mango powder": 4,
        "coriander leaves": 4,
        "vegetable oil": 5,
        "salt": 5
    },

"Chicken Burger": {
        "chicken": 9,
        "breadcrumbs": 8,
        "green chilli": 5,
        "cumin powder": 5,
        "salt": 5,
        "black pepper": 5,
        "eggs": 6,
        "burger buns": 8,
        "lettuce": 9,
        "tomatoes": 9,
        "cheese": 9,
        "mayonnaise": 9,
        "ketchup": 6,
        "vegetable oil": 6
    },

"Vegetable Burger": {
        "potatoes": 10,
        "onion": 6,
        "garlic": 5,
        "ginger": 5,
        "green chilli": 5,
        "coriander leaves": 5,
        "cumin powder": 5,
        "coriander powder": 5,
        "breadcrumbs": 6,
        "salt": 5,
        "black pepper": 5,
        "burger buns": 8,
        "lettuce": 9,
        "tomatoes": 9,
        "cheese": 9,
        "mayonnaise": 9,
        "ketchup": 6,
        "vegetable oil": 6
    },

"Mushroom Stir-Fry": {
        "mushrooms": 9,
        "onion": 7,
        "bell peppers": 7,
        "broccoli": 6,
        "carrots": 6,
        "garlic": 5,
        "ginger": 5,
        "soy sauce": 7,
        "vegetable oil": 5,
        "salt": 5,
        "black pepper": 5,
    },

"Vegetable Spaghetti": {
        "spaghetti": 9,
        "bell peppers": 7,
        "carrots": 7,
        "onion": 6,
        "garlic": 5,
        "tomatoes": 5,
        "olive oil": 6,
        "salt": 5,
        "black pepper": 5,
        "oregano": 5,
    },

}

  # Your recipes dictionary here
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(recipes.values())
y = np.arange(len(recipes))
def get_top_recipes(ingredients, top_n=6):
    # Transform input ingredients into binary format
    input_vec = mlb.transform([ingredients])

    # Predict the probabilities of each recipe
    probs = loaded_model.predict(input_vec)[0]

    # Calculate weighted probabilities
    weighted_probs = [probs[i] * sum(recipes[recipe_name].get(ingredient, 0) for ingredient in ingredients) for i, recipe_name in enumerate(recipes.keys())]

    # Get the indices of the top N recipes based on weighted probabilities
    top_indices = np.argsort(weighted_probs)[::-1][:top_n]

    # Get the top N recipe names and their matching scores
    top_recipes = [(list(recipes.keys())[i], weighted_probs[i]) for i in top_indices]
    for i, (recipe, score) in enumerate(top_recipes, 1):
        print(f"{i}. {recipe}: {score}")

    return top_recipes

@app.route('/recommend_recipe', methods=['GET'])
def recommend_recipe():
    ingredients = ['cheese','tomato','bread','onion']
    
    # Validate input
    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400
    
    # Get recommended recipe
    recommended_recipe = get_top_recipes(ingredients)
  
    return jsonify({"recipe": recommended_recipe})

@app.route('/home')
def home():
    return jsonify({"msg": "hello world !"})
if __name__ == '__main__':
   
    app.run(debug=True)
