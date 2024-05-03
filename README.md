### For training deta using keras model follow this code :-
```
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

recipes = {

 "Spaghetti Carbonara": {
"spaghetti": 10,
 "eggs": 8,
 "cheese": 9
 },

 "Chicken Tikka Masala": {
 "chicken": 10,
"yogurt": 6,
 "tomatoes": 7,
"onion": 7,
"garlic": 7,
 "ginger": 6,
"cream": 5,
"butter": 5,
},

"Vegetable Stir Fry": {
        "broccoli": 7,
        "carrots": 7,
        "peas": 6,
        "onion": 6,
        "garlic": 6,
        "ginger": 5,
        "soy sauce": 8,
    },

    "Caprese Salad": {
        "tomatoes": 9,
        "cheese": 9,
        "basil leaves": 8
    },

    "Vegetable Curry": {
        "potatoes": 8,
        "cauliflower": 7,
        "carrots": 7,
        "peas": 6,
        "onion": 6,
        "garlic": 6,
        "ginger": 5,
        "tomatoes": 10
    },
  	 "Cauli Flower Sabji":{
        "cauliflower":10,
        "peas":9,
        "potatoes":7,
        "green chilli":7       
        
     },

"Vegetable Noodles": {
        "noodles": 10,
        "carrots": 8,
        "cabbage": 7,
        "garlic": 6,
        "ginger": 6,
        "soy sauce": 8,
        "bean sprouts": 5
    },

"Chicken Noodles": {
        "noodles": 10,
        "carrots": 8,
        "cabbage": 7,
        "onion": 7,
        "garlic": 6,
        "ginger": 6,
        "soy sauce": 8,
        "chicken": 10,
        "green onions": 6,
        "chilli flakes": 4
    },

    "Chicken Biryani": {
        "chicken": 10,
        "basmati rice": 10,
        "onion": 8,
        "tomatoes": 7,
        "yogurt": 9,
        "green chilli": 6,
        "saffron": 7,
    },

"Vegetable Biryani": {
        "basmati rice": 10,
        "carrots": 8,
        "potatoes": 8,
        "cauliflower": 7,
        "beans": 7,
        "onion": 7,
        "tomatoes": 6,
        "yogurt": 6,
        "green chilli": 5
    },

"Pav Bhaji": {
        "potatoes": 10,
        "tomatoes": 8,
        "onion": 8,
        "peas": 7,
        "cauliflower": 6,
        "carrots": 6,
        "green beans": 6,
        "ginger": 6,
        "garlic": 6,
        "butter": 8,
        "pav bhaji masala": 8
    },
    "Chicken Pasta": {
        "chicken": 10,
        "pasta": 10,
        "tomatoes": 6,
        "onion": 6,
        "garlic": 6,
        "cream": 7,
        "cheese": 6,
        "basil": 6,
        "oregano": 6
    },

"Vegetable Pasta": {
        "pasta": 10,
        "bell peppers": 9,
        "mushrooms": 8,
        "tomatoes": 9,
        "onion": 7,
        "garlic": 6,
        "cream": 5,
        "cheese": 6,
        "basil": 6,
        "oregano": 5
    },

"Chicken Tacos": {
        "chicken": 10,
        "taco shells": 10,
        "lettuce": 8,
        "tomatoes": 9,
        "onion": 8,
        "cream": 6,
        "garlic": 5
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
        "garlic": 5
    },


"Chocolate Cake": {
        "flour": 10,
        "cocoa powder": 10,
        "baking powder": 7,
        "baking soda": 7,
        "eggs": 10,
        "butter": 9,
        "cream": 9,
        "dark chocolate": 10
    },

"Matar Paneer": {
        "paneer": 10,
        "peas": 10,
        "onion": 6,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5
    },

    "Chicken Manchurian": {
        "chicken": 10,
        "garlic":5,
        "green onions": 9,
        "onion": 8,
        "soy sauce": 7,
        "cornstarch": 6,
        "flour": 6,
        "eggs": 9,
        "tomatoes": 8,
        "vinegar": 6,
        "cabbage": 9,
    },

    "Vegetable Manchurian": {
        "cabbage": 10,
        "carrots": 8,
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
        "vinegar": 5
    },

"Aloo Paratha": {
        "wheat flour": 10,
        "potatoes":10,
        "onion": 5,
        "green chilli": 6
    },

"Corn Flour Onion Rings": {
        "onion": 10,
        "corn flour": 10,
        "flour": 10,
        "paprika": 5
    },

"Buttered Corn": {
        "corn": 10,
        "butter": 10,
        "salt": 9,
        "parsley": 5
    },

"Apple Pie": {
        "apple": 10,
        "flour": 8,
        "sugar": 8,
        "butter": 8,
        "lemon juice": 6,
        "cinnamon": 6,
        "nutmeg": 5,
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
        "cheese": 7,
        "onion": 5
    },

"Vegetable Sandwich": {
        "bread": 10,
        "cucumber": 9,
        "lettuce": 7,
        "onion": 10,
        "cheese": 9,
        "mayonnaise": 8,
        "mustard": 5,
        "tomatoes": 9,
        "butter": 7
    },

"Vegetable Chilli": {
        "kidney beans": 9,
        "black beans": 8,
        "onion": 7,
        "garlic": 7,
        "carrots": 6,
        "corn": 6,
        "tomatoes": 6,
        "chilli powder": 7,
        "paprika": 5,
        "oregano": 5
    },

"Chicken Chilli": {
        "chicken": 10,
        "bell peppers": 9,
        "onion": 8,
        "garlic": 7,
        "ginger": 6,
        "green chilli": 7,
        "soy sauce": 6,
        "tomatoes": 5
    },

"Paneer Chilli": {
        "paneer": 10,
        "onion": 7,
        "garlic": 7,
        "ginger": 6,
        "green chilli": 7,
        "tomatoes": 6
    },

"French Fries": {
        "potatoes": 10,
        "salt": 7,
        "paprika": 6
    },

"Dal": {
        "moong dal": 10,
        "onion": 8,
        "tomatoes": 7,
        "garlic": 7,
        "ginger": 6,
        "green chilli": 6
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
        "coconut milk": 6
    },

"Vegetable Fried Rice": {
        "basmati rice": 10,
        "carrots": 8,
        "onion": 7,
        "garlic": 6,
        "ginger": 6,
        "soy sauce": 6
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
        "soy sauce": 7,
        "eggs": 8
    },

"Chicken Tikka Pizza": {
        "flour": 10,
        "chicken": 10,
        "onion": 6,
        "cheese": 9,
        "tomatoes": 7,
        "garlic": 5,
        "red chilli flakes": 5
    },

"Paneer Tikka Pizza": {
        "flour": 10,
        "paneer": 10,
        "onion": 7,
        "bell peppers": 7,
        "cheese": 7,
        "tomatoes": 6,
        "black pepper": 5,
        "red chilli flakes": 4
    },

"Garlic Bread": {
        "bread": 10,
        "butter": 9,
        "garlic": 10,
        "parsley": 9,
        "salt": 5,
    },

    "Nimbu Pani": {
        "lemon": 10,
        "sugar": 7
    },

"Chickoo Milkshake": {
        "chickoo": 10,
        "milk": 9,
        "sugar": 7,
    },

"mango Milkshake": {
        "mango": 10,
        "milk": 9,
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
        "vanilla extract": 6
    },

"Mint Chutney": {
        "mint leaves": 10,
        "coriander leaves": 10,
        "green chilli": 7,
        "ginger": 6,
        "garlic": 6,
        "lemon juice": 6
    },

"Butter Cashew Biscuits": {
        "flour": 9,
        "butter": 10,
        "powdered sugar": 7,
        "cashew nuts": 10,
        "vanilla extract": 5,
        "baking powder": 8
    },

"Rajma": {
        "rajma": 10,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5
    },

"Bread Pakora": {
        "bread": 10,
        "potatoes": 8,
        "flour": 8,
        "rice flour": 6,
        "ginger": 5,
        "green chilli": 5,
        "vegetable": 6
    },

"Onion Pakora": {
        "onion": 10,
        "flour": 7,
        "ginger": 6
    },

"Tomato Soup": {
        "tomatoes": 10,
        "onion": 6,
        "garlic": 6,
        "butter": 5,
        "sugar": 5
    },

"Chicken Soup": {
        "chicken": 10,
        "carrots": 7,
        "onion": 7,
        "garlic": 6,
        "salt": 5,
        "parsley": 5,
        "butter": 4
    },

"Paneer Tikka": {
        "paneer":10,
        "yogurt": 8,
        "onion": 7
    },

"Samosa": {
        "flour": 8,
        "potatoes": 10,
        "onion": 7,
        "ginger": 6,
        "green chilli": 6
    },

"Masala Chai": {
        "milk": 9,
        "tea leaves": 10,
        "ginger": 8,
        "cinnamon": 5,
        "cardamom": 5,
        "cloves": 8,
        "sugar": 10
    },

"Palak Paneer": {
        "spinach": 10,
        "paneer": 10,
        "onion": 7,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cream": 4
    },
"Baingan Bharta": {
        "eggplant": 9,
        "tomatoes": 8,
        "onion": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5
    },

"Chana Masala": {
        "chickpeas": 10,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5
    },

"Dal Makhani": {
        "urad dal": 9,
        "rajma": 8,
        "onion": 7,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "cream": 4,
        "butter": 4
    },

"Butter Chicken": {
        "chicken": 10,
        "tomatoes": 9,
        "butter": 10,
        "cream": 10,
        "green chilli": 5,
        "red chilli powder": 5,
        "garam masala": 5
    },

"Bhindi Masala": {
        "lady finger": 10,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "garam masala": 7
    },

"Tandoori Chicken": {
        "chicken": 10,
        "yogurt": 8,
        "ginger-garlic paste": 7,
        "lemon juice": 7,
        "paprika": 6,
        "garam masala": 9,
        "turmeric powder": 5,
        "red chilli powder": 5
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
        "green chilli": 5
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
        "potatoes": 10,
        "cashew nuts": 7,
        "cream": 10,
        "onion": 6,
        "tomatoes": 10,
        "ginger": 5,
        "garlic": 5,
        "saffron": 4,
    },

"Masala Dosa": {
        "rice flour": 10,
        "potatoes": 10,
        "onion": 7,
        "green chilli": 6,
        "ginger": 6,
        "mustard seeds": 7
    },

"Aloo Tikki": {
        "potatoes": 9,
        "bread": 8,
        "onion": 6,
        "ginger": 5
    },

    "Chicken Patty": {
        "chicken": 10,
        "bread": 7,
        "onion": 7,
        "ginger": 5,
    },

"Vegetable Cutlet": {
        "potatoes": 10,
        "carrots": 7,
        "onion": 6,
        "ginger": 5,
        "green chilli": 5,
        "bread": 10,
    },

"Vada Pav": {
        "potatoes": 10,
        "corn flour": 10,
        "green chilli": 6,
        "ginger": 5,
        "garlic": 5,
        "mustard seeds":9,
        "pav": 10,
    },

"Idli": {
        "rice flour": 10,
        "urad dal": 10,
        "fenugreek seeds": 5,
    },

"Paneer Butter Masala": {
        "paneer": 10,
        "onion": 6,
        "tomatoes": 10,
        "cashew nuts": 7,
        "cream": 10,
        "butter": 10,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
    },

"Chole Bhature": {
        "chickpeas": 10,
        "onion": 8,
        "tomatoes": 10,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
        "baking soda": 8,
        "flour": 10,
        "yogurt": 8
    },

"Rasmalai": {
        "milk": 10,
        "sugar": 10,
        "lemon": 7,
        "saffron": 6,
        "cardamom pods": 6,
        "pistachios": 10,
        "almonds": 8,
        "rose water": 4
    },

    "Jalebi": {
        "flour": 10,
        "sugar": 10,
        "saffron": 6,
        "cardamom": 6
    },

    "Rasgulla": {
        "milk": 10,
        "lemon": 8,
        "sugar": 10,
        "cardamom": 6,
        "rose water": 6,
        "saffron": 5
    },

"Tandoori Roti": {
        "wheat flour": 10,
        "yogurt": 8,
        "baking powder": 8
    },

"Chicken 65": {
        "chicken": 10,
        "ginger-garlic paste": 7,
        "corn flour": 10,
        "soy sauce": 10     
    },

"Vegetable Frankie": {
        "potatoes": 10,
        "cabbage":8,
        "onion": 7,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
        "flour": 10
    },

    "Chicken Frankie": {
        "chicken": 10,
        "cabbage":8,
        "onion": 7,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
        "flour":10
    },

    "Vegetable Roll": {
        "flour": 10,
        "cabbage": 8,
        "potatoes": 10,
        "onion": 6,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
    },

    "Chicken Roll": {
        "chicken": 10,
        "cabbage": 8,
        "onion": 7,
        "ginger": 5,
        "garlic": 5,
        "green chilli": 5,
        "flour": 10
    },

"Masala Aloo": {
        "potatoes": 10,
        "onion": 8,
        "tomatoes": 7,
        "ginger": 6,
        "garlic": 6,
        "green chilli": 5,
    },

"Chicken Burger": {
        "chicken": 10,
        "bread": 6,
        "onion": 6,
        "green chilli": 5,
        "burger buns": 10,
        "lettuce": 10,
        "tomatoes": 7,
        "cheese": 7,
        "mayonnaise": 10,
    },

"Vegetable Burger": {
        "potatoes": 10,
        "tomatoes": 10,
        "onion": 6,
        "garlic": 5,
        "ginger": 5,
        "green chilli": 5,
        "bread": 6,
        "burger buns": 10,
        "lettuce": 10,
        "cheese": 10,
        "mayonnaise": 10,
    },

"Mushroom Stir-Fry": {
        "mushrooms": 10,
        "onion": 7,
        "bell peppers": 7,
        "broccoli": 6,
        "carrots": 6,
        "garlic": 7,
        "ginger": 7,
        "soy sauce": 8,
    },

"Vegetable Spaghetti": {
        "spaghetti": 10,
        "bell peppers": 8,
        "garlic": 8,
        "tomatoes": 10,
        "olive oil": 10,
        "oregano": 8
    },

}

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(recipes.values())
y = np.arange(len(recipes))

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(mlb.classes_),)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(recipes), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20  # Adjust as needed
model.fit(X, y, epochs=epochs)

# Save the trained model
model.save('recipe_model.h5')



# Load the saved model
loaded_model = load_model('recipe_model.h5')

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

    return top_recipes

# Example usage
ingredients = ["garlic","onion","tomatoes","cheese"]
top_recipes = get_top_recipes(ingredients)

print("Top 6 recommended recipes:")
for i, (recipe, score) in enumerate(top_recipes, 1):
    print(f"{i}. {recipe}: {score}")
```
>[!IMPORTANT]
>The latest version of code is present in tanmay-new-2 in both the repos - egrocery-deploy & recipe-recommendation.
