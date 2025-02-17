# SmartMealPlanner
- Joseph Whiteman, David Sanchez

### Datasets
- **Kaggle RecipeNLG Database**: https://www.kaggle.com/datasets/saldenisov/recipenlg
- **Branded Food Database**: https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_csv_2024-10-31.zip

# Milestone 2

### Explain what your AI agent does in terms of PEAS.  What is the "world" like?
**Performance**
Our agent aims to generate meal recommendations which balance nutritional quality (specifically calories, proteins, fats, and carbohydrates) with recipe simplicity (lower number of ingredients).  The concept of fewer ingredients indirectly and approximately supports our goal of optimizing nutritional meal plans while balancing financial constraints.

**Environment**
The "world" is the recipe domain which contains datasets such as RecipeNLG and USDA FoodData Central, providing ingredients and nutritional facts, respectively.  This world has proven to be extremely ambiguous and uncertain: ingredients, names, and especially that of nutritional facts and costs have varied widely across recipes, and therefore, causing noise in our data.

The environment is yet to include necessary external and additional inputs, such as our goal of incorporating user dietary preferences.  

**Actuators**
This agent and its outputs are its recommended meal plans for the user, acting by suggesting recipes for the user to cook to optimize their nutritional goals.

**Sensors**
The agent 