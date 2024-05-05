import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px  # Import Plotly Express library
from streamlit_echarts import st_echarts
import folium
from streamlit_folium import folium_static
#added for animation
import json
import requests
from streamlit_lottie import st_lottie

#function for loading animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
lottie_coding = load_lottiefile("homepage.json")
lottie_about = load_lottiefile("about.json")
lottie_error = load_lottiefile("error.json")#downloaded json file as argument

# Function to display the map
millet_regions = {
    'Foxtail Millet': ['karnataka', 'meghalaya'],
    'Finger Millet': ['meghalaya', 'uttarakhand', 'madhyapradesh', 'karnataka', 'kerala', 'tamilnadu', 'gujrat', 'odisha', 'west bengal', 'maharashtra'],
    'Banyard Millet': ['uttrakhand', 'tamil nadu'],
    'Browntop Millet': ['karnataka'],
    'Little Millet': ['maharashtra', 'karnataka', 'tamil nadu', 'odisha', 'madhya pradesh', 'andrapradesh'],
    'Kodo Millet': ['tamil nadu', 'karnataka', 'odisha', 'madhya pradesh'],
    'pearl Millet': ['jammu & kashmir', 'haryana', 'rajasthan', 'gujarat', 'maharashtra', 'karnataka', 'tamil nadu', 'telangana', 'uttar pradesh', 'gujarat', 'madhyapradesh'],
    'Proso Millet': ['tamil nadu', 'karnataka', 'andra pradesh', 'uttarakhand'],
    'Sorghum Millet': ['kerala', 'telangana', 'karnataka', 'maharashtra', 'rajasthan', 'haryana', 'uttarpradesh', 'madhya pradesh', 'andrapradesh'],
    'Buckwheat Millet':['jammu & kashmir','uttrakhand','chattisgarh'],
    'Kokan Red Rice':['maharashtra'],
    'Amaranth Millet':['kerala','karnataka','tamil nadu','maharashtra'],
}

# Function to display the map
def display_map(selected_millet):
    # Initialize map centered around India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # Add markers for regions where the selected millet is grown
    if selected_millet in millet_regions:
        regions = millet_regions[selected_millet]
        for region in regions:
            # Get coordinates for the region
            region_coordinates = {
                'karnataka': [15.3173, 75.7139],
                'meghalaya': [25.4670, 91.3662],
                'uttarakhand': [30.0668, 79.0193],
                'madhyapradesh': [22.9734, 78.6569],
                'kerala': [10.8505, 76.2711],
                'tamilnadu': [11.1271, 78.6569],
                'gujrat': [22.2587, 71.1924],
                'odisha': [20.9517, 85.0985],
                'west bengal': [22.9868, 87.8550],
                'maharashtra': [19.7515, 75.7139],
                'uttrakhand': [30.0668, 79.0193],
                'andrapradesh': [15.9129, 79.7400],
                'haryana': [29.0588, 76.0856],
                'rajasthan': [27.0238, 74.2179],
                'telangana': [18.1124, 79.0193],
                'uttar pradesh': [26.8467, 80.9462],
                'jammu & kashmir': [33.7782, 76.5762],
                'chattisgarh': [21.2787, 81.8661],
            }

            if region in region_coordinates:
                folium.Marker(location=region_coordinates[region], popup=region).add_to(m)

    # Display the map
    folium_static(m)

#Tensorflow Model Prediction
def model_prediction(test_image):  
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)    #converting image to array
    input_arr = np.array([input_arr])                               #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)    

# Load CSV data
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path, delimiter=',')
    return df  #return index of max element ie, index of max prob class

#sidebar
st.sidebar.title("Dashboard")
#dropdown menu to select different pages
app_mode= st.sidebar.selectbox("Select page",["Home","About","Prediction","Recipes"])

#Home Page
if(app_mode=="Home"):
    st_lottie(
    lottie_coding,

    speed=1.5,
    reverse=False,
    loop=True,
    quality="low",
    # renderer="svg",
    height=None,
    width=None,
    key=None,
)
    st.header("MILLETS RECOGNITION SYSTEM")
    container = st.container()
    container.write("In 2023, the world celebrated Millets Year, and now we're introducing our new image recognition system which aims to raise awareness about the importance of incorporating millets into our diets. By using machine learning, it helps identify different types of millets and provides essential nutritional information about millets, empowering people to make informed dietary choices. ")
    image_path = "images\millet.jpg" 
    st.image(image_path)     

#About Page
if(app_mode=="About"):
    selected_millet = st.sidebar.selectbox("Select Millet Type", list(millet_regions.keys()))
    # st.header("About")
    st_lottie(
                            lottie_about,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=None,
                            width=None,
                            key=None,
    )
    st.write("This is an application to predict millet types.")
    st.write("Here's the distribution of millet crops in India:")
    display_map(selected_millet)

## Prediction Page
if app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    
    # Show button
    if st.button("Show Image"):
        if test_image is not None:
            if not test_image.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                st.error("Please upload an image with JPG, JPEG, PNG, or WEBP format.")
                st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=200,
                            width=200,
                            key=None,
                )
            else:
                st.image(test_image, width=400)
        else:
            st.error("Please upload an image first.")
            st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=200,
                            width=200,
                            key=None,
                )

    # Predict button
    if st.button("Predict"):
        if test_image is not None:
            # Check file extension
            if not test_image.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                st.error("Please upload an image with JPG, JPEG, PNG, or WEBP format.")
                st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=200,
                            width=200,
                            key=None,
                )
            else:
                st.write("Our Prediction")
                result_index = model_prediction(test_image)
                
                # Reading Labels
                with open("labels.txt") as f:
                    content = f.readlines()
                label = [i.strip() for i in content]
                predicted_millet = label[result_index]
                print(predicted_millet)         # printing the predicted millet
                # Load millet information from CSV file
                df = load_data("millets_information.csv")


                predicted_millet_info = df[df['Millet_Name'] == predicted_millet].squeeze()
                st.success("The above image is of {}".format(predicted_millet_info['Name']))

                            

                # Display millet information
                st.subheader("Millet Information")
                st.write("**Name:**", predicted_millet_info['Name'])
                st.write("**Introduction:**", predicted_millet_info['Introduction'])
                st.write("**Botanical Name:**", predicted_millet_info['Botanical Name'])
                st.write("**Common Names:**", predicted_millet_info['Common Names'])
                st.write("**Cultivation Areas:**)", predicted_millet_info['Cultivation Areas'])
                st.write("**Appearance:**", predicted_millet_info['Appearance'])

                # Display benefits in a list format
                st.subheader("Benefits")
                benefits = predicted_millet_info['Benefits'].split('*')
                for benefit in benefits:
                    st.write(f"- {benefit.strip()}")



                # Display nutritional values as a doughnut chart
                st.subheader("Nutritional Value:")
                st.write("Per 100g of", predicted_millet_info['Name'],":")
                nutrients = ['Energy(kcal)', 'Carbohydrates(g)', 'Protein(g)', 'Fat(g)',  'Fiber(g)'] # Add more if needed
                nutrient_values = [float(predicted_millet_info[nutrient]) for nutrient in nutrients]  # Convert to int

                nutritional_options = {
                    "tooltip": {"trigger": "item"},
                    "legend": {"top": "5%", "left": "center"},
                    "series": [
                        {
                            "name": "Nutritional Values",
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "color": ["#023047", "#219EBC", "#8ECAE6","#FFB703","#FB8500"],

                            "avoidLabelOverlap": False,
                            "itemStyle": {
                                "borderRadius": 10,
                                "borderColor": "#fff",
                                "borderWidth": 2,
                            },
                            "label": {"show": False, "position": "center"},
                            "emphasis": {
                                "label": {"show": True, "fontSize": "16", "fontWeight": "bold"}
                            },
                            "labelLine": {"show": False},
                            "data": [
                                {"value": value, "name": nutrient}
                                for nutrient, value in zip(nutrients, nutrient_values)
                            ],
                        }
                    ],
                }
                st_echarts(options=nutritional_options, height="500px")

                st.subheader("Mineral Values:")
                minerals = ['Calcium(mg)', 'Iron(mg)', 'Pottasium(mg)', 'Magnesium(mg)', 'Zinc(mg)'] # Add more if needed
                mineral_values = [float(predicted_millet_info[mineral]) for mineral in minerals]  # Convert to int
                mineral_options = {
                    "tooltip": {"trigger": "item"},
                    "legend": {"top": "5%", "left": "center"},
                    "series": [
                        {
                            "name": "Mineral Values",
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "color": ["#DAF7A6", "#FFC300", "#FF5733","#C70039","#900C3F"],
                            "avoidLabelOverlap": False,
                            "itemStyle": {
                                "borderRadius": 10,
                                "borderColor": "#fff",
                                "borderWidth": 2,
                            },
                            "label": {"show": False, "position": "center"},
                            "emphasis": {
                                "label": {"show": True, "fontSize": "16", "fontWeight": "bold"}
                            },
                            "labelLine": {"show": False},
                            "data": [
                                {"value": value, "name": mineral}
                                for mineral, value in zip(minerals, mineral_values)
                            ],
                        }
                    ],
                }
                st_echarts(options=mineral_options, height="500px")

                recipes_data = load_data("millets_recipe.csv")
                predicted_millet_recipe = recipes_data[recipes_data['Millet_Name'] == predicted_millet_info['Millet_Name']].squeeze()
                print(predicted_millet_recipe)

                st.subheader("Millet Recipe:")
                st.write("**Recipe Name:**", predicted_millet_recipe['Recipes_name'])
                st.image(predicted_millet_recipe['Image'], caption=predicted_millet_recipe['Recipes_name'], width=400)
                st.write("**Ingredients :**", predicted_millet_recipe['Ingredients'])
                st.write("**Recipe:**")
                recipes = predicted_millet_recipe['Recipe'].split('*')
                for recipe in recipes:
                    st.write(f"- {recipe.strip()}")


                    
                
        else:
            st.error("Please upload an image before making a prediction.")
            st_lottie(
                            lottie_error,

                            speed=1.5,
                            reverse=False,
                            loop=True,
                            quality="low",
                            # renderer="svg",
                            height=200,
                            width=200,
                            key=None,
                )
# Recipes Page
elif app_mode == "Recipes":
    st.header("Millet Recipes")
    
    # Load millet recipes data
    recipes_data = load_data("millets_recipe.csv")
    
    # Dropdown menu to select the millet type
    selected_millet = st.selectbox("Select Millet Type", recipes_data['Name'].unique())
    
    # Filter recipes for the selected millet
    selected_recipes = recipes_data[recipes_data['Name'] == selected_millet]
    
    if not selected_recipes.empty:
        # Display recipes as flex cards
        for index, row in selected_recipes.iterrows():
            st.write(f"### {row['Recipes_name']}")
            st.image(row['Image'], caption=row['Recipes_name'], width=400)
            st.write("**Ingredients:**", row['Ingredients'])
            st.write("**Recipe:**")
            recipe_sentences = row['Recipe'].split('*')
            for sentence in recipe_sentences:
                st.write(f"- {sentence.strip()}")
            st.markdown("---")
    else:
        st.write("No recipes found for the selected millet.")