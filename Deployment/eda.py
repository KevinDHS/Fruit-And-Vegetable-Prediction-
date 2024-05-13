# acces folder
import streamlit as st
import os
import glob
import random

# data loading
import numpy as np
import pandas as pd
import cv2

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from PIL import Image

def run() :
    
    st.markdown("""
        <style>
        .title {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

   
    st.markdown('<h1 class="title">Vegetables Data</h1>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.write('')
    # st.title('Vegetables Data')

    st.subheader('Exploratory Data Analysis')
    st.write('')
    st.write('')

    # create variables for path
    main_path = 'fruit-and-vegetable-image-recognition'

    # train
    train_Bean = os.path.join(main_path, 'train', 'Bean')
    train_Bitter_Gourd = os.path.join(main_path, 'train', 'Bitter_Gourd')
    train_Bottle_Gourd = os.path.join(main_path, 'train', 'Bottle_Gourd')
    train_Brinjal = os.path.join(main_path, 'train', 'Brinjal')
    train_Broccoli = os.path.join(main_path, 'train', 'Broccoli')
    train_Cabbage = os.path.join(main_path, 'train', 'Cabbage')
    train_Capsicum = os.path.join(main_path, 'train', 'Capsicum')
    train_Carrot = os.path.join(main_path, 'train', 'Carrot')
    train_Cauliflower = os.path.join(main_path, 'train', 'Cauliflower')
    train_Cucumber = os.path.join(main_path, 'train', 'Cucumber')
    train_Papaya = os.path.join(main_path, 'train', 'Papaya')
    train_Potato = os.path.join(main_path, 'train', 'Potato')
    train_Pumpkin = os.path.join(main_path, 'train', 'Pumpkin')
    train_Radish = os.path.join(main_path, 'train', 'Radish')
    train_Tomato = os.path.join(main_path, 'train', 'Tomato')

    # Creating dataset train, test, validation
    def create_dataframe(list_of_images):
        data = []
        for image in list_of_images:
            data.append((image, image.split('\\')[-2]))

        return pd.DataFrame(data, columns=['images', 'label'])

    train_df = create_dataframe(
        glob.glob(os.path.join(train_Bean, '*.jpg')) +
        glob.glob(os.path.join(train_Bitter_Gourd, '*.jpg')) +
        glob.glob(os.path.join(train_Bottle_Gourd, '*.jpg')) +
        glob.glob(os.path.join(train_Brinjal, '*.jpg')) +
        glob.glob(os.path.join(train_Broccoli, '*.jpg')) +
        glob.glob(os.path.join(train_Cabbage, '*.jpg')) +
        glob.glob(os.path.join(train_Capsicum, '*.jpg')) +
        glob.glob(os.path.join(train_Carrot, '*.jpg')) +
        glob.glob(os.path.join(train_Cauliflower, '*.jpg')) +
        glob.glob(os.path.join(train_Cucumber, '*.jpg')) +
        glob.glob(os.path.join(train_Papaya, '*.jpg')) +
        glob.glob(os.path.join(train_Potato, '*.jpg')) +
        glob.glob(os.path.join(train_Pumpkin, '*.jpg')) +
        glob.glob(os.path.join(train_Radish, '*.jpg')) +
        glob.glob(os.path.join(train_Tomato, '*.jpg'))
    )
    train_df = train_df.sample(frac=1, random_state=7).reset_index(drop=True)

    # function to visualize each class
    def visualize_samples_by_label(df, label, num_samples=20):
        samples = df[df['label'] == label]['images'].iloc[:num_samples].tolist()
        num_cols = min(num_samples, 5)
        num_rows = (num_samples - 1) // num_cols + 1
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 2 * num_rows))
        count = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if count < len(samples):
                    sample = samples[count]
                    img = Image.open(sample)
                    ax = axes[i, j]
                    ax.imshow(img)
                    ax.axis('off')
                    count += 1
        plt.tight_layout()
        st.pyplot(fig)
        

    # visualize 'Bean' class
    st.write(' 1. Apple')
    visualize_samples_by_label(train_df, 'apple', num_samples=20)
    st.write('**Characteristics**')
    st.write('Color : green \n\n Shape: Beans are typically small, elongated, and have a curved shape.\n\n Nutritional Value: Beans are known for their high nutritional value. They are a good source of plant-based protein, dietary fiber, vitamins (such as folate and vitamin B6), minerals (such as iron and magnesium), and antioxidants.')
    st.write('Reference : https://www.medicalnewstoday.com/articles/320192#benefits')

    st.markdown('----')

    # visualize 'Bitter_Gourd' class
    st.write(' 2. Banana')
    visualize_samples_by_label(train_df, 'banana', num_samples=20)
    st.write('**Characteristics**')
    st.write('Shape and Appearance: Bitter gourd has a long, elongated shape with a rough, bumpy skin. The surface of the vegetable is typically green, and it may have ridges or spikes along its length.')
    st.write('Color: Bitter gourds are generally green.')
    st.write('**Benefits**')
    st.write('Bitter melon may be associated with adverse side effects. Pregnant women, people with underlying health problems, and those taking blood sugar-lowering medications should consult their doctor before use.')
    st.write('Reference : https://www.healthline.com/nutrition/bitter-melon#TOC_TITLE_HDR_9')

    st.markdown('----')

    # visualize 'Bottle_Gourd' class
    st.write(' 3. Cabbage')
    visualize_samples_by_label(train_df, 'cabbage', num_samples=20)
    st.write('**Characteristics**')
    st.write('Shape and Appearance: Bottle gourd has a unique elongated shape resembling a bottle or gourd.')
    st.write('Color : Bottle gourd usually has a light green or pale green color.')
    st.write('**Benefits**')
    st.write('Reduces stress, Benefits the heart, Helps in weight loss, Helps in treating sleeping disorders.')
    st.write('Reference : https://indianexpress.com/article/lifestyle/food-wine/health-benefits-of-the-lauki-5245683/')

    st.markdown('----')

    # visualize 'Brinjal' class
    st.write(' 4. Eggplant')
    visualize_samples_by_label(train_df, 'Eggplant', num_samples=20)
    st.write('**Characteristics**')
    st.write('Shape and Size: Brinjal has an elongated or oval shape with a smooth and shiny skin. The size can vary, ranging from small to large, depending on the variety. Some varieties may have a round or cylindrical shape.')
    st.write('Color: Brinjal comes in a variety of colors, including dark purple, light purple, green, white, and even striped or mottled combinations of these colors. The color may vary depending on the specific variety and stage of ripeness..')
    st.write('note : dataset used green Brinjal')
    st.write('**Benefits**')
    st.write('Eggplant is high in fiber but low in calories, both of which can help promote weight loss. It can also be used in place of higher-calorie ingredients..')
    st.write('Reference : https://www.healthline.com/nutrition/eggplant-benefits#TOC_TITLE_HDR_6')

    st.markdown('----')

    # visualize 'Broccoli' class
    st.write(' 5. Garlic')
    visualize_samples_by_label(train_df, 'garlic', num_samples=20)
    st.write('**Characteristics**')
    st.write('Appearance: Broccoli has a distinctive appearance, with a compact head composed of numerous small, tightly packed flower buds called florets. The florets are attached to a thick, edible stalk. The color of broccoli can range from dark green to bluish-green..')
    st.write('Shape: The head of broccoli is typically rounded or dome-shaped, resembling a small tree or bush. The florets form a clustered shape that extends from the stalk.')
    st.write('Color: The florets of broccoli are dark green in color, while the stalk is lighter green. Some varieties of broccoli may have purple or yellowish-green florets.')
    st.write('**Benefits**')
    st.write('Broccoli is high in many vitamins and minerals, including folate, potassium, manganese, and vitamins C and K1.')
    st.write('Reference : https://www.healthline.com/nutrition/foods/broccoli#plant-compounds')

    st.markdown('----')

    st.subheader('Analysis')
    st.write('Mostly image in Datasets have a Green colors and Many image have same color .')



if __name__=='__main__':
    run()
