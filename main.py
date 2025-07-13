import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import plotly.express as px
from streamlit_option_menu import option_menu
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function
import home_page
import base64


st.set_page_config(page_icon="✨", page_title="AutoEDA")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)




# Create a Streamlit sidebar
st.sidebar.title("AutoEDA: Automated Exploratory Data Analysis and Processing")

# Set custom CSS
custom_css = home_page.custom_css


# Create the introduction section
st.title("Welcome to AutoEDA")
# st.write('<div class="tagline">Unleash the Power of Data with AutoEDA!</div>', unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=['Home', 'Data Exploration', 'Data Preprocessing'],
    icons=['house-heart', 'bar-chart-fill', 'hammer'],
    orientation='horizontal'
)

if selected == 'Home':
    home_page.show_home_page()


# Create a button in the sidebar to upload CSV
uploaded_file = st.sidebar.file_uploader("Upload Your CSV File Here", type=["csv","xls"])
use_example_data = st.sidebar.checkbox("Use Example Titanic Dataset", value=False)


if uploaded_file:
    df = function.load_data(uploaded_file)


    # get a copy of original df from the session state or create a new one. this is for preprocessing purposes
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df.copy()

    

# Create a checkbox in the sidebar to choose between the example dataset and uploaded dataset

elif use_example_data:
    # Load the example dataset
    df = function.load_data(file="example_dataset/titanic.csv")

    # Set st.session_state.new_df to the example dataset for data preprocessing
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df


# Display the dataset preview or any other content here
if uploaded_file is None and selected!='Home' and not use_example_data:
    # st.subheader("Welcome to DataExplora!")
    st.markdown("#### Use the sidebar to upload a CSV file or use the provided example dataset and explore your data.")
    
else:
    
    if selected=='Data Exploration':

        tab1, tab2 = st.tabs(['📊 Dataset Overview :clipboard', "🔎 Data Exploration and Visualization"])
        num_columns, cat_columns = function.categorical_numerical(df)
        
        
        with tab1: # DATASET OVERVIEW TAB
            st.subheader("1. Dataset Preview")
            st.markdown("This section provides an overview of your dataset. You can select the number of rows to display and view the dataset's structure.")
            function.display_dataset_overview(df,cat_columns,num_columns)


            st.subheader("3. Missing Values")
            function.display_missing_values(df)
            
            st.subheader("4. Data Statistics and Visualization")
            function.display_statistics_visualization(df,cat_columns,num_columns)

            st.subheader("5. Data Types")
            function.display_data_types(df)

            st.subheader("Search for a specific column or datatype")
            function.search_column(df)

        with tab2: 

            function.display_individual_feature_distribution(df,num_columns)

            st.subheader("Scatter Plot")
            function.display_scatter_plot_of_two_numeric_features(df,num_columns)


            if len(cat_columns)!=0:
                st.subheader("Categorical Variable Analysis")
                function.categorical_variable_analysis(df,cat_columns)
            else:
                st.info("The dataset does not have any categorical columns")


            st.subheader("Feature Exploration of Numerical Variables")
            if len(num_columns)!=0:
                function.feature_exploration_numerical_variables(df,num_columns)

            else:
                st.warning("The dataset does not contain any numerical variables")

            # Create a bar graph to get relationship between categorical variable and numerical variable
            st.subheader("Categorical and Numerical Variable Analysis")
            if len(num_columns)!=0 and len(cat_columns)!=0:
                function.categorical_numerical_variable_analysis(df,cat_columns,num_columns)
                
            else:
                st.warning("The dataset does not have any numerical variables. Hence Cannot Perform Categorical and Numerical Variable Analysis")
        
    # DATA PREPROCESSING  
    if selected=='Data Preprocessing':
        # st.header("🛠️ Data Preprocessing")
        # Initial setup after loading the dataset
        if "original_df" not in st.session_state:
            st.session_state.original_df = df.copy()

        if "new_df" not in st.session_state:
            st.session_state.new_df = df.copy()

# Revert Button
        revert = st.button("Revert to Original Dataset", key="revert_button")

        if revert:
            st.session_state.new_df = st.session_state.original_df.copy()
            st.success("Dataset reverted to original.")


        # REMOVING UNWANTED COLUMNS
        st.subheader("Remove Unwanted Columns")
        columns_to_remove = st.multiselect(label='Select Columns to Remove',options=st.session_state.new_df.columns)

        if st.button("Remove Selected Columns"):
            if columns_to_remove:
                st.session_state.new_df = preprocessing_function.remove_selected_columns(st.session_state.new_df,columns_to_remove)
                st.success("Selected Columns Removed Sucessfully")
                
        st.dataframe(st.session_state.new_df)
       

       # Handle missing values in the dataset
        st.subheader("Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()

        if missing_count.any():

            selected_missing_option = st.selectbox(
                "Select how to handle missing data:",
                ["Remove Rows in Selected Columns", "Fill Missing Data in Selected Columns (Numerical Only)"]
            )

            if selected_missing_option == "Remove Rows in Selected Columns":
                columns_to_remove_missing = st.multiselect("Select columns to remove rows with missing data", options=st.session_state.new_df.columns)
                if st.button("Remove Rows with Missing Data"):
                    st.session_state.new_df = preprocessing_function.remove_rows_with_missing_data(st.session_state.new_df, columns_to_remove_missing)
                    st.success("Rows with missing data removed successfully.")

            elif selected_missing_option == "Fill Missing Data in Selected Columns (Numerical Only)":
                numerical_columns_to_fill = st.multiselect("Select numerical columns to fill missing data", options=st.session_state.new_df.select_dtypes(include=['number']).columns)
                fill_method = st.selectbox("Select fill method:", ["mean", "median", "mode"])
                if st.button("Fill Missing Data"):
                    if numerical_columns_to_fill:
                        st.session_state.new_df = preprocessing_function.fill_missing_data(st.session_state.new_df, numerical_columns_to_fill, fill_method)
                        st.success(f"Missing data in numerical columns filled with {fill_method} successfully.")

                    else:
                        st.warning("Please select a column to fill in the missing data")

            function.display_missing_values(st.session_state.new_df)

        else:
            st.info("The dataset does not contain any missing values")
    # DETECTING AND REMOVING DUPLICATES
        st.subheader("Duplicate Rows Detection")

        st.subheader("🧹 Duplicate Handling")

# Paste your block here
        # Initialize session state variables
        if "duplicate_count" not in st.session_state:
            st.session_state.duplicate_count = 0
            st.session_state.duplicate_rows = pd.DataFrame()
            st.session_state.duplicate_checked = False  # Track if button was clicked

        if st.button("Detect Duplicates"):
            st.session_state.duplicate_count, st.session_state.duplicate_rows = preprocessing_function.detect_duplicates(st.session_state.new_df)
            st.session_state.duplicate_checked = True  # Set flag to True

# Show result only if user clicked "Detect Duplicates"
        if st.session_state.duplicate_checked:
            if st.session_state.duplicate_count > 0:
                st.warning(f"⚠️ Found {st.session_state.duplicate_count} duplicate rows.")
                st.dataframe(st.session_state.duplicate_rows)
        else:
            st.success("✅ No duplicate rows found.")


        if st.button("Remove Duplicates"):
            if st.session_state.duplicate_count > 0:
                st.session_state.new_df, _ = preprocessing_function.remove_duplicates(st.session_state.new_df)
            st.session_state.duplicate_count = 0
            st.session_state.duplicate_rows = pd.DataFrame()
            st.success("✅ Duplicate rows deleted. Remaining duplicates: 0")

        # Display updated DataFrame
        # st.dataframe(st.session_state.new_df)


        encoding_tooltip = '''**One-Hot encoding** converts categories into binary values (0 or 1). It's like creating checkboxes for each category. This makes it possible for computers to work with categorical data.
        **Label encoding** assigns unique numbers to categories. It's like giving each category a name (e.g., Red, Green, Blue becomes 1, 2, 3). This helps computers understand and work with categories.
        '''
        st.subheader("Encode Categorical Data")

        new_df_categorical_columns = st.session_state.new_df.select_dtypes(include=['object']).columns

        if not new_df_categorical_columns.empty:
            select_categorical_columns = st.multiselect("Select Columns to perform encoding",new_df_categorical_columns)

            #choose the encoding method
            encoding_method = st.selectbox("Select Encoding Method:",['One Hot Encoding','Label Encoding'],help=encoding_tooltip)
    

            if st.button("Apply Encoding"):
                if encoding_method=="One Hot Encoding":
                    st.session_state.new_df = preprocessing_function.one_hot_encode(st.session_state.new_df,select_categorical_columns)
                    st.success("One-Hot Encoding Applied Sucessfully")

                if encoding_method=="Label Encoding":
                    st.session_state.new_df = preprocessing_function.label_encode(st.session_state.new_df,select_categorical_columns)
                    st.success("Label Encoding Applied Sucessfully")


            st.dataframe(st.session_state.new_df)
        else:
            st.info("The dataset does not contain any categorical columns")

        feature_scaling_tooltip='''**Standardization** scales your data to have a mean of 0 and a standard deviation of 1. It helps in comparing variables with different units. Think of it like making all values fit on the same measurement scale.
        **Min-Max scaling** transforms your data to fall between 0 and 1. It's like squeezing data into a specific range. This makes it easier to compare data points that vary widely.'''


        st.subheader("Feature Scaling")
        new_df_numerical_columns = st.session_state.new_df.select_dtypes(include=['number']).columns
        selected_columns = st.multiselect("Select Numerical Columns to Scale", new_df_numerical_columns)

        scaling_method = st.selectbox("Select Scaling Method:", ['Standardization', 'Min-Max Scaling'],help=feature_scaling_tooltip)

        if st.button("Apply Scaling"):
            if selected_columns:
                if scaling_method == "Standardization":
                    st.session_state.new_df = preprocessing_function.standard_scale(st.session_state.new_df, selected_columns)
                    st.success("Standardization Applied Successfully.")
                elif scaling_method == "Min-Max Scaling":
                    st.session_state.new_df = preprocessing_function.min_max_scale(st.session_state.new_df, selected_columns)
                    st.success("Min-Max Scaling Applied Successfully.")
            else:
                st.warning("Please select numerical columns to scale.")

        st.dataframe(st.session_state.new_df)

        st.subheader("Identify and Handle Outliers")

        
        selected_numeric_column = st.selectbox("Select Numeric Column for Outlier Handling:", new_df_numerical_columns)
        st.write(f"Selected column: {selected_numeric_column}")

# Boxplot to show outliers
        fig, ax = plt.subplots()
        sns.boxplot(data=st.session_state.new_df, x=selected_numeric_column, ax=ax)
        st.pyplot(fig)

# Detect outliers using IQR
        outliers = preprocessing_function.detect_outliers_iqr(st.session_state.new_df, selected_numeric_column)
        if outliers:
            st.warning("Detected Outliers:")
            st.write(outliers)
        else:
            st.info("No outliers detected using IQR.")

# Handling method selection
        outlier_handling_method = st.selectbox("Select Outlier Handling Method:", ["Remove Outliers", "Transform Outliers"])

# Apply selected method
        if st.button("Apply Outlier Handling"):
            if outlier_handling_method == "Remove Outliers":
                st.session_state.new_df = preprocessing_function.remove_outliers(
                st.session_state.new_df, selected_numeric_column, outliers)
                st.success("Outliers removed successfully.")
            elif outlier_handling_method == "Transform Outliers":
                st.session_state.new_df = preprocessing_function.transform_outliers(
                st.session_state.new_df, selected_numeric_column, outliers
        )
                st.success("Outliers transformed successfully.")

        # # Show the updated dataset
        # st.dataframe(st.session_state.new_df)
        
        # if st.session_state.new_df is not None:
        #     # Convert the DataFrame to CSV
        #     csv = st.session_state.new_df.to_csv(index=False)
        #     # Encode as base64
        #     b64 = base64.b64encode(csv.encode()).decode()
        #     # Create a download link
        #     href = f'data:file/csv;base64,{b64}'
        #     # Display a download button
        #     st.markdown(f'<a href="{href}" download="preprocessed_data.csv"><button>Download Preprocessed Data</button></a>', unsafe_allow_html=True)
        # else:
        #     st.warning("No preprocessed data available to download.")

        st.subheader("Preprocessed Dataset Preview")
        st.dataframe(st.session_state.new_df)
        st.write("Shape of current preprocessed data:", st.session_state.new_df.shape)


        if st.session_state.new_df is not None:
            csv = st.session_state.new_df.to_csv(index=False).encode('utf-8')
            st.download_button(
            label="Download Preprocessed Data",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv"
    )
        else:
            st.warning("No preprocessed data available to download.")
