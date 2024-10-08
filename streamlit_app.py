import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor

retail_store_data = pd.read_csv('csv/retail_store_data_category.csv')

top_category = retail_store_data.groupby('category')['total_amount'].sum().nlargest(3)
category_names = top_category.index.tolist()

rice_data = retail_store_data[retail_store_data['category'] == 'Rice']

# Extract relevant columns for quantities and amounts
quantity_columns = [col for col in rice_data.columns if 'quantity' in col.lower()]
amount_columns = [col for col in rice_data.columns if 'amount' in col.lower()]

# Melt the DataFrame to long format for quantities
quantities_melted = rice_data.melt(id_vars=['product_name', 'category'],
                                value_vars=quantity_columns,
                                var_name='month',
                                value_name='quantity')

# Melt the DataFrame to long format for amounts
amounts_melted = rice_data.melt(id_vars=['product_name', 'category'],
                                value_vars=amount_columns,
                                var_name='month',
                                value_name='amount')

# Remove '_quantity' and '_amount' suffix from the month column
quantities_melted['month'] = quantities_melted['month'].str.replace('_quantity', '')
amounts_melted['month'] = amounts_melted['month'].str.replace('_amount', '')

# Merge quantities and amounts dataframes on product_name, category, and month
monthly_sales = pd.merge(quantities_melted, amounts_melted, on=['product_name', 'category', 'month'])

# Filter out rows where both quantity and amount are zero to clean the data
monthly_sales = monthly_sales[(monthly_sales['quantity'] != 0) | (monthly_sales['amount'] != 0)]

# Group by month and sum the quantities and amounts
monthly_sales_summary = monthly_sales.groupby('month').sum().reset_index()

# Normalize the month names to a consistent format
def normalize_month(month):
    month_mapping = {
        'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
        'JULY': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
    }
    parts = month.split('_')
    if len(parts) > 1 and len(parts[1]) == 4:  # Check if year is already present in the correct format
        return month
    if parts[0].upper() in month_mapping:
        return f"{month_mapping[parts[0].upper()]}-{parts[1]}"
    return month

monthly_sales_summary['month'] = monthly_sales_summary['month'].apply(normalize_month)

monthly_rice_sales_summary = monthly_sales_summary[['month', 'quantity', 'amount']].rename(columns={'quantity': 'total_quantity', 'amount': 'total_amount'})
monthly_rice_sales_summary = monthly_rice_sales_summary[:-1]

monthly_rice_sales_summary['month']=pd.to_datetime(monthly_rice_sales_summary['month'], format='%b-%y')
monthly_rice_sales_summary = monthly_rice_sales_summary.sort_values(by='month')

oil_data = retail_store_data[retail_store_data['category'] == 'Oils']
# Extract relevant columns for quantities and amounts
quantity_columns = [col for col in oil_data.columns if 'quantity' in col.lower()]
amount_columns = [col for col in oil_data.columns if 'amount' in col.lower()]

# Melt the DataFrame to long format for quantities
quantities_melted = oil_data.melt(id_vars=['product_name', 'category'],
                                   value_vars=quantity_columns,
                                   var_name='month',
                                   value_name='quantity')

# Melt the DataFrame to long format for amounts
amounts_melted = oil_data.melt(id_vars=['product_name', 'category'],
                                value_vars=amount_columns,
                                var_name='month',
                                value_name='amount')

# Remove '_quantity' and '_amount' suffix from the month column
quantities_melted['month'] = quantities_melted['month'].str.replace('_quantity', '')
amounts_melted['month'] = amounts_melted['month'].str.replace('_amount', '')

# Merge quantities and amounts dataframes on product_name, category, and month
monthly_sales = pd.merge(quantities_melted, amounts_melted, on=['product_name', 'category', 'month'])

# Filter out rows where both quantity and amount are zero to clean the data
monthly_sales = monthly_sales[(monthly_sales['quantity'] != 0) | (monthly_sales['amount'] != 0)]

# Group by month and sum the quantities and amounts
monthly_sales_summary = monthly_sales.groupby('month').sum().reset_index()

# Normalize the month names to a consistent format
def normalize_month(month):
    month_mapping = {
        'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
        'JULY': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
    }
    parts = month.split('_')
    if len(parts) > 1 and len(parts[1]) == 4:  # Check if year is already present in the correct format
        return month
    if parts[0].upper() in month_mapping:
        return f"{month_mapping[parts[0].upper()]}-{parts[1]}"
    return month

monthly_sales_summary['month'] = monthly_sales_summary['month'].apply(normalize_month)

# Create the final DataFrame with only month, total quantity, and total amount
monthly_oil_sales_summary = monthly_sales_summary[['month', 'quantity', 'amount']].rename(columns={'quantity': 'total_quantity', 'amount': 'total_amount'})

monthly_oil_sales_summary = monthly_oil_sales_summary[:-1]
monthly_oil_sales_summary['month']=pd.to_datetime(monthly_oil_sales_summary['month'], format='%b-%y')
monthly_oil_sales_summary = monthly_oil_sales_summary.sort_values(by='month')

lentils_data = retail_store_data[retail_store_data['category'] == 'Lentils']

# Extract relevant columns for quantities and amounts
quantity_columns = [col for col in lentils_data.columns if 'quantity' in col.lower()]
amount_columns = [col for col in lentils_data.columns if 'amount' in col.lower()]

# Melt the DataFrame to long format for quantities
quantities_melted = lentils_data.melt(id_vars=['product_name', 'category'],
                                   value_vars=quantity_columns,
                                   var_name='month',
                                   value_name='quantity')

# Melt the DataFrame to long format for amounts
amounts_melted = lentils_data.melt(id_vars=['product_name', 'category'],
                                value_vars=amount_columns,
                                var_name='month',
                                value_name='amount')

# Remove '_quantity' and '_amount' suffix from the month column
quantities_melted['month'] = quantities_melted['month'].str.replace('_quantity', '')
amounts_melted['month'] = amounts_melted['month'].str.replace('_amount', '')

# Merge quantities and amounts dataframes on product_name, category, and month
monthly_sales = pd.merge(quantities_melted, amounts_melted, on=['product_name', 'category', 'month'])

# Filter out rows where both quantity and amount are zero to clean the data
monthly_sales = monthly_sales[(monthly_sales['quantity'] != 0) | (monthly_sales['amount'] != 0)]

# Group by month and sum the quantities and amounts
monthly_sales_summary = monthly_sales.groupby('month').sum().reset_index()

# Normalize the month names to a consistent format
def normalize_month(month):
    month_mapping = {
        'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
        'JULY': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
    }
    parts = month.split('_')
    if len(parts) > 1 and len(parts[1]) == 4:  # Check if year is already present in the correct format
        return month
    if parts[0].upper() in month_mapping:
        return f"{month_mapping[parts[0].upper()]}-{parts[1]}"
    return month

monthly_sales_summary['month'] = monthly_sales_summary['month'].apply(normalize_month)

# Create the final DataFrame with only month, total quantity, and total amount
monthly_lentils_sales_summary = monthly_sales_summary[['month', 'quantity', 'amount']].rename(columns={'quantity': 'total_quantity', 'amount': 'total_amount'})

monthly_lentils_sales_summary = monthly_lentils_sales_summary[:-1]
monthly_lentils_sales_summary['month']=pd.to_datetime(monthly_lentils_sales_summary['month'], format='%b-%y')
monthly_lentils_sales_summary = monthly_lentils_sales_summary.sort_values(by='month')

combined_category_data = pd.concat([
    monthly_rice_sales_summary.assign(category='Rice'),
    monthly_oil_sales_summary.assign(category='Oils'),
    monthly_lentils_sales_summary.assign(category='Lentils')
], ignore_index=True)

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)


# st.title('Sales Prediction using Machine Learning')
with st.sidebar:
    section_name = option_menu("Menu",
                               options=['Dashboard', 'Models'],
                               icons=['search', 'clipboard-data-fill'])

# if section_name == 'About': 
#     retail_store_image = Image.open('image/retail store.jpg')
#     st.image(retail_store_image)

if section_name == 'Dashboard':
    st.header('Retail Store Sales Insights')

    st.subheader('Categories by Quantity Sold')
    category_data = retail_store_data.groupby('category').sum().reset_index()
    # # Extract quantity and amount columns
    # quantity_columns = [col for col in retail_store_data.columns if 'quantity' in col]
    # amount_columns = [col for col in retail_store_data.columns if 'amount' in col]
    total_quantity = category_data[quantity_columns].sum(axis=1)
    total_amount = category_data[amount_columns].sum(axis=1)
    category_summary = pd.DataFrame({
        'Category': category_data['category'],
        'Total Quantity Sold (Units)': total_quantity,
        'Total Revenue (INR)': total_amount
    })
    category_summary_sorted = category_summary.sort_values(by='Total Quantity Sold (Units)', ascending=False)

    # Create the funnel plot (horizontal bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x=category_summary_sorted['Total Quantity Sold (Units)'],
                y=category_summary_sorted['Category'],
                palette="viridis")

    ax.set_title('Categories by Quantity Sold')
    ax.set_xlabel('Total Quantity Sold (Units)')
    ax.set_ylabel('Category')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('---')

    st.subheader('Categories by Revenue')
    # Sort the category summary by total revenue (INR) in descending order
    category_summary_sorted_by_revenue = category_summary.sort_values(by='Total Revenue (INR)', ascending=False)

    # Create the funnel plot (horizontal bar chart) for revenue
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_summary_sorted_by_revenue['Total Revenue (INR)'],
                y=category_summary_sorted_by_revenue['Category'],
                palette="viridis")

    ax.set_title('Categories by Revenue')
    ax.set_xlabel('Total Revenue (INR)')
    ax.set_ylabel('Category')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('---')

    st.subheader('Trend Analysis of the Categories')
    combined_category_data['date'] = pd.to_datetime(combined_category_data['month'], errors='coerce')  # Ensure proper datetime conversion with error handling
    combined_category_data.set_index('date', inplace=True)

    # Monthly sales trends aggregation
    monthly_trends = combined_category_data.groupby(['category', pd.Grouper(freq='M')]).agg({
        'total_quantity': 'sum',
    }).unstack(level=0)

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_trends.plot(ax=ax, subplots=True)
    # Setting title and labels
    ax.set_title('Monthly Sales Trends by Category')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Quantity')
    st.pyplot(fig)

    st.markdown('---')
    st.subheader('Top 5 Products for Each Category')

    categories_of_interest = ['Rice', 'Oils', 'Lentils']
    filtered_data = retail_store_data[retail_store_data['category'].isin(categories_of_interest)]
    category_sales = filtered_data.groupby(['category', 'product_name'])['total_quantity'].sum().reset_index()
    category_sales_sorted = category_sales.sort_values(by=['category', 'total_quantity'], ascending=[True, False])
    top_5_rice = category_sales_sorted[category_sales_sorted['category'] == 'Rice'].head(5)
    top_5_oils = category_sales_sorted[category_sales_sorted['category'] == 'Oils'].head(5)
    top_5_lentils = category_sales_sorted[category_sales_sorted['category'] == 'Lentils'].head(5)
    top_5_combined = pd.concat([top_5_rice, top_5_oils, top_5_lentils])

    # Create the barplot using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    # Define custom colors correctly matched with category names
    palette_colors = {'Rice': 'blue', 'Oils': 'purple', 'Lentils': 'green'}
    sns.barplot(x='total_quantity', y='product_name', hue='category', data=top_5_combined, dodge=False, palette=palette_colors)

    # Adding titles and labels
    ax.set_title('Top 5 Products by Total Quantity for Each Category')
    ax.set_xlabel('Total Quantity Sold')
    ax.set_ylabel('Product Name')

    # Display the plot
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('---')
    st.subheader('Distribution Analysis')
    fig, ax = plt.subplots(figsize=(10, 6))
    combined_category_data.boxplot(column='total_quantity', by='category', ax=ax, grid=False)

    # Customizations
    plt.suptitle('')  # Clears the automatic title
    plt.title('Distribution of Total Quantity by Category')  # Adds your custom title
    plt.xlabel('Category')  # Label for the x-axis
    plt.ylabel('Total Quantity')  # Label for the y-axis
    st.pyplot(fig)

    st.markdown('---')
    st.subheader("Select Category")
    category_name_user_input = st.selectbox("Select category",
                                           options=category_names, index=None)
    st.subheader("Select year")
    year_for_graph = st.selectbox("Select year",
                                           options=[2018,2019,2020,2021,2022,2023], index=None) 
    
    data_filtration_submit_button = st.button("FETCH DATA")
    # st.write('CAtegory name:',category_name_user_input)

    st.markdown('---')
    if data_filtration_submit_button:
        if category_name_user_input:
            rice_data = retail_store_data[retail_store_data['category'] == category_name_user_input]

            # Extract relevant columns for quantities and amounts
            quantity_columns = [col for col in rice_data.columns if 'quantity' in col.lower()]
            amount_columns = [col for col in rice_data.columns if 'amount' in col.lower()]

            # Melt the DataFrame to long format for quantities
            quantities_melted = rice_data.melt(id_vars=['product_name', 'category'],
                                            value_vars=quantity_columns,
                                            var_name='month',
                                            value_name='quantity')

            # Melt the DataFrame to long format for amounts
            amounts_melted = rice_data.melt(id_vars=['product_name', 'category'],
                                            value_vars=amount_columns,
                                            var_name='month',
                                            value_name='amount')

            # Remove '_quantity' and '_amount' suffix from the month column
            quantities_melted['month'] = quantities_melted['month'].str.replace('_quantity', '')
            amounts_melted['month'] = amounts_melted['month'].str.replace('_amount', '')

            # Merge quantities and amounts dataframes on product_name, category, and month
            monthly_sales = pd.merge(quantities_melted, amounts_melted, on=['product_name', 'category', 'month'])

            # Filter out rows where both quantity and amount are zero to clean the data
            monthly_sales = monthly_sales[(monthly_sales['quantity'] != 0) | (monthly_sales['amount'] != 0)]

            # Group by month and sum the quantities and amounts
            monthly_sales_summary = monthly_sales.groupby('month').sum().reset_index()

            # Normalize the month names to a consistent format
            def normalize_month(month):
                month_mapping = {
                    'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
                    'JULY': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
                }
                parts = month.split('_')
                if len(parts) > 1 and len(parts[1]) == 4:  # Check if year is already present in the correct format
                    return month
                if parts[0].upper() in month_mapping:
                    return f"{month_mapping[parts[0].upper()]}-{parts[1]}"
                return month

            monthly_sales_summary['month'] = monthly_sales_summary['month'].apply(normalize_month)

            monthly_rice_sales_summary = monthly_sales_summary[['month', 'quantity', 'amount']].rename(columns={'quantity': 'total_quantity', 'amount': 'total_amount'})
            monthly_rice_sales_summary = monthly_rice_sales_summary[:-1]

            monthly_rice_sales_summary['month']=pd.to_datetime(monthly_rice_sales_summary['month'], format='%b-%y')
            monthly_rice_sales_summary = monthly_rice_sales_summary.sort_values(by='month')

            if year_for_graph:
                df_year = monthly_rice_sales_summary[monthly_rice_sales_summary['month'].dt.year == year_for_graph]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_year['month'], df_year['total_quantity'], marker='o')

                ax.set_title(f'Monthly Sale of Units for {year_for_graph}')
                ax.set_xlabel('Month')
                ax.set_ylabel('Total Quantity')
                ax.grid(True)
                ax.set_xticks(df_year['month'])  # Ensuring x-ticks are set to the month values
                ax.set_xticklabels(df_year['month'].dt.strftime('%b'), rotation=45)  # Formatting month names and rotating labels
                st.pyplot(fig)
                    
                st.markdown('---')

#Monthly sales display for rice category
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(monthly_rice_sales_summary['month'], monthly_rice_sales_summary['total_quantity'], marker='o', color='b', label='Total Quantity')

            # Adding labels and title
            ax.set_xlabel('Year')
            ax.set_ylabel('Total Quantity')
            ax.set_title(f'Monthly Sales of {category_name_user_input} Units Overall')

            # Format x-axis to show only years
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensures only whole numbers (years) are shown
            years = pd.DatetimeIndex(monthly_rice_sales_summary['month']).year  # Extract years
            unique_years = sorted(set(years))  # Get unique years
            ax.set_xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years])  # Set ticks at the start of each unique year
            ax.set_xticklabels(unique_years, rotation=45)  # Set tick labels to show only unique years
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.markdown('---')

#Yearly display of the sales for rice category
            monthly_rice_sales_summary['year'] = monthly_rice_sales_summary['month'].dt.year

            # Grouping by year to get total quantity per year
            yearly_sales_summary = monthly_rice_sales_summary.groupby('year')['total_quantity'].sum().reset_index()

            # Plotting the data using matplotlib and displaying it with Streamlit
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_sales_summary['year'], yearly_sales_summary['total_quantity'], marker='o', color='b', label='Total Quantity')

            # Adding labels and title
            ax.set_xlabel('Year')
            ax.set_ylabel('Total Quantity')
            ax.set_title(f'Yearly Sales of {category_name_user_input} Units')
            ax.set_xticks(yearly_sales_summary['year'])  # Setting x-ticks to display only years
            ax.set_xticklabels(yearly_sales_summary['year'], rotation=45)  # Rotating x-axis labels for better readability
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.markdown('---')

#Average sales for months
            monthly_rice_sales_summary['month_name'] = monthly_rice_sales_summary['month'].dt.month_name()

            # Group by month and calculate mean
            month_grouped = monthly_rice_sales_summary.groupby('month_name').mean().sort_values(by='total_quantity')

            # Plotting month-wise analysis using seaborn on a matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=month_grouped.index, y='total_quantity', data=month_grouped, ax=ax)
            ax.set_title(f'Average Sale of {category_name_user_input} by Month')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Total Quantity')
            st.pyplot(fig)

if section_name == 'Models':
    st.subheader("Select Category for Model")
    category_name_user_input = st.selectbox("Select category",
                                           options=category_names, index=None)
    
    if category_name_user_input == 'Rice':
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['LINEAR REGRESSION','ADABoost','XGBoost', 'RANDOM FOREST', 'PREDICTIONS'])
        
        rice_data = retail_store_data[retail_store_data['category'] == 'Rice']

        # Extract relevant columns for quantities and amounts
        quantity_columns = [col for col in rice_data.columns if 'quantity' in col.lower()]
        amount_columns = [col for col in rice_data.columns if 'amount' in col.lower()]

        # Melt the DataFrame to long format for quantities
        quantities_melted = rice_data.melt(id_vars=['product_name', 'category'],
                                        value_vars=quantity_columns,
                                        var_name='month',
                                        value_name='quantity')

        # Melt the DataFrame to long format for amounts
        amounts_melted = rice_data.melt(id_vars=['product_name', 'category'],
                                        value_vars=amount_columns,
                                        var_name='month',
                                        value_name='amount')

        # Remove '_quantity' and '_amount' suffix from the month column
        quantities_melted['month'] = quantities_melted['month'].str.replace('_quantity', '')
        amounts_melted['month'] = amounts_melted['month'].str.replace('_amount', '')

        # Merge quantities and amounts dataframes on product_name, category, and month
        monthly_sales = pd.merge(quantities_melted, amounts_melted, on=['product_name', 'category', 'month'])

        # Filter out rows where both quantity and amount are zero to clean the data
        monthly_sales = monthly_sales[(monthly_sales['quantity'] != 0) | (monthly_sales['amount'] != 0)]

        # Group by month and sum the quantities and amounts
        monthly_sales_summary = monthly_sales.groupby('month').sum().reset_index()

        # Normalize the month names to a consistent format
        def normalize_month(month):
            month_mapping = {
                'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
                'JULY': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
            }
            parts = month.split('_')
            if len(parts) > 1 and len(parts[1]) == 4:  # Check if year is already present in the correct format
                return month
            if parts[0].upper() in month_mapping:
                return f"{month_mapping[parts[0].upper()]}-{parts[1]}"
            return month

        monthly_sales_summary['month'] = monthly_sales_summary['month'].apply(normalize_month)

        monthly_rice_sales_summary = monthly_sales_summary[['month', 'quantity', 'amount']].rename(columns={'quantity': 'total_quantity', 'amount': 'total_amount'})
        monthly_rice_sales_summary = monthly_rice_sales_summary[:-1]

        monthly_rice_sales_summary['month']=pd.to_datetime(monthly_rice_sales_summary['month'], format='%b-%y')
        monthly_rice_sales_summary = monthly_rice_sales_summary.sort_values(by='month')
        monthly_rice_sales_summary['month_name'] = monthly_rice_sales_summary['month'].dt.month_name()
        monthly_rice_sales_summary['year'] = monthly_rice_sales_summary['month'].dt.year
        
        outliers_quantity = detect_outliers(monthly_rice_sales_summary, 'total_quantity')
        outliers_amount = detect_outliers(monthly_rice_sales_summary, 'total_amount')

        df_capped_rice = monthly_rice_sales_summary.copy()
        cap_outliers(df_capped_rice, 'total_quantity')
        cap_outliers(df_capped_rice, 'total_amount')

        with tab1:
            rice_data_lr = df_capped_rice.copy()
            for i in range(1, 3):
                rice_data_lr[f'lag{i}'] = rice_data_lr['total_quantity'].shift(i)

            rice_data_lr.dropna(inplace=True)   
            # Split data into features (X) and target variable (y)
            X = rice_data_lr.drop(['month', 'total_quantity','month_name'], axis=1)
            y = rice_data_lr['total_quantity'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # st.subheader("Linear Regression Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,y_pred)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,y_pred),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,y_pred),2)}")

            st.subheader("Linear Regression Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(rice_data_lr['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(rice_data_lr['month'][len(X_train):], y_pred, label='Predicted', color='red', linestyle='--')
            # Setting labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
        with tab2:
            rice_data_ada = df_capped_rice.copy()
            rice_data_ada['rolling_mean1'] = rice_data_ada['total_quantity'].rolling(window=2).mean()
            for i in range(1, 7):
                rice_data_ada[f'moving_average_{i}'] = rice_data_ada['total_quantity'].shift(i)    
            rice_data_ada.dropna(inplace=True)
            # Split data into features (X) and target variable (y)
            X = rice_data_ada.drop(['month', 'total_quantity','month_name'], axis=1)
            y = rice_data_ada['total_quantity']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            adaboost = AdaBoostRegressor()
            adaboost.fit(X_train, y_train)
            adaboost_pred = adaboost.predict(X_test)

            # st.subheader("ADABoost Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,adaboost_pred)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,adaboost_pred),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,adaboost_pred),2)}")
            
            st.subheader("ADABoost Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(rice_data_ada['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(rice_data_ada['month'][len(X_train):], adaboost_pred, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)  

        with tab3:
            rice_data_xgb = df_capped_rice.copy()
            rice_data_xgb['rolling_mean1'] = rice_data_xgb['total_quantity'].rolling(window=2).mean()
            for i in range(1, 2):
                rice_data_xgb[f'moving_average_{i}'] = rice_data_xgb['total_quantity'].shift(i)
            rice_data_xgb.dropna(inplace=True)
            # Split data into features (X) and target variable (y)
            X = rice_data_xgb.drop(['month', 'total_quantity','month_name'], axis=1)
            y = rice_data_xgb['total_quantity']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            xgb = XGBRFRegressor()
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)  

            # st.subheader("XGBoost Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,xgb_pred)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,xgb_pred),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,xgb_pred),2)}")
            
            st.subheader("XGBoost Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(rice_data_xgb['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(rice_data_xgb['month'][len(X_train):], xgb_pred, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig) 

        with tab4:
            rice_data_rf = df_capped_rice.copy()
            rice_data_rf['rolling_mean1'] = rice_data_rf['total_quantity'].rolling(window=2).mean()
            for i in range(1, 2):
                rice_data_rf[f'moving_average_{i}'] = rice_data_rf['total_quantity'].shift(i)
            rice_data_rf.dropna(inplace=True)  
            # Split data into features (X) and target variable (y)
            X = rice_data_rf.drop(['month', 'total_quantity','month_name'], axis=1)
            y = rice_data_rf['total_quantity']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)  

            # st.subheader("Random Forest Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,y_pred_rf)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,y_pred_rf),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,y_pred_rf),2)}")

            st.subheader("Random Forest Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(rice_data_rf['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(rice_data_rf['month'][len(X_train):], y_pred_rf, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
        with tab5:
            # Take the last known data point (X_test.iloc[-1]) and create a copy
            future_predictions = []
            last_known_data = X_test.iloc[-1:].copy()

            # Drop 'moving_average_2' from last_known_data if the model was trained without it
            if 'moving_average_2' in last_known_data.columns:
                last_known_data = last_known_data.drop(columns=['moving_average_2'])

            # Start month-on-month predictions
            for i in range(3):  # Predicting for the next 3 months
                # Predict the next value
                next_pred = xgb.predict(last_known_data)[0]

                # Append the prediction to the list
                future_predictions.append(next_pred)

                # Update lag features using the predicted value
                # Shift the lag features to prepare for the next prediction
                last_known_data[f'moving_average_1'] = next_pred
                last_known_data[f'moving_average_2'] = last_known_data[f'moving_average_1']

                # Drop 'moving_average_2' again if necessary in this loop (if not used during training)
                if 'moving_average_2' in last_known_data.columns:
                    last_known_data = last_known_data.drop(columns=['moving_average_2'])

            st.write(f'Predictions for July: {future_predictions[0]:.2f} KGS')    
            st.write(f'Predictions for August: {future_predictions[1]:.2f} KGS')
            st.write(f'Predictions for September: {future_predictions[2]:.2f} KGS')    
    
    if category_name_user_input == 'Oils':
        oil_data = retail_store_data[retail_store_data['category'] == 'Oils']
        # Extract relevant columns for quantities and amounts
        quantity_columns = [col for col in oil_data.columns if 'quantity' in col.lower()]
        amount_columns = [col for col in oil_data.columns if 'amount' in col.lower()]

        # Melt the DataFrame to long format for quantities
        quantities_melted = oil_data.melt(id_vars=['product_name', 'category'],
                                        value_vars=quantity_columns,
                                        var_name='month',
                                        value_name='quantity')

        # Melt the DataFrame to long format for amounts
        amounts_melted = oil_data.melt(id_vars=['product_name', 'category'],
                                        value_vars=amount_columns,
                                        var_name='month',
                                        value_name='amount')

        # Remove '_quantity' and '_amount' suffix from the month column
        quantities_melted['month'] = quantities_melted['month'].str.replace('_quantity', '')
        amounts_melted['month'] = amounts_melted['month'].str.replace('_amount', '')

        # Merge quantities and amounts dataframes on product_name, category, and month
        monthly_sales = pd.merge(quantities_melted, amounts_melted, on=['product_name', 'category', 'month'])

        # Filter out rows where both quantity and amount are zero to clean the data
        monthly_sales = monthly_sales[(monthly_sales['quantity'] != 0) | (monthly_sales['amount'] != 0)]

        # Group by month and sum the quantities and amounts
        monthly_sales_summary = monthly_sales.groupby('month').sum().reset_index()

        # Normalize the month names to a consistent format
        def normalize_month(month):
            month_mapping = {
                'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
                'JULY': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
            }
            parts = month.split('_')
            if len(parts) > 1 and len(parts[1]) == 4:  # Check if year is already present in the correct format
                return month
            if parts[0].upper() in month_mapping:
                return f"{month_mapping[parts[0].upper()]}-{parts[1]}"
            return month

        monthly_sales_summary['month'] = monthly_sales_summary['month'].apply(normalize_month)

        # Create the final DataFrame with only month, total quantity, and total amount
        monthly_oil_sales_summary = monthly_sales_summary[['month', 'quantity', 'amount']].rename(columns={'quantity': 'total_quantity', 'amount': 'total_amount'})

        monthly_oil_sales_summary = monthly_oil_sales_summary[:-1]
        monthly_oil_sales_summary['month']=pd.to_datetime(monthly_oil_sales_summary['month'], format='%b-%y')
        monthly_oil_sales_summary = monthly_oil_sales_summary.sort_values(by='month')
        monthly_oil_sales_summary['year'] = monthly_oil_sales_summary['month'].dt.year
        monthly_oil_sales_summary['quarter'] = monthly_oil_sales_summary['month'].dt.quarter
        monthly_oil_sales_summary['month_num'] = monthly_oil_sales_summary['month'].dt.month

        outliers_quantity = detect_outliers(monthly_oil_sales_summary, 'total_quantity')
        outliers_amount = detect_outliers(monthly_oil_sales_summary, 'total_amount')

        df_capped_oil = monthly_oil_sales_summary.copy()
        cap_outliers(df_capped_oil, 'total_quantity')
        cap_outliers(df_capped_oil, 'total_amount')

        tab1, tab2, tab3, tab4, tab5 = st.tabs(['LINEAR REGRESSION','ADABoost','XGBoost', 'RANDOM FOREST', 'PREDICTIONS'])

        oil_data = df_capped_oil.copy()
        for i in range(1, 4):
            oil_data[f'moving_average{i}'] = oil_data['total_quantity'].shift(i)

        oil_data.dropna(inplace=True)  
        # Split data into features (X) and target variable (y)
        X = oil_data.drop(['month', 'total_quantity'], axis=1)
        y = oil_data['total_quantity'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False) 

        with tab1:  
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # st.subheader("Linear Regression Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,y_pred)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,y_pred),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,y_pred),2)}")

            st.subheader("Linear Regression Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(oil_data['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(oil_data['month'][len(X_train):], y_pred, label='Predicted', color='red', linestyle='--')
            # Setting labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            adaboost = AdaBoostRegressor()
            adaboost.fit(X_train, y_train)
            adaboost_pred = adaboost.predict(X_test)

            # st.subheader("ADABoost Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,adaboost_pred)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,adaboost_pred),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,adaboost_pred),2)}")
            
            st.subheader("ADABoost Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(oil_data['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(oil_data['month'][len(X_train):], adaboost_pred, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)  

        with tab3:
            xgb = XGBRFRegressor()
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)

            # st.subheader("XGBoost Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,xgb_pred)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,xgb_pred),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,xgb_pred),2)}")
            
            st.subheader("XGBoost Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(oil_data['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(oil_data['month'][len(X_train):], xgb_pred, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig) 

        with tab4:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            # st.subheader("Random Forest Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,y_pred_rf)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,y_pred_rf),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,y_pred_rf),2)}")

            st.subheader("Random Forest Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(oil_data['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(oil_data['month'][len(X_train):], y_pred_rf, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig) 
        with tab5:
            if 'moving_average_1' not in oil_data.columns:
                oil_data['moving_average_1'] = oil_data['total_quantity'].shift(1)
                oil_data['moving_average_2'] = oil_data['total_quantity'].shift(2)
                oil_data['moving_average_3'] = oil_data['total_quantity'].shift(3)

            # Now that 'moving_average_1' exists, proceed with your future prediction
            last_total_amount = oil_data['total_amount'].iloc[-1]
            last_year = oil_data['year'].iloc[-1]
            last_quarter = oil_data['quarter'].iloc[-1]
            last_month_num = oil_data['month_num'].iloc[-1]  # Assuming 'month_num' is the numeric representation of months
            last_moving_avg1 = oil_data['moving_average_1'].iloc[-1]
            last_moving_avg2 = oil_data['moving_average_2'].iloc[-1]
            last_moving_avg3 = oil_data['moving_average_3'].iloc[-1]

            # Create a DataFrame to hold future data for the next 3 months
            future_data = pd.DataFrame({
                'total_amount': [last_total_amount] * 3,  # Assuming future total amount remains the same for simplicity
                'year': [last_year + 1] * 3,  # Increment the year for future predictions if necessary
                'quarter': [(last_quarter % 4) + 1, ((last_quarter + 1) % 4) + 1, ((last_quarter + 2) % 4) + 1],  # Cycle through quarters
                'month_num': [(last_month_num % 12) + 1, ((last_month_num + 1) % 12) + 1, ((last_month_num + 2) % 12) + 1],  # Add the 'month_num' column
                'moving_average1': [last_moving_avg1, np.nan, np.nan],  # Use correct name as per training data
                'moving_average2': [last_moving_avg2, np.nan, np.nan],  # Use correct name as per training data
                'moving_average3': [last_moving_avg3, np.nan, np.nan]   # Use correct name as per training data
            })

            # Populate the 'moving_average' columns for future months
            for i in range(1, 3):
                future_data['moving_average1'].iloc[i] = future_data['moving_average1'].iloc[i-1]
                future_data['moving_average2'].iloc[i] = future_data['moving_average2'].iloc[i-1]
                future_data['moving_average3'].iloc[i] = future_data['moving_average3'].iloc[i-1]

            # Step 2: Use the trained XGBoost model to predict the next 3 months
            future_predictions = xgb.predict(future_data)

            # Step 3: Display the predictions
            future_months = ['Next Month 1', 'Next Month 2', 'Next Month 3']  # Placeholder names for future months
            future_df = pd.DataFrame({
                'month': future_months,
                'predicted_quantity': future_predictions
            })
  

            # Display the predicted values for each month
            st.write(f"Predicted value for July: {future_df['predicted_quantity'][0]:.2f} Litres")
            st.write(f"Predicted value for August: {future_df['predicted_quantity'][1]:.2f} Litres")
            st.write(f"Predicted value for September: {future_df['predicted_quantity'][2]:.2f} Litres")    

    if category_name_user_input == 'Lentils':
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['LINEAR REGRESSION','ADABoost','XGBoost', 'RANDOM FOREST', 'PREDICTIONS'])
       
        lentils_data = retail_store_data[retail_store_data['category'] == 'Lentils']

        # Extract relevant columns for quantities and amounts
        quantity_columns = [col for col in lentils_data.columns if 'quantity' in col.lower()]
        amount_columns = [col for col in lentils_data.columns if 'amount' in col.lower()]

        # Melt the DataFrame to long format for quantities
        quantities_melted = lentils_data.melt(id_vars=['product_name', 'category'],
                                        value_vars=quantity_columns,
                                        var_name='month',
                                        value_name='quantity')

        # Melt the DataFrame to long format for amounts
        amounts_melted = lentils_data.melt(id_vars=['product_name', 'category'],
                                        value_vars=amount_columns,
                                        var_name='month',
                                        value_name='amount')

        # Remove '_quantity' and '_amount' suffix from the month column
        quantities_melted['month'] = quantities_melted['month'].str.replace('_quantity', '')
        amounts_melted['month'] = amounts_melted['month'].str.replace('_amount', '')

        # Merge quantities and amounts dataframes on product_name, category, and month
        monthly_sales = pd.merge(quantities_melted, amounts_melted, on=['product_name', 'category', 'month'])

        # Filter out rows where both quantity and amount are zero to clean the data
        monthly_sales = monthly_sales[(monthly_sales['quantity'] != 0) | (monthly_sales['amount'] != 0)]

        # Group by month and sum the quantities and amounts
        monthly_sales_summary = monthly_sales.groupby('month').sum().reset_index()

        # Normalize the month names to a consistent format
        def normalize_month(month):
            month_mapping = {
                'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APRIL': 'Apr', 'MAY': 'May', 'JUNE': 'Jun',
                'JULY': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
            }
            parts = month.split('_')
            if len(parts) > 1 and len(parts[1]) == 4:  # Check if year is already present in the correct format
                return month
            if parts[0].upper() in month_mapping:
                return f"{month_mapping[parts[0].upper()]}-{parts[1]}"
            return month

        monthly_sales_summary['month'] = monthly_sales_summary['month'].apply(normalize_month)

        # Create the final DataFrame with only month, total quantity, and total amount
        monthly_lentils_sales_summary = monthly_sales_summary[['month', 'quantity', 'amount']].rename(columns={'quantity': 'total_quantity', 'amount': 'total_amount'})

        monthly_lentils_sales_summary = monthly_lentils_sales_summary[:-1]
        monthly_lentils_sales_summary['month']=pd.to_datetime(monthly_lentils_sales_summary['month'], format='%b-%y')
        monthly_lentils_sales_summary = monthly_lentils_sales_summary.sort_values(by='month')
        monthly_lentils_sales_summary['year'] = monthly_lentils_sales_summary['month'].dt.year
        monthly_lentils_sales_summary['quarter'] = monthly_lentils_sales_summary['month'].dt.quarter
        
        # Detect outliers
        outliers_quantity = detect_outliers(monthly_lentils_sales_summary, 'total_quantity')
        outliers_amount = detect_outliers(monthly_lentils_sales_summary, 'total_amount')

        # Cap outliers
        df_capped_lentils = monthly_lentils_sales_summary.copy()
        cap_outliers(df_capped_lentils, 'total_quantity')
        cap_outliers(df_capped_lentils, 'total_amount')

        with tab1:
            lentil_data =df_capped_lentils.copy()
            # Split data into features (X) and target variable (y)
            X = lentil_data.drop(['month', 'total_quantity'], axis=1)
            y = lentil_data['total_quantity']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)

            # st.subheader("Linear Regression Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,y_pred_lr)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,y_pred_lr),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,y_pred_lr),2)}")

            st.subheader("Linear Regression Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(lentil_data['month'][:len(y)], y, label='Actual', color='blue')
            ax.plot(lentil_data['month'][len(X_train):], y_pred_lr, label='Predicted', color='red', linestyle='--')

            # Setting labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)
            plt.figure(figsize=(10, 6))

        with tab2:
            lentils_data_ada = df_capped_lentils.copy()
            lentils_data_ada['rolling_mean2'] = lentils_data_ada['total_quantity'].rolling(window=2).mean()
            # Create new columns for moving averages of each month
            for i in range(1, 3):
                lentils_data_ada[f'moving_average_{i}'] = lentils_data_ada['total_quantity'].shift(i)
            lentils_data_ada.dropna(inplace=True)
            # Split data into features (X) and target variable (y)
            X = lentils_data_ada.drop(['month', 'total_quantity'], axis=1)
            y = lentils_data_ada['total_quantity']
            adaboost = AdaBoostRegressor()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            adaboost.fit(X_train, y_train)
            adaboost_pred = adaboost.predict(X_test)

            # st.subheader("ADABoost Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,adaboost_pred)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,adaboost_pred),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,adaboost_pred),2)}")
            
            st.subheader("ADABoost Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(lentils_data_ada['month'][:len(y)], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(lentils_data_ada['month'][len(X_train):], adaboost_pred, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)  

        with tab3:
            lentils_data_xgb = df_capped_lentils.copy()
            lentils_data_xgb['rolling_mean2'] = lentils_data_xgb['total_quantity'].rolling(window=2).mean()
            for i in range(1, 2):
                lentils_data_xgb[f'moving_average_{i}'] = lentils_data_xgb['total_quantity'].shift(i)

            lentils_data_xgb.dropna(inplace=True)
            # Define X and y after dropping NaN values
            X = lentils_data_xgb.drop(['month', 'total_quantity'], axis=1)
            y = lentils_data_xgb['total_quantity']

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

            # Fit the XGBoost model
            xgb = XGBRFRegressor()
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)

            # Plotting Actual vs Predicted Graph
            st.subheader("XGBoost Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Adjust the 'month' column to match the length of 'y' after dropna()
            months_for_actual = lentils_data_xgb['month']

            # Plot actual values
            ax.plot(months_for_actual, y, label='Actual', color='blue')

            # Plot predicted values (ensure correct slicing for predictions)
            ax.plot(months_for_actual[len(X_train):], xgb_pred, label='Predicted', color='red', linestyle='--')

            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)

            # Add legend
            ax.legend()

            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            lentils_data_rf = df_capped_lentils.copy()
            lentils_data_rf['rolling_mean2'] = lentils_data_rf['total_quantity'].rolling(window=2).mean()
            for i in range(1, 2):
                lentils_data_rf[f'moving_average_{i}'] = lentils_data_rf['total_quantity'].shift(i)

            lentils_data_rf = lentils_data_rf.dropna()

          
            # Split data into features (X) and target variable (y)
            X = lentils_data_rf.drop(['month', 'total_quantity'], axis=1)
            y = lentils_data_rf['total_quantity']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            # st.subheader("Random Forest Model Parameters")
            # st.warning(f"ROOT MEAN SQUARED ERROR : {round(np.sqrt(mean_squared_error(y_test,y_pred_rf)),2)}")
            # st.info(f"MEAN ABSOLUTE ERROR : {round(mean_absolute_error(y_test,y_pred_rf),2)}")
            # st.error(f"R2 SCORE : {round(r2_score(y_test,y_pred_rf),2)}")

            st.subheader("Random Forest Model Actual Vs Predicted Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot actual values
            ax.plot(lentils_data_rf['month'], y, label='Actual', color='blue')
            # Plot predicted values (ensure the slicing of the predicted data starts from where the training data ends)
            ax.plot(lentils_data_rf['month'][len(X_train):], y_pred_rf, label='Predicted', color='red', linestyle='--')
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Quantity')
            ax.set_title('Actual vs Predicted Values')
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
            # Add legend
            ax.legend()
            # Adjust layout
            plt.tight_layout()
            st.pyplot(fig)  
        with tab5:
            # Get the last known values from lentils_data_xgb
            last_total_amount = lentils_data_xgb['total_amount'].iloc[-1]
            last_year = lentils_data_xgb['year'].iloc[-1]
            last_quarter = lentils_data_xgb['quarter'].iloc[-1]
            last_rolling_mean2 = lentils_data_xgb['rolling_mean2'].iloc[-1]
            last_moving_avg = lentils_data_xgb['moving_average_1'].iloc[-1]

            # Create a DataFrame to hold future data
            future_data = pd.DataFrame({
                'total_amount': [last_total_amount] * 3,  # Assuming future total amount remains the same
                'year': [last_year + 1] * 3,  # Increment the year for future predictions if necessary
                'quarter': [(last_quarter % 4) + 1, ((last_quarter + 1) % 4) + 1, ((last_quarter + 2) % 4) + 1],  # Cycle through quarters
                'rolling_mean2': [0] * 3,  # Use placeholder (0) since future rolling mean cannot be computed
                'moving_average_1': [last_moving_avg, np.nan, np.nan]  # Start with the last known value
            })

            # Populate 'moving_average_1' for future months (similarly to how it was calculated before)
            for i in range(1, 3):
                future_data['moving_average_1'].iloc[i] = future_data['moving_average_1'].iloc[i-1]

            # Step 2: Use the trained XGBoost model to predict the next 3 months
            future_predictions = xgb.predict(future_data)

            # Step 3: Print or plot the predictions
            future_months = ['Month 1', 'Month 2', 'Month 3']  # Placeholder names for future months
            future_df = pd.DataFrame({
                'month': future_months,
                'predicted_quantity': future_predictions
            })

            # Display the predicted values for each month
            st.write(f"Predicted value for July: {future_df['predicted_quantity'][0]:.2f} KGS")
            st.write(f"Predicted value for August: {future_df['predicted_quantity'][1]:.2f} KGS")
            st.write(f"Predicted value for September: {future_df['predicted_quantity'][2]:.2f} KGS")







