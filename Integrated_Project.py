import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import PyPDF2
import os
import joblib
from tabula.io import read_pdf
from datetime import datetime
import calendar
import camelot

# 🎨 Apply Background Image
def add_bg_from_url():
    st.set_page_config(layout="wide")
    st.markdown(
    """
    <style>
    .main > div {
        max-width: 2200%;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    .stApp {
        margin: 0 auto;
        max-width: 200%;
    }
        .stApp {
            background: url("https://source.unsplash.com/1600x900/?finance,stocks,money") no-repeat center fixed;
            background-size: cover;
        }
        .css-18e3th9 {
            background-color: rgba(0, 0, 0, 0.5) !important;  /* Darken background */
        }
        .stTitle {
            color: #f4d03f; /* Gold */
            font-weight: bold;
            text-align: center;
            font-size: 2rem;
        }
        .stSidebar {
            background-color: light !important;
        }
        .stButton > button {
            background-color: #f4d03f !important; /* Gold */
            color: black !important;
            font-weight: bold;
            border-radius: 10px;
        }
        .stSelectbox, .stNumberInput {
            background-color: rgba(255, 255, 255, 0.8) !important;
        }
        .stDataEditor {
            border: 1px solid #f4d03f !important;
            border-radius: 10px;
        }
        .privacy-notice {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            width: 400px;
            text-align: center;
        }
        .privacy-notice button {
            background-color: #f4d03f;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# Privacy Notice Pop-up
def privacy_notice():
    st.markdown(
        """
        <div class="privacy-notice">
            <h3>Privacy & Data Protection Notice</h3>
            <p>We value your privacy. Our software does not collect, store, or share your personal information with any third-party entity. Any data processed during your use of this application remains solely within your control and is not retained on our systems.</p>
            <p>By proceeding, you acknowledge and agree to these terms.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add an "OK" button to dismiss the notice
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("OK"):
            st.session_state.privacy_accepted = True
            st.rerun()  # Rerun the app to hide the notice

# Initialize session state for privacy notice
if 'privacy_accepted' not in st.session_state:
    st.session_state.privacy_accepted = False

# Show Privacy Notice if not accepted
if not st.session_state.privacy_accepted:
    privacy_notice()
else:
    # Main app content
    st.markdown('<h1 class="stTitle">💰 AI-Powered Personal Finance Tool </h1>', unsafe_allow_html=True)


# Function to unlock PDF
def unlock_pdf(input_pdf_path, output_pdf_path):
    try:
        with open(input_pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.is_encrypted:
                password = st.text_input("Enter PDF password:", type="password")
                if pdf_reader.decrypt(password):
                    pdf_writer = PyPDF2.PdfWriter()
                    for page in pdf_reader.pages:
                        pdf_writer.add_page(page)
                    with open(output_pdf_path, 'wb') as output_file:
                        pdf_writer.write(output_file)
                    st.success("PDF successfully unlocked!")
                    return output_pdf_path
                else:
                    st.error("Incorrect password.")
                    return None
            else:
                return input_pdf_path
    except Exception as e:
        st.error(f"Error unlocking PDF: {e}")
        return None

# Function to extract tables from PDF
def extract_pdf_table(pdf_path):
    try:
        #tables = read_pdf(pdf_path, pages='all', multiple_tables=True, lattice=True)
        tables = camelot.read_pdf(pdf_path, pages="all")    
        if tables:
            st.success("Table extracted successfully!")
            df = pd.concat([table.df for table in tables], ignore_index=True)
            return df
        else:
            st.error("No table found in the PDF.")
            return None
    except Exception as e:
        st.error(f"Error extracting table: {e}")
        return None

# Function to read Excel data
def read_excel(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

# Function to clean text from Excel
def clean_excel_data(df):
    try:
        df.replace('0',np.nan,inplace=True)
        df.replace(r"\*+",np.nan,regex=True,inplace=True)
        df.dropna(thresh=np.ceil(len(df.columns)*0.75),axis=0,inplace=True)
        words_to_filter = ["Generated", "Branch",'Opening']
        mask = ~df.apply(lambda row: row.astype(str).str.contains('|'.join(words_to_filter), case=False).any(), axis=1)
        df_filtered=df[mask]
        df_filtered.reset_index(drop=True,inplace=True)
        df_filtered.columns=df_filtered.loc[0,:]
        df_filtered=df_filtered.drop(0,axis=0)
        return df_filtered
    except Exception as e:
        st.error(f"Error cleaning Excel file: {e}")
        return None   

def classify_transactions(dataframe, model_path):
    try:
        loaded_model = joblib.load(model_path)
        if 'Transaction Description' in dataframe.columns:
            descriptions = dataframe['Transaction Description']
            predictions = loaded_model.predict(descriptions)
            dataframe['Predicted Category'] = predictions
            if 'Credit' in dataframe.columns:
                dataframe.loc[dataframe['Credit'] > 0, 'Predicted Category'] = 'Income'
            return dataframe[['Transaction Description','Predicted Category']]
        else:
            st.error("Error: 'Transaction Description' column missing in dataframe.")
            return dataframe
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return dataframe
    except Exception as e:
        st.error(f"An error occurred during transaction classification: {e}")
        return dataframe

# Function to apply budget rule
def apply_budget_rule(dataframe, selected_month, selected_year, income):
    try:
        dataframe = dataframe[((dataframe['Transaction Month'] == selected_month) & (dataframe['Transaction Year'] == selected_year))]
        dataframe['Credit'] = pd.to_numeric(dataframe['Credit'], errors='coerce')
        dataframe['Debit'] = pd.to_numeric(dataframe['Debit'], errors='coerce')

        if income<=0:
            income_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'income']['Credit'].sum()
        else:
            income_total = income

        needs_percent = 50
        wants_percent = 30
        savings_debts_percent = 20

        needs_limit = (needs_percent / 100) * income_total
        wants_limit = (wants_percent / 100) * income_total
        savings_debts_limit = (savings_debts_percent / 100) * income_total

        needs_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'needs']['Debit'].sum()
        wants_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'wants']['Debit'].sum()
        savings_debts_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'savings']['Debit'].sum()

        st.write(f"Income: ₹{income_total}")
        st.write(f"\nExpense Breakdown for {selected_month},{selected_year}:")
        st.write(f"Needs: ₹{needs_total:.2f} / ₹{needs_limit:.2f}")
        st.write(f"Wants: ₹{wants_total:.2f} / ₹{wants_limit:.2f}")
        st.write(f"Savings & Debts: ₹{savings_debts_total:.2f} / ₹{savings_debts_limit:.2f}")

        if needs_total > needs_limit:
            st.warning(f"Warning: Your 'Needs' expenses have exceeded the limit by ₹{needs_total - needs_limit:.2f}")
        if wants_total > wants_limit:
            st.warning(f"Warning: Your 'Wants' expenses have exceeded the limit by ₹{wants_total - wants_limit:.2f}")
        if savings_debts_total > savings_debts_limit:
            st.warning(f"Warning: Your 'Savings & Debts' have exceeded the limit by ₹{savings_debts_total - savings_debts_limit:.2f}")
    except Exception as e:
        st.error(f"An error occurred during budget rule application: {e}")

def apply_budget_rule2(dataframe, income):
    try:
        dataframe['Credit'] = pd.to_numeric(dataframe['Credit'], errors='coerce')
        dataframe['Debit'] = pd.to_numeric(dataframe['Debit'], errors='coerce')

        if income<=0:
            income_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'income']['Credit'].sum()
        else:
            income_total = income

        needs_percent = 50
        wants_percent = 30
        savings_debts_percent = 20

        needs_limit = (needs_percent / 100) * income_total
        wants_limit = (wants_percent / 100) * income_total
        savings_debts_limit = (savings_debts_percent / 100) * income_total

        needs_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'needs']['Debit'].sum()
        wants_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'wants']['Debit'].sum()
        savings_debts_total = dataframe[dataframe['Reviewed Category'].str.lower() == 'savings']['Debit'].sum()

        st.write(f"Income: ₹{income_total}")
        st.write(f"Needs: ₹{needs_total:.2f} / ₹{needs_limit:.2f}")
        st.write(f"Wants: ₹{wants_total:.2f} / ₹{wants_limit:.2f}")
        st.write(f"Savings: ₹{savings_debts_total:.2f} / ₹{savings_debts_limit:.2f}")

        # Check for exceeded limits
        if needs_total > needs_limit:
            st.warning(f"Warning: Your 'Needs' expenses have exceeded the limit by ₹{needs_total - needs_limit:.2f}",icon= '⚠')
        if wants_total > wants_limit:
            st.warning(f"Warning: Your 'Wants' expenses have exceeded the limit by ₹{wants_total - wants_limit:.2f}",icon= '⚠')
        if savings_debts_total > savings_debts_limit:
            st.balloons()
            st.success(f"Congratulations: Your 'Savings' have exceeded the limit by ₹{savings_debts_total - savings_debts_limit:.2f}",icon= '🎉')
    except Exception as e:
        st.error(f"An error occurred during budget rule application: {e}")

# App Title
#st.title("AI-Powered Personal Finance Tool")
st.sidebar.header("📂 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Upload an Excel", type=["xls","xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "xlsx" or file_type == "xls":
        df = read_excel(uploaded_file)
        df_cleaned = clean_excel_data(df)
        if df is not None:
            ##Invoke model with df_cleaned
            model_path = "trained_model_balanced.joblib"
            if not os.path.exists(model_path):
                st.error("Error: Trained model not found.")

            renaming_rules = {
                "Transaction Description": ["Narration", "Description"],
                "Credit": ["Credit", "Deposit"],
                "Debit": ["Debit", "Withdrawal"]
            }

            # Create renaming dictionary in a single loop
            rename_dict = {
                col: new_name
                for col in df_cleaned.columns
                for new_name, keywords in renaming_rules.items()
                if any(keyword in col for keyword in keywords)
            }

            # Apply renaming
            df_cleaned = df_cleaned.rename(columns=rename_dict)

            df_predicted = classify_transactions(df_cleaned[['Transaction Description','Credit']], model_path)
            df_cleaned=df_cleaned.merge(df_predicted, on="Transaction Description", how="left")
            df_cleaned['Reviewed Category']=df_cleaned['Predicted Category'] ##will be equated with Predicted Category column.
            #creating month and year columns from Date:
            #Getting first column containing 'Date':
            date_col = next((col for col in df_cleaned.columns if 'Date' in col), None)

            if date_col:
                # Convert to datetime format
                df_cleaned["Transaction Year"] = pd.to_datetime(df_cleaned[date_col], format="%d/%m/%y").dt.year
                df_cleaned["Transaction Month"] = pd.to_datetime(df_cleaned[date_col], format="%d/%m/%y").dt.strftime("%B")

            #Code to let user change category:
            editable_column = "Reviewed Category"
            category_options = ["Savings", "Wants", "Needs"]

            # Dynamically configure columns: All read-only except the editable one
            column_config = {col: st.column_config.TextColumn(col, disabled=True) for col in df_cleaned.columns}
            column_config[editable_column] = st.column_config.SelectboxColumn(editable_column, options=category_options)
            

            st.subheader("Uploaded Data")
            #columns to hide:
            columns_to_hide = ["Transaction Year", "Transaction Month",'Closing Balance','Value Dt','Chq./Ref.No.']
            # Display DataFrame with only one editable dropdown column
            edited_df = st.data_editor(df_cleaned.drop(columns=columns_to_hide), column_config=column_config, num_rows="fixed",use_container_width=True, height=500)
            #st.dataframe(edited_df.style.set_properties(**{"background-color": "gray", "color": "black"}))
            for col in columns_to_hide:
                edited_df[col] = df_cleaned[col]

            # Budget Rule Application
            st.header("Budget Rule Application")
            #months = list(calendar.month_name)[1:]
            #selected_month = st.sidebar.selectbox("Select a Month", months)
            #selected_year = st.sidebar.number_input("Select Year:", value=2024)
            #income = st.sidebar.number_input("Enter Monthly Income (optional):", value=0)

            #if st.sidebar.button("Apply Budget Rule"):
                #apply_budget_rule(df_cleaned, selected_month, selected_year, income)

            ##Budget Application Code 2:
            # Month and Year selection
            months = list(calendar.month_name)[1:]  # ['January', 'February', ..., 'December']
            years = sorted(edited_df["Transaction Year"].unique())  # Get unique years in sorted order

            # Select start and end month-year range
            col1, col2 = st.columns(2)
            with col1:
                start_month = st.selectbox("Select Start Month", months, index=0)
                start_year = st.selectbox("Select Start Year", years, index=0)

            with col2:
                end_month = st.selectbox("Select End Month", months, index=len(months) - 1)
                end_year = st.selectbox("Select End Year", years, index=len(years) - 1)
            income2 = st.number_input("Enter Income for the selected period (optional):", value=0)
            month_to_num = {month: i for i, month in enumerate(calendar.month_name) if month}
            start_month_num = month_to_num[start_month]
            end_month_num = month_to_num[end_month]
            year_month_df = edited_df[
                    ((edited_df["Transaction Year"] > start_year) | ((edited_df["Transaction Year"] == start_year) & (edited_df["Transaction Month"].map(month_to_num) >= start_month_num))) &
                    ((edited_df["Transaction Year"] < end_year) | ((edited_df["Transaction Year"] == end_year) & (edited_df["Transaction Month"].map(month_to_num) <= end_month_num)))
                ]

            if st.button("Apply Budgeting Rule"):
                #st.write(year_month_df)
                st.write(f"Expense Breakdown for **{start_month} {start_year}** to **{end_month} {end_year}**")
                apply_budget_rule2(year_month_df, income2)

            year_month_df['Credit'] = pd.to_numeric(year_month_df['Credit'], errors='coerce')
            year_month_df['Debit'] = pd.to_numeric(year_month_df['Debit'], errors='coerce')
            st.header("🔍 Visualizations:")
            col1, col2 = st.columns([1, 2]) 
            with col1:
                # Select column to group by
                st.write('\n')
                st.write('\n')
                group_column = st.selectbox("Select Grouping Column", year_month_df.columns)

                # Select column to aggregate
                agg_column = st.selectbox("Select Aggregation Column", year_month_df.select_dtypes(include=['number']).columns)

                # Select aggregation function
                agg_function = st.radio("Select Aggregation", ["Sum", "Count"])

                # Select chart type
                chart_type = st.selectbox("📊 Select Chart Type", ["Bar Chart", "Pie Chart", "Line Chart"])
            with col2:
                # 🎯 Perform Aggregation
                if agg_function == "Sum":
                    df_agg = year_month_df.groupby(group_column, as_index=False)[agg_column].sum()
                else:  # Count
                    df_agg = year_month_df.groupby(group_column, as_index=False)[agg_column].count()

                # 🎯 Create Visualization based on selected chart type
                if chart_type == "Bar Chart":
                    fig = px.bar(df_agg, x=group_column, y=agg_column, text_auto=True, title=f"{agg_function} of {agg_column} by {group_column}")
                elif chart_type == "Pie Chart":
                    fig = px.pie(df_agg, names=group_column, values=agg_column, title=f"{agg_function} of {agg_column} by {group_column}")
                else:  # Line Chart
                    fig = px.line(df_agg, x=group_column, y=agg_column, markers=True, title=f"{agg_function} of {agg_column} by {group_column}")

                # 🎯 Display Outputs
                st.plotly_chart(fig)

    elif file_type == "pdf":
        #unlocked_pdf_path = "unlocked_statement.pdf"
        #unlocked_pdf = unlock_pdf(uploaded_file.name, unlocked_pdf_path)
        #if unlocked_pdf:
            #df = extract_pdf_table(unlocked_pdf)
        #st.write(df)
        st.warning("🔔 PDF processing is under development.")
else:
    st.info("📢 Please upload a file to get started!")    
