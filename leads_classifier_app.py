import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # Import Streamlit

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

# --- Configuration Parameters ---
NUM_LEADS = 500  # Increased for better model training
CONVERSION_RATE_BASE = 0.05  # Base conversion rate (5%)
DATE_RANGE_DAYS = 365  # Data spanning 1 year
SEED = 42  # For reproducibility

# Initialize Faker for realistic-looking data
fake = Faker()
Faker.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def generate_lead_id(num_leads):
    """Generates unique lead IDs."""
    return [f"LID-{i:05d}" for i in range(num_leads)]

def generate_lead_creation_dates(num_leads, date_range_days):
    """Generates lead creation dates within a specified range."""
    start_date = datetime.now() - timedelta(days=date_range_days)
    return [start_date + timedelta(days=random.randint(0, date_range_days)) for _ in range(num_leads)]

def generate_categorical_feature(num_leads, options, weights=None):
    """Generates categorical data based on provided options and optional weights."""
    return random.choices(options, weights=weights, k=num_leads)

def generate_numerical_feature(num_leads, distribution_type='uniform', **kwargs):
    """
    Generates numerical data based on specified distribution.
    'uniform': low, high
    'normal': loc, scale
    'poisson': lam
    """
    if distribution_type == 'uniform':
        return np.random.randint(kwargs.get('low', 0), kwargs.get('high', 100), num_leads).astype(float)
    elif distribution_type == 'normal':
        return np.random.normal(loc=kwargs.get('loc', 0), scale=kwargs.get('scale', 1), size=num_leads).astype(float)
    elif distribution_type == 'poisson':
        return np.random.poisson(lam=kwargs.get('lam', 1), size=num_leads).astype(float)
    else:
        raise ValueError("Unsupported distribution type")

def generate_conversion_status_and_dates(df, base_rate):
    """
    Determines conversion status based on feature rules and generates conversion/last activity dates.
    Embeds logical relationships to make the data predictive.
    """
    df['conversion_prob'] = base_rate

    # Adjust base rate by industry
    industry_conversion_boost = {
        'Home Services - HVAC': 0.03,
        'Home Services - Plumbing': 0.02,
        'Home Services - Landscaping': 0.01,
        'B2B Software': 0.07,
        'Financial Advisory': 0.06
    }
    for industry, boost in industry_conversion_boost.items():
        df.loc[df['industry'] == industry, 'conversion_prob'] += boost

    # Rule 1: High-intent Lead Sources (e.g., Referrals, Organic)
    df.loc[df['lead_source'].isin(['Referral', 'Organic Search', 'Website Form']), 'conversion_prob'] += 0.15
    # Rule 2: High-intent Lead Types (e.g., SQL, MQL)
    df.loc[df['lead_type'].isin(['Sales Qualified Lead (SQL)', 'Marketing Qualified Lead (MQL)']), 'conversion_prob'] += 0.20
    # Rule 3: Faster Initial Response Time (lower is better for conversion)
    df.loc[df['initial_response_time_hours'] <= 6, 'conversion_prob'] += 0.10
    df.loc[df['initial_response_time_hours'] > 48, 'conversion_prob'] -= 0.08
    # Rule 4: More Interactions (up to a point)
    df.loc[df['num_interactions'] >= 3, 'conversion_prob'] += 0.07
    df.loc[df['num_interactions'] >= 5, 'conversion_prob'] += 0.05
    # Rule 5: Higher Engagement Score and Website Visits
    df.loc[df['engagement_score_implicit'] >= 7, 'conversion_prob'] += 0.12
    df.loc[df['num_website_visits'] >= 5, 'conversion_prob'] += 0.08
    df.loc[df['clicked_marketing_email'] == True, 'conversion_prob'] += 0.06
    # Rule 6: Higher Deal Value Estimate
    df.loc[df['deal_value_estimate'] >= 5000, 'conversion_prob'] += 0.08

    # Cap probabilities
    df['conversion_prob'] = df['conversion_prob'].clip(0.01, 0.99)
    df['conversion_status'] = (np.random.rand(len(df)) < df['conversion_prob']).astype(int)

    df['conversion_date'] = pd.NaT
    df['last_activity_date'] = pd.NaT

    for idx, row in df.iterrows():
        if row['conversion_status'] == 1:
            days_to_convert = random.randint(1, 60)
            df.at[idx, 'conversion_date'] = row['lead_creation_date'] + timedelta(days=days_to_convert)
            df.at[idx, 'last_activity_date'] = df.at[idx, 'conversion_date']
        else:
            days_since_creation = (datetime.now() - row['lead_creation_date']).days
            if days_since_creation > 0:
                df.at[idx, 'last_activity_date'] = row['lead_creation_date'] + timedelta(days=random.randint(0, days_since_creation))
            else:
                df.at[idx, 'last_activity_date'] = row['lead_creation_date']
    return df.drop(columns=['conversion_prob'])

def generate_lead_names(num_leads):
    return [fake.name() for _ in range(num_leads)]

def generate_lead_emails(num_leads):
    return [fake.email() for _ in range(num_leads)]

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='lead_creation_date'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col])

        X_copy['lead_age_days'] = (pd.to_datetime(datetime.now()) - X_copy[self.date_col]).dt.days
        X_copy['lead_age_days'] = X_copy['lead_age_days'].fillna(0).astype(int)

        X_copy['day_of_week_creation'] = X_copy[self.date_col].dt.dayofweek
        X_copy['hour_of_day_creation'] = X_copy[self.date_col].dt.hour
        return X_copy[['lead_age_days', 'day_of_week_creation', 'hour_of_day_creation']]

def get_feature_names(column_transformer):
    output_features = []
    for name, preprocessor_pipeline, features in column_transformer.transformers_:
        if name == 'num':
            output_features.extend(features)
        elif name == 'cat':
            if hasattr(preprocessor_pipeline, 'named_steps') and 'onehot' in preprocessor_pipeline.named_steps:
                onehot_encoder = preprocessor_pipeline.named_steps['onehot']
                feature_names = onehot_encoder.get_feature_names_out(features)
                output_features.extend(feature_names)
            else:
                output_features.extend(preprocessor_pipeline.get_feature_names_out(features))
        elif name == 'date':
            output_features.extend(['lead_age_days', 'day_of_week_creation', 'hour_of_day_creation'])
    return output_features

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Lead Conversion Predictor MVP")

# --- Data Generation and Model Training (Cached for Speed) ---
@st.cache_resource
def load_and_train_model():
    """Generates data, trains, and tunes models. Cached for Streamlit efficiency."""
    # Data Generation
    lead_ids = generate_lead_id(NUM_LEADS)
    lead_creation_dates = generate_lead_creation_dates(NUM_LEADS, DATE_RANGE_DAYS)
    lead_names = generate_lead_names(NUM_LEADS)
    lead_emails = generate_lead_emails(NUM_LEADS)

    lead_sources_options = ['Website Form', 'Google Ads', 'Facebook Ads', 'Referral', 'Cold Call', 'LinkedIn', 'Trade Show', 'Organic Search', 'Email Campaign']
    lead_sources_weights = [0.25, 0.15, 0.15, 0.10, 0.05, 0.10, 0.05, 0.10, 0.05]
    lead_sources = generate_categorical_feature(NUM_LEADS, lead_sources_options, lead_sources_weights)

    initial_contact_methods_options = ['Email', 'Phone Call', 'Website Chat', 'In-Person Meeting']
    initial_contact_methods_weights = [0.4, 0.3, 0.2, 0.1]
    initial_contact_methods = generate_categorical_feature(NUM_LEADS, initial_contact_methods_options, initial_contact_methods_weights)

    lead_types_options = ['Inquiry', 'Marketing Qualified Lead (MQL)', 'Sales Qualified Lead (SQL)', 'Cold Prospect', 'Existing Customer Inquiry']
    lead_types_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    lead_types = generate_categorical_feature(NUM_LEADS, lead_types_options, lead_types_weights)

    initial_response_time_hours = np.round(np.random.exponential(scale=10, size=NUM_LEADS)).astype(int)
    initial_response_time_hours = np.maximum(1, initial_response_time_hours)
    initial_response_time_hours[initial_response_time_hours > 96] = 96

    num_interactions = generate_numerical_feature(NUM_LEADS, distribution_type='poisson', lam=2) + 1
    num_website_visits = generate_numerical_feature(NUM_LEADS, distribution_type='poisson', lam=3)
    clicked_marketing_email = np.random.choice([True, False], size=NUM_LEADS, p=[0.2, 0.8])

    geographic_regions_options = ['Northeast', 'Southeast', 'Northwest', 'Southwest']
    geographic_regions = generate_categorical_feature(NUM_LEADS, geographic_regions_options)

    industry_options = ['Home Services - HVAC', 'Home Services - Plumbing', 'Home Services - Landscaping', 'B2B Software', 'Financial Advisory']
    industry_weights = [0.25, 0.2, 0.15, 0.2, 0.2]
    industry = generate_categorical_feature(NUM_LEADS, industry_options, industry_weights)

    engagement_score_implicit = generate_numerical_feature(NUM_LEADS, distribution_type='uniform', low=1, high=11).astype(float)
    engagement_score_implicit[np.random.rand(NUM_LEADS) < 0.05] = np.nan

    deal_value_estimate = np.round(np.random.lognormal(mean=7, sigma=1.2, size=NUM_LEADS)).astype(float)
    deal_value_estimate[deal_value_estimate < 100] = 100
    deal_value_estimate[deal_value_estimate > 50000] = 50000
    deal_value_estimate[np.random.rand(NUM_LEADS) < 0.03] = np.nan

    data = {
        'lead_id': lead_ids,
        'lead_name': lead_names,
        'lead_email': lead_emails,
        'lead_creation_date': lead_creation_dates,
        'lead_source': lead_sources,
        'initial_contact_method': initial_contact_methods,
        'lead_type': lead_types,
        'initial_response_time_hours': initial_response_time_hours,
        'num_interactions': num_interactions,
        'geographic_region': geographic_regions,
        'industry': industry,
        'engagement_score_implicit': engagement_score_implicit,
        'deal_value_estimate': deal_value_estimate,
        'num_website_visits': num_website_visits,
        'clicked_marketing_email': clicked_marketing_email,
    }
    df = pd.DataFrame(data)
    df = generate_conversion_status_and_dates(df.copy(), CONVERSION_RATE_BASE)

    X = df.drop(columns=['lead_id', 'lead_name', 'lead_email', 'conversion_status', 'conversion_date', 'last_activity_date'])
    y = df['conversion_status']

    # Identify features
    numerical_features = ['initial_response_time_hours', 'num_interactions', 'engagement_score_implicit', 'deal_value_estimate', 'num_website_visits']
    categorical_features = ['lead_source', 'initial_contact_method', 'lead_type', 'geographic_region', 'industry', 'clicked_marketing_email']
    date_features = ['lead_creation_date']

    # Preprocessing pipelines
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    date_transformer = Pipeline(steps=[('date_extractor', DateFeatureExtractor(date_col='lead_creation_date')), ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', date_transformer, date_features)
        ],
        remainder='drop'
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # Model Pipelines
    logistic_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(solver='liblinear', random_state=SEED, class_weight='balanced'))])
    random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=SEED, class_weight='balanced'))])

    # Cross-Validation Strategy
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Hyperparameter Tuning
    param_grid_lr = {'classifier__C': [0.1, 1.0, 10.0], 'classifier__solver': ['liblinear']}
    grid_search_lr = GridSearchCV(logistic_pipeline, param_grid_lr, cv=cv_strategy, scoring='f1', n_jobs=-1)
    grid_search_lr.fit(X_train, y_train)
    best_lr_model = grid_search_lr.best_estimator_

    param_grid_rf = {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [10, 20, None], 'classifier__min_samples_leaf': [1, 5], 
                     'classifier__max_features': ['sqrt', 'log2', None]}
    grid_search_rf = GridSearchCV(random_forest_pipeline, param_grid_rf, cv=cv_strategy, scoring='f1', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_

    # For Summary Dashboard calculations
    y_prob_rf_test = best_rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf_test = (y_prob_rf_test >= 0.5).astype(int) # Default threshold for classification report

    # Get the fitted preprocessor from one of the trained models.
    # The preprocessor within best_rf_model has been fitted during GridSearchCV.
    fitted_preprocessor_from_rf_model = best_rf_model.named_steps['preprocessor']
    return best_lr_model, best_rf_model, X_test, y_test, y_prob_rf_test, y_pred_rf_test, df, get_feature_names(fitted_preprocessor_from_rf_model)

# Load and train the model (this will run only once due to st.cache_resource)
best_lr_model, best_rf_model, X_test, y_test, y_prob_rf_test, y_pred_rf_test, full_df, feature_names = load_and_train_model()

def get_priority(prob):
        if prob >= HIGH_PRIORITY_THRESHOLD:
            return 'High Priority'
        elif prob >= MEDIUM_PRIORITY_THRESHOLD:
            return 'Medium Priority'
        else:
            return 'Low Priority'

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Interactive Demo", "Summary Dashboard"])

# Define conversion priority thresholds
HIGH_PRIORITY_THRESHOLD = 0.65 # Adjust these based on your F1/ROC curves
MEDIUM_PRIORITY_THRESHOLD = 0.35

if app_mode == "Interactive Demo":
    st.header("Lead Conversion Probability Predictor")
    st.markdown("""
    You can predict for a single lead by manually entering its details, or upload a CSV file to get batch predictions for multiple leads.
    """)

    st.markdown("---") # Separator for visual clarity

    # --- Predict Single Lead Section ---
    st.markdown("### Predict Single Lead")
    st.write("Enter details for one lead below:")

    col1, col2 = st.columns(2)

    with col1:
        # Added unique keys to widgets
        lead_creation_date = st.date_input("Lead Creation Date", datetime.now().date(), key='single_creation_date')
        lead_source = st.selectbox("Lead Source", ['Website Form', 'Google Ads', 'Facebook Ads', 'Referral', 'Cold Call', 'LinkedIn', 'Trade Show', 'Organic Search', 'Email Campaign'], key='single_lead_source')
        initial_contact_method = st.selectbox("Initial Contact Method", ['Email', 'Phone Call', 'Website Chat', 'In-Person Meeting'], key='single_initial_contact_method')
        lead_type = st.selectbox("Lead Type", ['Inquiry', 'Marketing Qualified Lead (MQL)', 'Sales Qualified Lead (SQL)', 'Cold Prospect', 'Existing Customer Inquiry'], key='single_lead_type')
        geographic_region = st.selectbox("Geographic Region", ['Northeast', 'Southeast', 'Northwest', 'Southwest'], key='single_geo_region')
        industry = st.selectbox("Industry", ['Home Services - HVAC', 'Home Services - Plumbing', 'Home Services - Landscaping', 'B2B Software', 'Financial Advisory'], key='single_industry')

    with col2:
        # Added unique keys to widgets
        initial_response_time_hours = st.number_input("Initial Response Time (hours)", min_value=1, max_value=100, value=6, key='single_response_time')
        num_interactions = st.number_input("Number of Interactions", min_value=0, max_value=20, value=2, key='single_num_interactions')
        num_website_visits = st.number_input("Number of Website Visits", min_value=0, max_value=50, value=3, key='single_num_visits')
        engagement_score_implicit = st.slider("Engagement Score (1-10)", min_value=1, max_value=10, value=5, key='single_engagement_score')
        deal_value_estimate = st.number_input("Estimated Deal Value ($)", min_value=100, max_value=50000, value=1000, key='single_deal_value')
        clicked_marketing_email = st.checkbox("Clicked Marketing Email?", value=False, key='single_clicked_email')

    predict_button = st.button("Predict Single Lead")

    if predict_button:
        # Create a DataFrame from the input data
        input_data = pd.DataFrame({
            'lead_creation_date': [pd.Timestamp(lead_creation_date)],
            'lead_source': [lead_source],
            'initial_contact_method': [initial_contact_method],
            'lead_type': [lead_type],
            'initial_response_time_hours': [float(initial_response_time_hours)],
            'num_interactions': [float(num_interactions)],
            'geographic_region': [geographic_region],
            'industry': [industry],
            'engagement_score_implicit': [float(engagement_score_implicit)],
            'deal_value_estimate': [float(deal_value_estimate)],
            'num_website_visits': [float(num_website_visits)],
            'clicked_marketing_email': [bool(clicked_marketing_email)]
        })

        # Make prediction using the best Random Forest model
        predicted_prob = best_rf_model.predict_proba(input_data)[:, 1][0]

        # Determine priority
        priority = get_priority(predicted_prob) # Use the global get_priority function
        priority_emoji = "🚀" if priority == "High Priority" else ("💡" if priority == "Medium Priority" else "🐢")
        priority_color = "green" if priority == "High Priority" else ("orange" if priority == "Medium Priority" else "red")

        st.subheader("Prediction Results:")
        st.metric(label="Predicted Conversion Probability", value=f"{predicted_prob:.2%}")
        st.markdown(f"**Conversion Priority: <span style='color:{priority_color}'>{priority} {priority_emoji}</span>**", unsafe_allow_html=True)

        # --- Explain "Why" with Feature Importance ---
        st.subheader("Why This Prediction?")
        st.write("These are the most influential factors for this lead's conversion probability:")

        rf_feature_importance = best_rf_model.named_steps['classifier'].feature_importances_
        rf_feature_importance_series = pd.Series(rf_feature_importance, index=feature_names)
        rf_feature_importance_series = rf_feature_importance_series.sort_values(ascending=False) # Ensure sorting

        most_important_features = rf_feature_importance_series.head(10) # Display top 10 as per your original code
        
        fig_feat_imp, ax_feat_imp = plt.subplots(figsize=(8, 4), dpi=200)
        sns.barplot(x=most_important_features.values, y=most_important_features.index, palette='viridis', ax=ax_feat_imp)
        ax_feat_imp.set_title('Top Influential Features (Overall Model Importance)')
        ax_feat_imp.set_xlabel('Importance')
        ax_feat_imp.set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig_feat_imp)
        st.caption("Note: This shows the overall importance of features to the model, not necessarily their unique influence on *this specific* prediction. For precise individual 'why', techniques like SHAP are used.")

    st.markdown("---") # Visual separator between single and batch
    
    # --- Predict Multiple Leads (CSV Upload) Section ---
    st.markdown("### Predict Multiple Leads (CSV Upload)")
    st.write("Upload a CSV file containing multiple lead inputs to get batch predictions.")
    st.info("💡 **Tip:** Ensure your CSV has columns named exactly like the input features (e.g., 'lead_creation_date', 'lead_source', 'deal_value_estimate', etc.).")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Leads Preview:")
            st.dataframe(batch_df.head())

            required_cols = [
                'lead_creation_date', 'lead_source', 'initial_contact_method', 'lead_type',
                'initial_response_time_hours', 'num_interactions', 'geographic_region',
                'industry', 'engagement_score_implicit', 'deal_value_estimate',
                'num_website_visits', 'clicked_marketing_email'
            ]
            
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Error: Missing required columns in CSV: {', '.join(missing_cols)}. Please ensure all expected input features are present.")
            else:
                # Convert clicked_marketing_email to boolean for robustness
                if 'clicked_marketing_email' in batch_df.columns:
                    batch_df['clicked_marketing_email'] = batch_df['clicked_marketing_email'].astype(str).str.lower().isin(['true', '1', 'yes'])
                
                # Convert date column to datetime objects
                if 'lead_creation_date' in batch_df.columns:
                    batch_df['lead_creation_date'] = pd.to_datetime(batch_df['lead_creation_date'], errors='coerce')
                    if batch_df['lead_creation_date'].isnull().any():
                        st.warning("Some 'lead_creation_date' values could not be parsed and might result in NaNs during prediction.")

                st.write("Predicting conversions for uploaded leads...")
                
                # Make predictions using the best Random Forest model
                batch_probs = best_rf_model.predict_proba(batch_df)[:, 1]
                
                # Add predictions and priority to the DataFrame
                batch_df['predicted_conversion_prob'] = batch_probs
                batch_df['priority'] = batch_df['predicted_conversion_prob'].apply(get_priority) # Use the global get_priority function

                st.write("Predictions Complete:")
                # Display relevant columns for batch output
                # Assuming 'lead_id' might not always be in the uploaded CSV, but if it is, it's good to show.
                display_cols = ['predicted_conversion_prob', 'priority']
                if 'lead_id' in batch_df.columns:
                    display_cols.insert(0, 'lead_id')
                if 'lead_source' in batch_df.columns: # Add other key identifying columns
                    display_cols.insert(1, 'lead_source')
                if 'deal_value_estimate' in batch_df.columns:
                    display_cols.insert(2, 'deal_value_estimate')

                st.dataframe(batch_df[display_cols].head(10)) 

                # Option to download full results
                csv_output = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Predictions CSV",
                    data=csv_output,
                    file_name="lead_predictions_batch.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"An error occurred while processing the CSV: {e}")
            st.warning("Please ensure the CSV format is correct and columns match expectations. Common issues: incorrect column names, or invalid data types for 'clicked_marketing_email' (should be True/False, 0/1) or 'lead_creation_date' (e.g., 'YYYY-MM-DD').")

elif app_mode == "Summary Dashboard":
    st.header("📊 Lead Conversion Model Performance & Business Impact")
    st.markdown("""
    This dashboard summarizes the performance of the lead conversion prediction model on unseen data
    and illustrates its potential business impact.
    """)

    # --- Overall Conversion Rate ---
    overall_conversion_rate = y_test.mean()
    st.subheader("Overall Conversion Rate")
    st.metric(label="Overall Conversion Rate", value=f"{overall_conversion_rate:.2%}")

    # --- Model Performance on Test Set ---
    st.subheader("Model Performance (Random Forest on Test Data)")

    col_perf1, col_perf2 = st.columns(2)
    with col_perf1:
        st.metric(label="ROC AUC Score", value=f"{roc_auc_score(y_test, y_prob_rf_test):.2f}")
    with col_perf2:
        st.metric(label="F1-Score (for Convert=1)", value=f"{f1_score(y_test, y_pred_rf_test, pos_label=1):.2f}")

    st.markdown("---")

    # --- Thresholding for Actionable Insights ---
    st.subheader("Actionable Lead Prioritization")
    st.write(f"""
    Leads are prioritized based on their predicted conversion probability:
    - **High Priority**: Probability >= {HIGH_PRIORITY_THRESHOLD:.2f} 🚀
    - **Medium Priority**: Probability between {MEDIUM_PRIORITY_THRESHOLD:.2f} and {HIGH_PRIORITY_THRESHOLD:.2f} 💡
    - **Low Priority**: Probability < {MEDIUM_PRIORITY_THRESHOLD:.2f} 🐢
    """)

    # Classify test set leads by priority
    test_df_results = X_test.copy()
    test_df_results['actual_conversion'] = y_test
    test_df_results['predicted_prob'] = y_prob_rf_test

    test_df_results['priority'] = test_df_results['predicted_prob'].apply(get_priority)

    priority_counts = test_df_results['priority'].value_counts(normalize=True).sort_index()
    priority_conversion_rates = test_df_results.groupby('priority')['actual_conversion'].mean().sort_index()

    col_pri1, col_pri2 = st.columns(2)
    with col_pri1:
        st.write("#### Leads by Priority (% of Test Set)")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        ax_pie.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', startangle=90, colors=['gold', 'lightcoral', 'lightskyblue'])
        ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig_pie)

    with col_pri2:
        st.write("#### Actual Conversion Rate by Priority")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 6))
        sns.barplot(x=priority_conversion_rates.index, y=priority_conversion_rates.values, palette='coolwarm', ax=ax_bar)
        ax_bar.set_ylim(0, 1) # Ensure y-axis goes from 0 to 1 for rates
        ax_bar.set_ylabel('Actual Conversion Rate')
        ax_bar.set_title('') # Title already above
        plt.tight_layout()
        st.pyplot(fig_bar)

    st.markdown("---")

    # --- Estimated Sales Time Saved (Conceptual) ---
    st.subheader("Estimated Sales Time Saved (Conceptual)")
    st.write(f"""
    By leveraging this model, sales teams can strategically prioritize leads,
    focusing efforts on those most likely to convert.

    Assuming sales can **deprioritize leads classified as 'Low Priority'** (those with < {MEDIUM_PRIORITY_THRESHOLD:.2f} probability):
    """)
    low_priority_percent = priority_counts.get('Low Priority', 0)
    st.metric(label="Percentage of Leads that could be Deprioritized", value=f"{low_priority_percent:.1%}")

    # You can add a deeper calculation here, e.g., if a sales rep spends 10 mins on a low-priority lead, and handles 100 leads a week:
    # 100 leads/week * 10 mins/lead * low_priority_percent = minutes saved.
    # For a simple MVP, just showing the percentage of leads to deprioritize is strong enough.
    st.markdown(f"""
    This means sales reps could potentially save time previously spent on {low_priority_percent:.1%} of leads, allowing them to **focus more effectively** on High and Medium priority leads.
    """)

    st.markdown("---")
    st.subheader("Model Limitations & Next Steps")
    st.write("""
    * **Synthetic Data:** This model is trained on synthetic data. Real-world performance may vary.
    * **Feature Engineering:** Further improvements would come from richer, real-world features and deeper feature engineering.
    * **Advanced Techniques:** Exploring models like XGBoost or LightGBM, and more advanced imbalance handling (e.g., SMOTE) could yield higher performance.
    """)