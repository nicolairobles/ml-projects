# Business-Focused Beginner AI/ML Projects
## Customer Churn & Fraud Detection with Streamlit Dashboards

---

## **Project 1: Customer Churn Prediction Dashboard (RECOMMENDED START)**

### **Why This Project is Perfect for Beginners:**
- Uses real business problem everyone understands
- Pre-existing dataset (Telco Customer Churn from Kaggle)
- Clear success metric (predict who will cancel subscription)
- Beautiful dashboard shows predictions + business insights
- **1-2 weeks to complete**

---

### **What You'll Build:**

#### **Interactive Business Dashboard with:**
1. **Upload customer data** (CSV file with customer info)
2. **Instant predictions** - which customers are at risk of churning
3. **Risk scores** - high/medium/low risk categories with confidence levels
4. **Visual analytics:**
   - Churn rate by customer segment
   - Key factors driving churn (contract type, payment method, tenure)
   - Retention recommendations for each at-risk customer
5. **Export predictions** - download CSV with churn predictions for business action

---

### **Demo Strategy:**

**Live Dashboard Features:**
- Upload sample customer data → See instant churn predictions
- Interactive filters (filter by contract type, payment method, service type)
- Business KPIs displayed prominently:
  - "32% of month-to-month customers likely to churn"
  - "Fiber optic customers 2.5x more likely to cancel"
  - "Customers with electronic check payment have highest churn risk"
- Actionable recommendations panel
- Export button to download results

**Demo Flow:**
1. Open dashboard → Upload customer file
2. See predictions load in real-time
3. Click on high-risk customers → See detailed analysis
4. Show business insights charts
5. Export predictions as CSV

---

### **Tech Stack:**
- **Data:** Telco Customer Churn dataset (Kaggle - free download)
- **ML Model:** Random Forest or XGBoost (pre-built, no training needed initially)
- **Dashboard:** Streamlit (turns Python into beautiful web app)
- **Deployment:** Streamlit Cloud (free hosting)

---

### **Step-by-Step Implementation:**

#### **Phase 1: Basic Model (Days 1-3)**

**Day 1: Setup & Data Exploration**
```python
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset (download from Kaggle)
df = pd.read_csv('telecom_churn.csv')

# Quick exploration
st.title("Customer Churn Prediction Dashboard")
st.write(f"Total Customers: {len(df)}")
st.write(f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
```

**Day 2: Train Simple Model**
```python
# Prepare data (convert Yes/No to 1/0, handle categorical variables)
# Feature engineering
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model for dashboard
import joblib
joblib.dump(model, 'churn_model.pkl')
```

**Day 3: Basic Dashboard**
```python
st.title("🎯 Customer Churn Predictor")

# Upload file
uploaded_file = st.file_uploader("Upload customer data (CSV)", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    
    # Add results to dataframe
    data['Churn_Risk'] = predictions
    data['Churn_Probability'] = probabilities
    
    # Show results
    st.subheader("Predictions")
    st.dataframe(data[['customerID', 'Churn_Risk', 'Churn_Probability']])
    
    # Download button
    st.download_button("Download Predictions", 
                      data.to_csv(index=False),
                      "churn_predictions.csv")
```

#### **Phase 2: Business Analytics (Days 4-7)**

**Add Business Insights:**
```python
import plotly.express as px

# Key metrics at top
col1, col2, col3 = st.columns(3)
with col1:
    high_risk = len(data[data['Churn_Probability'] > 0.7])
    st.metric("High Risk Customers", high_risk, 
             delta=f"{high_risk/len(data):.1%} of total")

with col2:
    avg_risk = data['Churn_Probability'].mean()
    st.metric("Average Churn Risk", f"{avg_risk:.1%}")

with col3:
    potential_loss = high_risk * 1000  # $1000 per customer
    st.metric("Potential Revenue at Risk", f"${potential_loss:,}")

# Visualization: Churn by segment
fig = px.bar(data.groupby('Contract')['Churn_Risk'].mean(),
            title="Churn Rate by Contract Type",
            labels={'value': 'Churn Rate', 'Contract': 'Contract Type'})
st.plotly_chart(fig)

# Feature importance
st.subheader("Key Factors Driving Churn")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

fig = px.bar(feature_importance, x='Importance', y='Feature',
            orientation='h', title="Top 10 Churn Predictors")
st.plotly_chart(fig)
```

**Add Filters:**
```python
# Sidebar filters
st.sidebar.header("Filters")
contract_filter = st.sidebar.multiselect("Contract Type", 
                                        data['Contract'].unique())
risk_threshold = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.5)

# Apply filters
filtered_data = data.copy()
if contract_filter:
    filtered_data = filtered_data[filtered_data['Contract'].isin(contract_filter)]
filtered_data = filtered_data[filtered_data['Churn_Probability'] >= risk_threshold]

st.write(f"Showing {len(filtered_data)} customers")
st.dataframe(filtered_data)
```

**Add Recommendations:**
```python
st.subheader("💡 Retention Recommendations")

for idx, row in data[data['Churn_Probability'] > 0.7].head(10).iterrows():
    with st.expander(f"Customer {row['customerID']} - {row['Churn_Probability']:.1%} risk"):
        st.write(f"**Contract:** {row['Contract']}")
        st.write(f"**Tenure:** {row['tenure']} months")
        st.write(f"**Monthly Charges:** ${row['MonthlyCharges']}")
        
        # Smart recommendations based on customer profile
        if row['Contract'] == 'Month-to-month':
            st.info("💼 **Recommend:** Offer 1-year contract with 15% discount")
        if row['tenure'] < 12:
            st.info("🎁 **Recommend:** Provide loyalty bonus or service upgrade")
        if row['PaymentMethod'] == 'Electronic check':
            st.info("💳 **Recommend:** Incentivize automatic payment method")
```

#### **Phase 3: Polish & Deploy (Days 8-10)**

**Add Professional Styling:**
```python
# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Add logo/branding
st.sidebar.image("logo.png", use_column_width=True)
```

**Deploy to Streamlit Cloud:**
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect GitHub repo
4. Click "Deploy"
5. Get shareable link!

---

### **Existing Resources to Learn From:**

#### **Complete Tutorials:**
1. **YouTube: "Customer Churn Prediction Using ML"** (53 min)
   https://www.youtube.com/watch?v=da_xqw1oAD8
   - Full EDA, multiple models (Logistic Regression, Random Forest, XGBoost)
   - Handles class imbalance
   - Model evaluation with ROC-AUC

2. **365 Data Science Written Tutorial**
   https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/
   - Step-by-step Random Forest implementation
   - Telco dataset walkthrough
   - Feature engineering examples

3. **Reddit: Full Production Churn Project** (3 repos!)
   https://www.reddit.com/r/learnmachinelearning/comments/1nk4u0f/a_full_churn_prediction_project_from_eda_to/
   - EDA & Data Pipeline
   - Model Training & Evaluation  
   - Production-ready deployment
   - Updated weekly with improvements

4. **Microsoft Fabric Tutorial**
   https://learn.microsoft.com/en-us/fabric/data-science/customer-churn
   - End-to-end notebook
   - scikit-learn + LightGBM
   - Power BI visualizations

#### **Kaggle Notebooks:**
- "CUSTOMER CHURN PREDICTION": https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction
- Complete EDA, feature engineering, multiple models

#### **Dataset:**
- **Telco Customer Churn** (Kaggle): 7,043 customers, 21 features
- Features: tenure, contract type, monthly charges, internet service, payment method, etc.

---

## **Project 2: Real-Time Fraud Detection Dashboard**

### **Why This Project is Great:**
- High-impact business problem ($3.4B lost to fraud annually)
- Visual patterns in data (fraud vs legitimate transactions)
- Real-time detection simulation
- **1-2 weeks to complete**

---

### **What You'll Build:**

#### **Real-Time Fraud Monitoring Dashboard:**
1. **Transaction stream simulator** - shows incoming transactions in real-time
2. **Instant fraud detection** - flags suspicious transactions immediately
3. **Alert system** - highlights fraudulent transactions with confidence scores
4. **Analytics panel:**
   - Fraud detection rate
   - False positive rate
   - Transaction patterns over time
   - Geographic fraud hotspots (if location data available)
5. **Investigation panel** - drill down into flagged transactions

---

### **Demo Strategy:**

**Live Dashboard Features:**
- **Start simulation** button → transactions stream in real-time
- Fraudulent transactions flash red with alert sound (optional)
- Counter shows: "Detected 14 fraudulent transactions out of 847 processed"
- Click on flagged transaction → see why it was flagged (unusual amount, location, time, etc.)
- Charts update in real-time showing fraud patterns

**Demo Flow:**
1. Click "Start Real-Time Monitoring"
2. Watch transactions scroll by
3. See red alerts pop up for suspicious transactions
4. Pause and investigate a flagged transaction
5. Show analytics dashboard with fraud trends

---

### **Tech Stack:**
- **Data:** Credit Card Fraud Detection dataset (Kaggle - 284,807 transactions)
- **ML Model:** Logistic Regression, Random Forest, or XGBoost
- **Dashboard:** Streamlit with real-time updates
- **Deployment:** Streamlit Cloud (free)

---

### **Step-by-Step Implementation:**

#### **Phase 1: Basic Fraud Detector (Days 1-4)**

**Day 1-2: Load Dataset & Train Model**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load credit card fraud dataset from Kaggle
df = pd.read_csv('creditcard.csv')

# Note: This dataset has 28 anonymized features (V1-V28) + Time, Amount, Class
# Class: 0 = legitimate, 1 = fraudulent

# Split data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, 'fraud_detector.pkl')
```

**Day 3-4: Basic Streamlit Dashboard**
```python
import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("🚨 Real-Time Fraud Detection System")

# Load model
model = joblib.load('fraud_detector.pkl')

# Load transaction data
df = pd.read_csv('creditcard.csv')

# Simulation controls
st.sidebar.header("Simulation Controls")
speed = st.sidebar.slider("Transactions per second", 1, 10, 5)
start_sim = st.sidebar.button("Start Real-Time Monitoring")

# Metrics display
col1, col2, col3 = st.columns(3)
metric_transactions = col1.empty()
metric_frauds = col2.empty()
metric_rate = col3.empty()

# Transaction feed
st.subheader("Transaction Stream")
transaction_feed = st.empty()

# Fraud alerts
st.subheader("🚨 Fraud Alerts")
fraud_alerts = st.empty()

if start_sim:
    total_transactions = 0
    total_frauds_detected = 0
    fraud_list = []
    
    for idx, row in df.iterrows():
        # Simulate real-time processing
        time.sleep(1/speed)
        
        # Make prediction
        features = row.drop('Class').values.reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        total_transactions += 1
        
        # Update metrics
        if prediction == 1:
            total_frauds_detected += 1
            fraud_list.append({
                'Transaction ID': idx,
                'Amount': row['Amount'],
                'Fraud Probability': f"{probability:.2%}",
                'Time': row['Time']
            })
        
        # Display current transaction
        status = "🔴 FRAUD DETECTED" if prediction == 1 else "✅ Legitimate"
        transaction_feed.write(f"Transaction #{total_transactions}: Amount ${row['Amount']:.2f} - {status}")
        
        # Update metrics
        metric_transactions.metric("Total Transactions", total_transactions)
        metric_frauds.metric("Frauds Detected", total_frauds_detected)
        metric_rate.metric("Fraud Rate", f"{total_frauds_detected/total_transactions:.2%}")
        
        # Show fraud alerts
        if len(fraud_list) > 0:
            fraud_alerts.dataframe(pd.DataFrame(fraud_list).tail(10))
        
        # Stop after 100 transactions for demo
        if total_transactions >= 100:
            st.success("✅ Simulation Complete")
            break
```

#### **Phase 2: Add Analytics (Days 5-7)**

**Enhanced Dashboard with Visualizations:**
```python
import plotly.graph_objects as go
import plotly.express as px

# Add time series chart showing fraud over time
st.subheader("Fraud Detection Over Time")

fraud_timeline = pd.DataFrame(fraud_list)
if len(fraud_timeline) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fraud_timeline.index,
        y=fraud_timeline['Amount'],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Fraudulent Transactions'
    ))
    fig.update_layout(
        title="Fraudulent Transaction Amounts",
        xaxis_title="Transaction Number",
        yaxis_title="Amount ($)"
    )
    st.plotly_chart(fig)

# Distribution comparison
st.subheader("Transaction Amount Distribution")
fig = px.histogram(df, x='Amount', color='Class', 
                  labels={'Class': 'Transaction Type'},
                  title="Legitimate vs Fraudulent Transaction Amounts",
                  color_discrete_map={0: 'green', 1: 'red'})
st.plotly_chart(fig)

# Model performance metrics
st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.metric("Precision", "94.2%", help="Of flagged transactions, 94.2% are actually fraudulent")
    st.metric("Recall", "87.5%", help="Model catches 87.5% of all fraudulent transactions")

with col2:
    st.metric("False Positive Rate", "0.8%", help="Only 0.8% of legitimate transactions are incorrectly flagged")
    st.metric("F1 Score", "0.91", help="Overall balance between precision and recall")
```

**Add Investigation Panel:**
```python
st.subheader("🔍 Transaction Investigation")

if len(fraud_list) > 0:
    selected_fraud = st.selectbox("Select transaction to investigate:", 
                                 [f"Transaction #{f['Transaction ID']}" for f in fraud_list])
    
    # Get transaction details
    trans_id = int(selected_fraud.split('#')[1])
    transaction = df.loc[trans_id]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Transaction Details:**")
        st.write(f"Amount: ${transaction['Amount']:.2f}")
        st.write(f"Time: {transaction['Time']} seconds from first transaction")
        st.write(f"Actual Label: {'FRAUD' if transaction['Class'] == 1 else 'Legitimate'}")
    
    with col2:
        st.write("**Risk Factors:**")
        # Feature importance for this specific transaction
        features = transaction.drop('Class').values.reshape(1, -1)
        probability = model.predict_proba(features)[0][1]
        
        st.progress(probability, text=f"Fraud Probability: {probability:.2%}")
        
        # Show top contributing features
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            st.write("Top risk indicators:")
            for _, row in feature_importance.iterrows():
                st.write(f"• {row['Feature']}: {row['Importance']:.3f}")
```

#### **Phase 3: Business Insights (Days 8-10)**

**Add Business Context:**
```python
st.subheader("💼 Business Impact Analysis")

# Calculate potential losses
avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
detected_frauds = total_frauds_detected
prevented_loss = detected_frauds * avg_fraud_amount

col1, col2, col3 = st.columns(3)
col1.metric("Average Fraud Amount", f"${avg_fraud_amount:.2f}")
col2.metric("Frauds Prevented", detected_frauds)
col3.metric("Loss Prevention", f"${prevented_loss:,.2f}", delta="Saved")

# Alert system
st.subheader("⚙️ Alert Configuration")
threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 
                     help="Transactions above this probability trigger alerts")

st.info(f"Current setting will flag ~{(df['Class'].sum() * threshold):.0f} transactions per day")

# Historical performance
st.subheader("📊 Historical Performance")
col1, col2 = st.columns(2)

with col1:
    # Confusion matrix
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, predictions)
    fig = px.imshow(cm, 
                   labels=dict(x="Predicted", y="Actual"),
                   x=['Legitimate', 'Fraud'],
                   y=['Legitimate', 'Fraud'],
                   color_continuous_scale='Reds',
                   text_auto=True)
    st.plotly_chart(fig)

with col2:
    # ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier', 
                            line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate')
    st.plotly_chart(fig)
```

---

### **Existing Resources:**

#### **Complete Tutorials:**
1. **YouTube: "Fraud Detection Using ML - Full Project"** (43 min)
   https://www.youtube.com/watch?v=4Od5_z28iIE
   - 94% accuracy fraud detection
   - Data preprocessing, feature engineering
   - Model optimization techniques

2. **GeeksforGeeks: Credit Card Fraud Detection**
   https://www.geeksforgeeks.org/machine-learning/ml-credit-card-fraud-detection/
   - Step-by-step implementation
   - Handling imbalanced datasets
   - Model evaluation metrics

3. **Microsoft Fabric Tutorial**
   https://learn.microsoft.com/en-us/fabric/data-science/fraud-detection
   - End-to-end fraud detection workflow
   - Feature engineering
   - MLflow experiment tracking

4. **Reddit Discussion: Extremely Imbalanced Dataset**
   https://www.reddit.com/r/learnmachinelearning/comments/1g6jx90/trying_to_build_an_effective_fraud_detection/
   - Strategies for 1:47,500 imbalance ratio
   - SMOTE and synthetic data generation
   - cGAN approaches

#### **Dataset:**
- **Credit Card Fraud Detection** (Kaggle): 284,807 transactions
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Features anonymized for privacy (V1-V28 are PCA components)
- Only 0.172% are fraudulent (highly imbalanced)

---

## **Comparison: Which Project to Start With?**

| Factor | Customer Churn | Fraud Detection |
|--------|---------------|----------------|
| **Difficulty** | Easier | Slightly harder |
| **Dataset Balance** | More balanced (~27% churn) | Highly imbalanced (~0.17% fraud) |
| **Business Appeal** | Strong (subscription businesses) | Very strong (financial services) |
| **Interpretability** | High (clear features like contract type) | Medium (anonymized features) |
| **Demo "Wow Factor"** | Good (clear recommendations) | Excellent (real-time simulation) |
| **Learning Value** | Feature engineering, classification | Imbalanced data handling, real-time processing |

### **Recommendation:**
**Start with Customer Churn** if your team wants:
- Clearer feature interpretation
- More straightforward model training
- Business recommendations that make intuitive sense

**Choose Fraud Detection** if your team wants:
- More impressive real-time demo
- Experience with imbalanced datasets
- Financial services domain experience

---

## **Combined Project Idea: Customer Risk Dashboard**

**Why not combine both?**

Build a unified "Customer Risk Management Dashboard" that includes:
1. **Churn prediction** module
2. **Fraud detection** module  
3. **Customer lifetime value** calculator
4. **Retention recommendations** engine

This creates a comprehensive portfolio piece showing multiple ML use cases in one cohesive business application.

**Demo strategy:**
- Tab 1: Churn Risk Analysis
- Tab 2: Fraud Monitoring
- Tab 3: Customer Value Insights
- Tab 4: Recommended Actions

**Implementation time:** 3-4 weeks
**Portfolio value:** One comprehensive project vs. two separate ones

---

## **Quick Start Action Plan:**

### **Week 1: Customer Churn MVP**
- Day 1-2: Download dataset, train basic model
- Day 3-4: Build simple Streamlit dashboard
- Day 5-7: Add visualizations and deploy

### **Week 2: Fraud Detection MVP**
- Day 8-10: Train fraud detection model
- Day 11-13: Build real-time simulation dashboard
- Day 14: Polish and deploy

### **Week 3: Enhancement** (Optional)
- Combine both into unified dashboard
- Add advanced features (what-if scenarios, automated reports)
- Write blog post about learnings

### **Result:**
By end of Week 2, your team has:
✅ **2 live business dashboards** with real ML models
✅ **Shareable links** for portfolio and LinkedIn
✅ **Interview stories** about handling business problems with ML
✅ **Concrete experience** with classification, imbalanced data, deployment

---

## **Key Takeaways:**

1. **Both projects use existing datasets** - no data collection needed
2. **Both have Streamlit dashboards** - visual, demo-able, shareable
3. **Both solve real business problems** - strong interview talking points
4. **Both are beginner-friendly** - use standard ML algorithms (Random Forest, XGBoost)
5. **Both can be deployed free** - Streamlit Cloud hosting

**The secret:** Focus on the business problem, not just the algorithm. Employers care more about "How did you solve the churn problem?" than "What hyperparameters did you tune?"
