"""
Load Excel files to PostgreSQL - FIXED VERSION
This will actually load your data!
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os
from pathlib import Path

# Database connection
DB_URL = "postgresql://postgres:postgres123@localhost:5432/churn_analytics"

print("="*80)
print("📊 LOADING YOUR EXCEL DATA TO POSTGRESQL")
print("="*80)

try:
    # Create engine
    engine = create_engine(DB_URL)
    print("\n✅ Connected to PostgreSQL")
    
    # ============================================================================
    # STEP 1: Find your Excel file
    # ============================================================================
    
    print("\n📁 Looking for your Excel files...")
    
    # Check common locations
    possible_files = [
        "data/processed/churn_engineered.xlsx",
        "data/processed/churn_engineered.csv",
        "data/raw/Telco-Customer-Churn.xlsx",
        "data/raw/Telco-Customer-Churn.csv",
        "churn_engineered.xlsx",
        "Telco-Customer-Churn.xlsx"
    ]
    
    file_found = None
    for filepath in possible_files:
        if Path(filepath).exists():
            file_found = filepath
            print(f"   ✅ Found: {filepath}")
            break
    
    if not file_found:
        print("\n❌ No data file found automatically.")
        print("\n📝 Please enter the FULL PATH to your Excel file:")
        print("   Example: C:\\Users\\YourName\\Desktop\\churn_data.xlsx")
        file_found = input("\n   Enter path: ").strip().strip('"').strip("'")
        
        if not Path(file_found).exists():
            print(f"\n❌ File not found: {file_found}")
            print("\n💡 Please check:")
            print("   1. File path is correct")
            print("   2. File extension (.xlsx or .csv)")
            print("   3. No typos in path")
            exit(1)
    
    # ============================================================================
    # STEP 2: Load the file
    # ============================================================================
    
    print(f"\n📂 Loading file: {file_found}")
    
    if file_found.endswith('.xlsx'):
        df = pd.read_excel(file_found)
    elif file_found.endswith('.csv'):
        df = pd.read_csv(file_found)
    else:
        print("❌ Unsupported file format. Use .xlsx or .csv")
        exit(1)
    
    print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"\n📋 Columns in file:")
    for col in df.columns[:10]:  # Show first 10
        print(f"      - {col}")
    if len(df.columns) > 10:
        print(f"      ... and {len(df.columns) - 10} more columns")
    
    # ============================================================================
    # STEP 3: Clean and prepare data
    # ============================================================================
    
    print("\n🔧 Preparing data for database...")
    
    # Convert column names to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle TotalCharges if it's a string
    if 'totalcharges' in df.columns:
        if df['totalcharges'].dtype == 'object':
            df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
            df['totalcharges'].fillna(0, inplace=True)
    
    # Convert Churn to binary if needed
    if 'churn' in df.columns:
        if df['churn'].dtype == 'object':
            df['churn'] = (df['churn'].str.upper() == 'YES').astype(int)
    
    # Convert SeniorCitizen to string
    if 'seniorcitizen' in df.columns:
        df['seniorcitizen'] = df['seniorcitizen'].map({0: 'No', 1: 'Yes'}).fillna('No')
    
    # Remove customerID column if exists (it causes issues)
    if 'customerid' in df.columns:
        customer_ids = df['customerid'].copy()
        df = df.drop('customerid', axis=1)
        print("   ℹ️  Removed customerid column (will be handled separately)")
    
    print(f"   ✅ Data prepared: {len(df)} rows ready")
    
    # ============================================================================
    # STEP 4: Load main predictions table
    # ============================================================================
    
    print("\n1️⃣ Loading PREDICTIONS table...")
    df.to_sql('predictions', engine, if_exists='replace', index=False, method='multi', chunksize=500)
    print(f"   ✅ Inserted {len(df)} rows into predictions table")
    
    # ============================================================================
    # STEP 5: Create summary metrics
    # ============================================================================
    
    print("\n2️⃣ Creating SUMMARY_METRICS table...")
    
    total_customers = len(df)
    churned = df['churn'].sum() if 'churn' in df.columns else 0
    retained = total_customers - churned
    churn_rate = (churned / total_customers * 100) if total_customers > 0 else 0
    avg_charges = df['monthlycharges'].mean() if 'monthlycharges' in df.columns else 0
    avg_tenure = df['tenure'].mean() if 'tenure' in df.columns else 0
    
    summary_data = {
        'metric_name': [
            'Total Customers',
            'Churned Customers',
            'Retained Customers',
            'Churn Rate (%)',
            'Avg Monthly Charges',
            'Avg Tenure (months)'
        ],
        'metric_value': [
            total_customers,
            churned,
            retained,
            round(churn_rate, 2),
            round(avg_charges, 2),
            round(avg_tenure, 2)
        ],
        'metric_category': [
            'Overview',
            'Overview',
            'Overview',
            'Overview',
            'Revenue',
            'Customer'
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_sql('summary_metrics', engine, if_exists='replace', index=False)
    print(f"   ✅ Created {len(df_summary)} summary metrics")
    
    # ============================================================================
    # STEP 6: Create contract analysis
    # ============================================================================
    
    print("\n3️⃣ Creating CONTRACT_ANALYSIS table...")
    
    if 'contract' in df.columns and 'churn' in df.columns:
        contract_df = df[['contract', 'churn']].copy()
        contract_df = contract_df[contract_df['churn'].isin([0, 1])]
        
        contract_stats = contract_df.groupby('contract').agg(
            customer_count=('churn', 'count'),
            churners=('churn', 'sum')
        ).reset_index()
        
        contract_stats['avg_churn_prob'] = contract_stats['churners'] / contract_stats['customer_count']
        contract_stats['churn_rate_pct'] = contract_stats['avg_churn_prob'] * 100
        
        # Add additional stats
        if 'monthlycharges' in df.columns:
            charges_by_contract = df.groupby('contract')['monthlycharges'].mean().reset_index()
            charges_by_contract.columns = ['contract', 'avg_monthly_charges']
            contract_stats = contract_stats.merge(charges_by_contract, on='contract')
        
        if 'tenure' in df.columns:
            tenure_by_contract = df.groupby('contract')['tenure'].mean().reset_index()
            tenure_by_contract.columns = ['contract', 'avg_tenure']
            contract_stats = contract_stats.merge(tenure_by_contract, on='contract')
        
        contract_stats.to_sql('contract_analysis', engine, if_exists='replace', index=False)
        print(f"   ✅ Created contract analysis with {len(contract_stats)} rows")
    
    # ============================================================================
    # STEP 7: Create internet analysis
    # ============================================================================
    
    print("\n4️⃣ Creating INTERNET_ANALYSIS table...")
    
    if 'internetservice' in df.columns and 'churn' in df.columns:
        internet_df = df[['internetservice', 'churn']].copy()
        internet_df = internet_df[internet_df['churn'].isin([0, 1])]
        
        internet_stats = internet_df.groupby('internetservice').agg(
            customer_count=('churn', 'count'),
            churners=('churn', 'sum')
        ).reset_index()
        
        internet_stats['avg_churn_prob'] = internet_stats['churners'] / internet_stats['customer_count']
        internet_stats['churn_rate_pct'] = internet_stats['avg_churn_prob'] * 100
        
        internet_stats.to_sql('internet_analysis', engine, if_exists='replace', index=False)
        print(f"   ✅ Created internet analysis with {len(internet_stats)} rows")
    
    # ============================================================================
    # STEP 8: Create tenure analysis
    # ============================================================================
    
    print("\n5️⃣ Creating TENURE_ANALYSIS table...")
    
    if 'tenure' in df.columns and 'churn' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72, 100],
            labels=['0-1 years', '1-2 years', '2-4 years', '4-6 years', '6+ years']
        )
        
        tenure_df = df[['tenure_group', 'churn']].copy()
        tenure_df = tenure_df[tenure_df['churn'].isin([0, 1])]
        
        tenure_stats = tenure_df.groupby('tenure_group').agg(
            customer_count=('churn', 'count'),
            churners=('churn', 'sum')
        ).reset_index()
        
        tenure_stats['avg_churn_prob'] = tenure_stats['churners'] / tenure_stats['customer_count']
        tenure_stats['churn_rate_pct'] = tenure_stats['avg_churn_prob'] * 100
        
        tenure_stats.to_sql('tenure_analysis', engine, if_exists='replace', index=False)
        print(f"   ✅ Created tenure analysis with {len(tenure_stats)} rows")
    
    # ============================================================================
    # STEP 9: Create monthly trends
    # ============================================================================
    
    print("\n6️⃣ Creating MONTHLY_TRENDS table...")
    
    if 'tenure' in df.columns and 'churn' in df.columns:
        monthly_data = []
        
        for month in range(1, 13):
            month_df = df[df['tenure'] >= month].copy()
            if len(month_df) > 0:
                high_risk = len(month_df[month_df['churn'] == 1])
                monthly_data.append({
                    'month': f'Month {month}',
                    'total_customers': len(month_df),
                    'churners': high_risk,
                    'churn_rate_pct': (high_risk / len(month_df) * 100) if len(month_df) > 0 else 0
                })
        
        df_monthly = pd.DataFrame(monthly_data)
        df_monthly.to_sql('monthly_trends', engine, if_exists='replace', index=False)
        print(f"   ✅ Created monthly trends with {len(df_monthly)} rows")
    
    # ============================================================================
    # VERIFICATION
    # ============================================================================
    
    print("\n" + "="*80)
    print("✅ DATA LOADED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📊 Verifying data in PostgreSQL...")
    
    tables = ['predictions', 'summary_metrics', 'contract_analysis', 
              'internet_analysis', 'tenure_analysis', 'monthly_trends']
    
    with engine.connect() as conn:
        for table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"   ✅ {table}: {count} rows")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("\n1. Go to Power BI Desktop")
    print("2. Click 'Refresh' button (or Close & Apply)")
    print("3. Your tables should now have data!")
    print("4. Start building your dashboard")
    print("\n" + "="*80)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\n🔍 Troubleshooting:")
    print("1. Make sure PostgreSQL is running")
    print("2. Check database password: postgres123")
    print("3. Verify database exists: churn_analytics")
    print("4. Install packages: pip install pandas sqlalchemy psycopg2-binary openpyxl")
    print("\n💡 If error persists, share the error message!")
    
    import traceback
    print("\n📋 Full error details:")
    traceback.print_exc()