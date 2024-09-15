import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import pytz  # For timezone handling

# Nepal Timezone
nepal_tz = pytz.timezone('Asia/Kathmandu')

def fetch_data():
    url = "https://www.sharesansar.com/live-trading"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        table = soup.find('table')
        rows = table.find_all('tr')
        
        data = []
        for row in rows[1:]:  # Skip header row
            columns = row.find_all('td')
            row_data = [col.text.strip() for col in columns]
            if len(row_data) == 10:  # Ensure the row has the correct number of columns
                data.append(row_data)
        
        df = pd.DataFrame(data, columns=['S.No', 'Symbol', 'LTP', 'Point Change', '% Change', 'Open', 'High', 'Low', 'Volume', 'Prev. Close'])
        return df
    else:
        print("Failed to retrieve data")
        return pd.DataFrame()  # Return empty DataFrame on failure

# Initialize model and scaler
df = fetch_data()

df['S.No'] = pd.to_numeric(df['S.No'], errors='coerce')
df['LTP'] = pd.to_numeric(df['LTP'].str.replace(',', ''), errors='coerce')
df['Point Change'] = pd.to_numeric(df['Point Change'].str.replace(',', ''), errors='coerce')
df['% Change'] = pd.to_numeric(df['% Change'].str.replace('%', '').str.replace(',', ''), errors='coerce')
df['Open'] = pd.to_numeric(df['Open'].str.replace(',', ''), errors='coerce')
df['High'] = pd.to_numeric(df['High'].str.replace(',', ''), errors='coerce')
df['Low'] = pd.to_numeric(df['Low'].str.replace(',', ''), errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='coerce')
df['Prev. Close'] = pd.to_numeric(df['Prev. Close'].str.replace(',', ''), errors='coerce')

df = df.fillna(0)
df['Target'] = (df['Point Change'] > 0).astype(int)
features = ['S.No', 'LTP', 'Open', 'High', 'Low', 'Volume', 'Prev. Close']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained.")

# File to store the data
csv_file = 'stock_data.csv'

while True:
    #here the data is only fetched from nepse between 11AM and 12:30PM
    current_time = datetime.now(nepal_tz)
    
    if current_time.hour >= 11 and (current_time.hour < 12 or (current_time.hour == 12 and current_time.minute <= 30)):
        df = fetch_data()
        
        if not df.empty:
            df['S.No'] = pd.to_numeric(df['S.No'], errors='coerce')
            df['LTP'] = pd.to_numeric(df['LTP'].str.replace(',', ''), errors='coerce')
            df['Point Change'] = pd.to_numeric(df['Point Change'].str.replace(',', ''), errors='coerce')
            df['% Change'] = pd.to_numeric(df['% Change'].str.replace('%', '').str.replace(',', ''), errors='coerce')
            df['Open'] = pd.to_numeric(df['Open'].str.replace(',', ''), errors='coerce')
            df['High'] = pd.to_numeric(df['High'].str.replace(',', ''), errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'].str.replace(',', ''), errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='coerce')
            df['Prev. Close'] = pd.to_numeric(df['Prev. Close'].str.replace(',', ''), errors='coerce')
            df = df.fillna(0)

            X_live = df[features]
            X_live = scaler.transform(X_live)
            predictions = model.predict(X_live)
            df['Prediction'] = predictions
            
            # Append data to CSV
            df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)
            print(df[['Symbol', 'LTP', 'Prediction']])
        
    else:
        print("Outside trading hours.")
    
    time.sleep(600)  # Wait for 10 minutes before fetching data again
