from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # Read the latest data from the CSV file
    csv_file = 'stock_data.csv'
    
    try:
        df = pd.read_csv(csv_file)
        data = df[['Symbol', 'LTP', 'Prediction']].tail(20).to_dict(orient='records')  # Show the latest 20 rows
    except FileNotFoundError:
        data = []

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
