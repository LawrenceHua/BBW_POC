from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io
from chatbot_logic import nlp_query_response

matplotlib.use('Agg')  # Use a non-interactive backend

# Set up Flask app
app = Flask(__name__)

# Load the dataset
file_path = 'Syntheic_Data.xlsx'  # Update to your correct file path
df = pd.read_excel(file_path)

# Function to rank suppliers by a given criterion
def rank_suppliers(df, inci_name, criterion):
    filtered_df = df[df['INCI'] == inci_name]
    
    if criterion == 'price':
        sorted_df = filtered_df.sort_values(by='Current Price Q2 unit $', ascending=True)
        column_name = 'Current Price Q2 unit $'
    elif criterion == 'lead_time':
        sorted_df = filtered_df.sort_values(by='Lead Time (wks)', ascending=True)
        column_name = 'Lead Time (wks)'
    elif criterion == 'moq':
        sorted_df = filtered_df.sort_values(by='MOQ (Lbs)', ascending=True)
        column_name = 'MOQ (Lbs)'
    elif criterion == 'score':
        sorted_df = recommend_suppliers(filtered_df)
        column_name = 'Score'
    else:
        return None, None

    if sorted_df.empty:
        return None, None

    sorted_df = sorted_df[['Vendor', 'INCI', column_name]].reset_index(drop=True)
    sorted_df['Rank'] = sorted_df.index + 1
    return sorted_df.head(10), column_name

# Function to create a bar plot
def create_plot(df, criterion):
    plt.figure(figsize=(10, 6))

    unique_vendors = [f"{vendor} ({idx})" if df['Vendor'].duplicated().iloc[idx] else vendor 
                      for idx, vendor in enumerate(df['Vendor'])]

    if criterion == 'price':
        plt.bar(unique_vendors, df['Current Price Q2 unit $'], color='lightblue')
        plt.ylabel('Price ($)')
    elif criterion == 'lead_time':
        plt.bar(unique_vendors, df['Lead Time (wks)'], color='lightgreen')
        plt.ylabel('Lead Time (weeks)')
    elif criterion == 'moq':
        plt.bar(unique_vendors, df['MOQ (Lbs)'], color='lightcoral')
        plt.ylabel('MOQ (Lbs)')
    elif criterion == 'score':
        plt.bar(unique_vendors, df['Score'], color='lightpink')
        plt.ylabel('Score')

    plt.xlabel('Vendor')
    plt.xticks(rotation=45)
    plt.title(f'Top 10 Suppliers by {criterion.capitalize()}')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# Function to recommend suppliers based on weighted score (Scorecard)
def recommend_suppliers(df, price_limit=None, lead_time_limit=None):
    if price_limit is not None:
        df = df[df['Current Price Q2 unit $'] <= price_limit]
    if lead_time_limit is not None:
        df = df[df['Lead Time (wks)'] <= lead_time_limit]

    price_max, price_min = df['Current Price Q2 unit $'].max(), df['Current Price Q2 unit $'].min()
    lead_time_max, lead_time_min = df['Lead Time (wks)'].max(), df['Lead Time (wks)'].min()
    moq_max, moq_min = df['MOQ (Lbs)'].max(), df['MOQ (Lbs)'].min()

    df['Score'] = df.apply(calculate_supplier_score, axis=1,
                           args=(price_max, price_min, lead_time_max, lead_time_min, moq_max, moq_min))

    recommended_suppliers = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    recommended_suppliers['Rank'] = recommended_suppliers.index + 1

    return recommended_suppliers[['Rank', 'Vendor', 'INCI', 'Current Price Q2 unit $', 'Lead Time (wks)', 'MOQ (Lbs)', 'Score']].head(10)

def calculate_supplier_score(row, price_max, price_min, lead_time_max, lead_time_min, moq_max, moq_min):
    normalized_price = 1 if price_max == price_min else (price_max - row['Current Price Q2 unit $']) / (price_max - price_min)
    normalized_lead_time = 1 if lead_time_max == lead_time_min else (lead_time_max - row['Lead Time (wks)']) / (lead_time_max - lead_time_min)
    normalized_moq = 1 if moq_max == moq_min else (moq_max - row['MOQ (Lbs)']) / (moq_max - moq_min)

    score = (0.7 * normalized_price) + (0.2 * normalized_lead_time) + (0.1 * normalized_moq)
    return score

# Route to render the ranking form and handle submissions
@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    column_name = None
    criterion = None
    img = None

    if request.method == 'POST':
        inci_name = request.form.get('inci_name')
        criterion = request.form.get('criterion')
        price_limit = request.form.get('price_limit')
        lead_time_limit = request.form.get('lead_time_limit')

        price_limit = float(price_limit) if price_limit else None
        lead_time_limit = int(lead_time_limit) if lead_time_limit else None

        if inci_name and criterion:
            if criterion == 'score':
                results = recommend_suppliers(df[df['INCI'] == inci_name], price_limit, lead_time_limit)
                column_name = 'Score'
            else:
                results, column_name = rank_suppliers(df, inci_name, criterion)

            if results is not None:
                img = create_plot(results, criterion)

    return render_template('index.html', results=results.to_dict(orient='records') if results is not None else None, column_name=column_name, criterion=criterion, image_available=(img is not None))

# Route to serve the generated chart
@app.route('/chart.png')
def chart_png():
    inci_name = request.args.get('inci_name')
    criterion = request.args.get('criterion')
    if inci_name and criterion:
        if criterion == 'score':
            results = recommend_suppliers(df[df['INCI'] == inci_name])
        else:
            results, _ = rank_suppliers(df, inci_name, criterion)
        if results is not None:
            img = create_plot(results, criterion)
            return send_file(img, mimetype='image/png')
    return "No chart available.", 404

# Route to render the chatbot page
@app.route('/chat')
def chat():
    return render_template('chat.html')

# Route to handle chatbot queries
@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query cannot be empty."})

    response = nlp_query_response(df, user_query)

    if isinstance(response, dict) and "graphs" in response:
        return jsonify({
            "response": response["response"],
            "graphs": {
                "dashboard_1": response["graphs"].get("dashboard_1", ""),
                "dashboard_2": response["graphs"].get("dashboard_2", ""),
                "dashboard_3": response["graphs"].get("dashboard_3", "")
            }
        })

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5010)
