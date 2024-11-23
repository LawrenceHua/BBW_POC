from flask import Flask, request, jsonify, render_template
import pandas as pd
from chatbot_logic import nlp_query_response

app = Flask(__name__)

# Load dataset
file_path = 'Syntheic_Data.xlsx'
df = pd.read_excel(file_path)

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle chatbot queries."""
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query cannot be empty."})

    # Process the query
    response = nlp_query_response(df, user_query)

    # Handle graph responses
    if isinstance(response, dict) and "graphs" in response:
        # Each graph is assigned to a specific dashboard
        return jsonify({
            "response": response["response"],
            "graphs": {
                "dashboard_1": response["graphs"].get("dashboard_1", ""),
                "dashboard_2": response["graphs"].get("dashboard_2", ""),
                "dashboard_3": response["graphs"].get("dashboard_3", "")
            }
        })

    # Textual responses
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True, port=7010)
