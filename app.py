from flask import Flask, request, jsonify, render_template
import pandas as pd
from chatbot_logic import nlp_query_response, generate_graph

app = Flask(__name__)

# Load dataset
file_path = 'Syntheic_Data.xlsx'
df = pd.read_excel(file_path)

@app.route('/')
def home():
    """Render the homepage with example queries."""
    example_queries = [
        "Generate the top 5 vendors by price",
        "Generate a price graph",
        "What is the lowest price for Cocamidopropyl Betaine?"
    ]
    return render_template('index.html', examples=example_queries)

@app.route('/query', methods=['POST'])
def query():
    """Handle chatbot queries."""
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query cannot be empty."})

    # Process the query
    response = nlp_query_response(df, user_query)

    # If the response contains a graph, include the graph data
    if isinstance(response, dict) and "graph" in response:
        return jsonify({
            "response": response["response"],
            "graph": response["graph"]
        })

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=7010)
