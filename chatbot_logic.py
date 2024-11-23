import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import openai

# Set up OpenAI API Key
openai.api_key = "sk-proj--a-_Otm3E8qJh3cA"

# Use non-GUI backend for Matplotlib
plt.switch_backend('Agg')

# Load Dataset
file_path = 'Syntheic_Data.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Utility Functions
def summarize_dataset(df):
    """Generates a quick summary of the dataset."""
    try:
        summary = (
            f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Columns include: {', '.join(df.columns.tolist())}. "
            f"Sample values: {df.head(3).to_dict(orient='records')}"
        )
        return summary
    except Exception as e:
        return f"Error summarizing dataset: {e}"

def generate_graph(filtered_df, x_column, y_column, title, kind="bar"):
    """Generates a graph (scatter or bar) and returns it as a base64 string."""
    if filtered_df.empty:
        return None

    plt.figure(figsize=(8, 5))
    if kind == "scatter":
        colors = plt.cm.get_cmap("tab10", len(filtered_df))
        for idx, (vendor, row) in enumerate(filtered_df.iterrows()):
            vendor = row["Vendor"]  # Explicitly extract the vendor name
            plt.scatter(row[x_column], row[y_column], color=colors(idx), edgecolors='black', s=100, label=vendor)
            plt.annotate(vendor, (row[x_column], row[y_column]), fontsize=8)
        plt.legend(loc="best")
    else:  # Default to bar chart
        filtered_df.plot.bar(x=x_column, y=y_column, color=plt.cm.tab10(range(len(filtered_df))), legend=False)
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()

    return graph_base64

def create_combined_graph(filtered_df, x_column, y_column, title):
    """Generates a combined scatter plot for two metrics and returns it as a base64 string."""
    if filtered_df.empty:
        return None

    plt.figure(figsize=(8, 5))
    colors = plt.cm.get_cmap("tab10", len(filtered_df))
    for idx, (vendor, row) in enumerate(filtered_df.iterrows()):
        vendor = row["Vendor"]  # Explicitly extract the vendor name
        plt.scatter(row[x_column], row[y_column], color=colors(idx), edgecolors='black', s=100, label=vendor)
        plt.annotate(vendor, (row[x_column], row[y_column]), fontsize=8)
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()

    return graph_base64

def rank_suppliers(df, max_unit_price=None, top_n=5):
    """Ranks suppliers based on weighted criteria and optional unit price filter."""
    try:
        if max_unit_price is not None:
            df = df[df["Current Price Q2 unit $"] <= max_unit_price]
        if df.empty:
            return "No suppliers meet the criteria for ranking."

        # Weighted score calculation
        df["Score"] = (
            0.7 * (df["Current Price Q2 unit $"].max() - df["Current Price Q2 unit $"]) +
            0.2 * (df["MOQ (Lbs)"].max() - df["MOQ (Lbs)"]) +
            0.1 * (df["Lead Time (wks)"].max() - df["Lead Time (wks)"])
        )
        ranked_df = df.sort_values(by="Score", ascending=False).drop_duplicates(subset="Vendor").head(top_n)
        response = "Here are the top suppliers based on your expected unit price:\n" + "\n".join(
            [f"{idx + 1}) {row['Vendor']} - Score: {row['Score']:.2f}" for idx, (_, row) in enumerate(ranked_df.iterrows())]
        )
        return response
    except Exception as e:
        return f"Error ranking suppliers: {e}"

def nlp_query_response(df, user_query):
    """Processes natural language queries and returns results."""
    user_query = user_query.lower().strip()

    # Handle structured queries first
    if "generate the top" in user_query and "vendors by price" in user_query:
        num_vendors = 5 if "top 5" in user_query else 10 if "top 10" in user_query else 5
        sorted_df = df.sort_values(by="Current Price Q2 unit $").drop_duplicates(subset="Vendor").head(num_vendors)
        result = "\n".join([f"{idx + 1}) {row['Vendor']} - ${row['Current Price Q2 unit $']:.2f}"
                            for idx, (_, row) in enumerate(sorted_df.iterrows())])
        return f"Here are the top {num_vendors} vendors by price:\n{result}"

    if "generate a price graph" in user_query:
        sorted_df = df.sort_values(by="Current Price Q2 unit $").drop_duplicates(subset="Vendor").head(10)
        graph = generate_graph(sorted_df, "Vendor", "Current Price Q2 unit $", "Price Graph")
        return {"response": "Generated a price graph.", "graphs": {"dashboard_1": graph}}

    if "scatter plots" in user_query:
        metrics = {
            "price": "Current Price Q2 unit $",
            "moq": "MOQ (Lbs)",
            "lead time": "Lead Time (wks)"
        }
        requested_metrics = [key for key in metrics if key in user_query]
        if not requested_metrics:
            return "Please specify metrics like price, MOQ, or lead time for the scatter plots."

        graphs = {}
        for i, metric in enumerate(requested_metrics):
            graph = generate_graph(df.drop_duplicates(subset="Vendor").head(10),
                                   "Vendor", metrics[metric], f"{metric.title()} Scatter Plot", kind="scatter")
            if graph:
                graphs[f"dashboard_{i + 1}"] = graph

        if len(requested_metrics) > 1:
            combined_graph = create_combined_graph(df.drop_duplicates(subset="Vendor").head(10),
                                                   metrics[requested_metrics[0]], metrics[requested_metrics[1]],
                                                   f"Combined: {requested_metrics[0].title()} vs {requested_metrics[1].title()}")
            graphs["dashboard_3"] = combined_graph

        return {"response": "Here are your scatter plots for the requested metrics.", "graphs": graphs}

    if "lowest price for" in user_query:
        material = user_query.split("for")[-1].strip()
        filtered_df = df[df['INCI'].str.contains(material, case=False, na=False)]
        if filtered_df.empty:
            return f"No data found for material: {material}"
        result = filtered_df.sort_values(by="Current Price Q2 unit $").iloc[0]
        return (f"The lowest price for {material} is offered by {result['Vendor']} "
                f"at ${result['Current Price Q2 unit $']:.2f}, with a lead time of {result['Lead Time (wks)']} weeks.")

    # If no other logic applies, use the API for unstructured queries
    dataset_summary = summarize_dataset(df)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a data scientist using supplier data to provide insights. When making the decisions for ranking, you look at the weights price (70%), MOQ (20%), and lead time (10%)."
                               f"Don't explain how, just provide concise answers within 200 tokens. Provide formatted and clear answers, but again, have them be concise"
                               f"Dataset summary:\n{dataset_summary}"
                },
                {"role": "user", "content": user_query}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error communicating with OpenAI: {e}"
