import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import openai

from tabulate import tabulate

# Use non-GUI backend for Matplotlib
matplotlib.use('Agg')

# Load the dataset
file_path = 'Syntheic_Data.xlsx'  # Ensure the dataset path is correct
df = pd.read_excel(file_path)

# Set up OpenAI API Key
openai.api_key = ""

def preprocess_query(user_query):
    """
    Maps natural language queries to dataset attributes.
    """
    mappings = {
        "lowest price": "Current Price Q2 unit $",
        "shortest lead time": "Lead Time (wks)",
        "vendor": "Vendor",
        "inci": "INCI",
        "top 10": None,  # New entry for top 10 queries
        "top 5": None,  # New entry for top 5 queries
    }

    # Identify key terms and map to columns
    mapped_query = {key: value for key, value in mappings.items() if key in user_query.lower()}
    return mapped_query


def generate_graph(filtered_df, x_column, y_column, title):
    """
    Generates a graph based on the filtered dataset and returns it as a base64 string.
    """
    plt.figure(figsize=(8, 5))
    filtered_df.plot.scatter(x=x_column, y=y_column, c='blue')
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()

    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert to base64 string
    graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()  # Close figure to free resources

    return graph_base64


def truncate_text(text, max_length=20):
    """
    Truncates text to a specified maximum length for display.
    """
    return text if len(text) <= max_length else text[:max_length - 3] + "..."


def nlp_query_response(df, user_query):
    """
    Processes the user's natural language query and returns the result as a formatted list, graph, or OpenAI-generated response.
    """
    user_query = user_query.lower().strip()  # Normalize to lowercase and trim spaces

    # Handle queries for top vendors
    if "generate the top" in user_query:
        try:
            # Extract the number of top vendors (e.g., "top 5", "top 10")
            num = int(user_query.split("top")[-1].split()[0])
            if num < 1 or num > 10:
                return "Please choose a number between 1 and 10 for 'top' vendors."
        except ValueError:
            num = 5  # Default to top 5 if not specified

        if "price" in user_query:
            sorted_df = df.sort_values(by="Current Price Q2 unit $").head(num)
            if not sorted_df.empty:
                result_list = [
                    f"{idx + 1}) {row['Vendor']}, ${row['Current Price Q2 unit $']:.2f}"
                    for idx, (_, row) in enumerate(sorted_df.iterrows())
                ]
                return f"Here are the top {num} vendors by lowest price:\n" + "\n".join(result_list)

        elif "lead time" in user_query:
            sorted_df = df.sort_values(by="Lead Time (wks)").head(num)
            if not sorted_df.empty:
                result_list = [
                    f"{idx + 1}) {row['Vendor']}, {row['Lead Time (wks)']} weeks"
                    for idx, (_, row) in enumerate(sorted_df.iterrows())
                ]
                return f"Here are the top {num} vendors by shortest lead time:\n" + "\n".join(result_list)

    elif "shortest lead time" in user_query or "lead times" in user_query:
        sorted_df = df.sort_values(by="Lead Time (wks)").head(10)
        if not sorted_df.empty:
            result_list = [
                f"{idx + 1}) {row['Vendor']}, {row['Lead Time (wks)']} weeks"
                for idx, (_, row) in enumerate(sorted_df.iterrows())
            ]
            return "Here is a list of vendors with the shortest lead times:\n" + "\n".join(result_list)

    elif "lowest price" in user_query:
        # Find the supplier with the lowest price for a specific INCI
        if "for" in user_query:
            inci_name = user_query.split("for")[-1].strip()
            filtered_df = df[df['INCI'].str.contains(inci_name, case=False, na=False)]
        else:
            return "Please specify an INCI material after 'lowest price for'."

        result = filtered_df.sort_values(by="Current Price Q2 unit $").head(1)
        if not result.empty:
            response = result.iloc[0].to_dict()
            return f"{response['Vendor']} offers the lowest price for {inci_name} at ${response['Current Price Q2 unit $']:.2f} with a lead time of {response['Lead Time (wks)']} weeks."
        else:
            return f"No matching suppliers found for {inci_name}."

    elif "generate a price graph" in user_query:
        graph = generate_graph(df, "Vendor", "Current Price Q2 unit $", "Vendor Price Graph")
        return {"response": "Generated a price graph.", "graph": graph}

    elif "generate a lead time graph" in user_query:
        graph = generate_graph(df, "Vendor", "Lead Time (wks)", "Vendor Lead Time Graph")
        return {"response": "Generated a lead time graph.", "graph": graph}

    elif "generate an moq graph" in user_query:
        graph = generate_graph(df, "Vendor", "MOQ (Lbs)", "Vendor MOQ Graph")
        return {"response": "Generated an MOQ graph.", "graph": graph}

    elif "moq trends" in user_query or "trends in moq" in user_query:
        avg_moq = df["MOQ (Lbs)"].mean()
        max_moq = df["MOQ (Lbs)"].max()
        min_moq = df["MOQ (Lbs)"].min()
        return (f"The MOQ trends show an average MOQ of {avg_moq:.2f} lbs, "
                f"a maximum of {max_moq} lbs, and a minimum of {min_moq} lbs.")

    # Fallback for unstructured queries (leveraging OpenAI)
    # Fallback for unstructured queries (leveraging OpenAI)
    else:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an assistant that provides insights based on the following dataset columns: {', '.join(df.columns)}."
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"Error communicating with OpenAI: {e}"





def chatbot():
    """
    Provides an interactive chat-based interface for users to ask queries.
    """
    print("Welcome to the Supplier Recommendation Tool! Ask your questions.")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = nlp_query_response(df, user_query)
        if isinstance(response, dict) and "graph" in response:
            print(response["response"])
            # Save the graph to a file
            with open("generated_graph.png", "wb") as f:
                f.write(base64.b64decode(response["graph"]))
            print("Graph saved as 'generated_graph.png'")
        else:
            print(f"AI: {response}")
