
# BBW Recommendation Tool

Welcome to the **BBW Recommendation Tool**! This application integrates AI-powered insights and interactive visualizations to rank suppliers and enhance decision-making through a custom UI.

---

## **Getting Started**

### **Prerequisites**
1. **OpenAI API Key**  
   Follow this guide to obtain an OpenAI API key:  
   [How to Get Your Own OpenAI API Key](https://medium.com/@lorenzozar/how-to-get-your-own-openai-api-key-f4d44e60c327#:~:text=Create%20A%20New%20API%20Key,button%20Create%20new%20secret%20key.&text=In%20the%20next%20pop%2C%20just,API%20key%20for%20different%20things.)  
   > **Note:** Most structured queries and all visualizations work without an API key, so feel free to explore quickly without one.  

2. **Update API Key**  
   Insert your new key in `chatbot_logic.py` at the following line:  
   [chatbot_logic.py#L8](https://github.com/LawrenceHua/BBW_POC/blob/main/chatbot_logic.py#L8)

### **Installation**
1. Open your terminal and run:
   ```bash
   python -r requirements.txt
   ```
2. Start the app:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:  
   [http://localhost:5010](http://localhost:5010)

---

## **Features**

### **1. AI-Powered Supplier Recommendations**
Upon launching the application, you'll encounter the **Custom PowerBI Visual**. This allows users to obtain AI-driven rankings and recommendations in a simple UI.

#### **Steps to Use:**
1. **Enter the INCI (raw material)**: Example - `Cocamidopropyl Betaine`  
2. **Select Ranking Criterion**: Choose from available options.  
3. **Click "Rank Suppliers"**:  
   - A table of vendors with corresponding visuals will be displayed.

---

### **2. AI Chatbot for Visualizations**
At the bottom of the screen, the **Chatbot Button** provides a "custom PowerBI dashboard." Users can interact with the chatbot using keywords like "bar chart" or "scatter plot" to generate visuals dynamically.

#### **Steps to Use:**
1. From the initial screen, click **"Chat with our AI Assistant"** to open the chatbot interface.  
2. Copy a bullet point or question into the Chatbox UI on the left.  
3. Click **"Send"**, and the chatbot will respond with insights or visuals.

---

## **Capabilities**
This tool showcases the potential of integrating **PowerBI + Azure AI** (and in the future, **PowerBI + Azure ML**) to create the perfect dashboard for BBW's needs.  

- **Supplier Scoring**  
- **Interactive Visualizations**  
- **AI-Driven Recommendations**

Explore the functionalities and see how this tool can revolutionize supplier management and analytics!
```
