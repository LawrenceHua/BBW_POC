document.addEventListener("DOMContentLoaded", function () {
    const chatInput = document.getElementById("user-query");
    const sendQuery = document.getElementById("send-query");
    const chatMessages = document.getElementById("chat-messages");
    const dashboards = {
        "dashboard_1": document.getElementById("graph-output-1"),
        "dashboard_2": document.getElementById("graph-output-2"),
        "dashboard_3": document.getElementById("graph-output-3")
    };

    function addChatMessage(content, isUser = false) {
        const message = document.createElement("div");
        message.className = isUser ? "user" : "bot";
        message.textContent = content;
        chatMessages.appendChild(message);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to the bottom
    }

    function updateDashboard(graphs) {
        // Clear all dashboards first
        Object.values(dashboards).forEach(dashboard => {
            dashboard.src = "";
            dashboard.alt = "Graph will appear here";
        });

        // Update dashboards with generated graphs
        for (const [dashboardId, graphData] of Object.entries(graphs)) {
            if (dashboards[dashboardId] && graphData) {
                dashboards[dashboardId].src = `data:image/png;base64,${graphData}`;
                dashboards[dashboardId].alt = "Generated Graph";
            }
        }
    }

    sendQuery.addEventListener("click", () => {
        const query = chatInput.value.trim();
        if (!query) return;

        // Add user's query to the chat messages
        addChatMessage(query, true);
        chatInput.value = "";

        // Send the query to the backend
        fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query })
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    addChatMessage(data.error, false);
                    return;
                }

                // Handle textual response
                if (data.response) {
                    addChatMessage(data.response, false);
                }

                // Handle graph data
                if (data.graphs) {
                    updateDashboard(data.graphs);
                }
            })
            .catch(error => {
                addChatMessage("An error occurred while processing your query. Please try again.", false);
                console.error("Error:", error);
            });
    });
});
