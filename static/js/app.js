$(document).ready(function () {
    $('#send-query').on('click', function () {
        const userQuery = $('#user-query').val();

        if (!userQuery.trim()) return;

        // Add user message to chat
        $('#chat-messages').append(`<div class="user">${userQuery}</div>`);

        // Send query to Flask
        $.ajax({
            url: '/query',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ query: userQuery }),
            success: function (data) {
                // Format bot response by replacing \n with <br>
                const formattedResponse = data.response.replace(/\n/g, '<br>');

                // Add bot response to chat
                $('#chat-messages').append(`<div class="bot">${formattedResponse}</div>`);

                // Scroll to the latest message
                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);

                // Display graph if available
                if (data.graph) {
                    $('#graph-output').attr('src', `data:image/png;base64,${data.graph}`);
                }
            },
            error: function () {
                $('#chat-messages').append('<div class="bot">Error processing your query.</div>');
            },
        });

        // Clear input field
        $('#user-query').val('');
    });
});
