import logging
import time
from flask import Flask, Response, json, request, jsonify, stream_with_context, g
import sqlite3
import os

app = Flask(__name__)

# Database setup (using SQLite for simplicity)
DATABASE = os.environ.get("DATABASE_URL", "buttons.db")  # Use environment variable for flexibility

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():  # Initialize the database if it doesn't exist
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# Initialize the db on the first run, if it is not present
with app.app_context():
    try:
        db = get_db()
        db.execute("SELECT 1 FROM buttons LIMIT 1") # check if the table exists
    except sqlite3.OperationalError:
        init_db()

# Global list to store connected clients (for SSE)
clients = []

@app.route('/add_button', methods=['POST'])
def add_button():
    data = request.get_json()
    if not data or 'label' not in data or 'color' not in data:
        return jsonify({'error': 'Missing label or color'}), 400

    label = data['label']
    color = data['color']

    db = get_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO buttons (label, color) VALUES (?, ?)", (label, color))
    db.commit()

    # Notify connected clients about the new button
    for client in clients:
        #client['queue'].put(jsonify({'type': 'add', 'button': {'id': client['cursor'].lastrowid, 'label': label, 'color': color}}))  # Include ID
        client['queue'].put(({'type': 'add', 'button': {'id': client['cursor'].lastrowid, 'label': label, 'color': color}}))  # Include ID
        
        #time.sleep(1)  # For testing
        
    
    return jsonify({'message': 'Button added'}), 201


@app.route('/remove_button/<int:button_id>', methods=['DELETE'])
def remove_button(button_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM buttons WHERE id = ?", (button_id,))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({'error': 'Button not found'}), 404

    # Notify connected clients about the removed button
    for client in clients:
        client['queue'].put(({'type': 'remove', 'id': button_id}))

    return jsonify({'message': 'Button removed'}), 200


@app.route('/get_buttons', methods=['GET'])
def get_buttons():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, label, color FROM buttons")
    buttons = cursor.fetchall()
    button_list = []
    for button in buttons:
        button_list.append({'id': button[0], 'label': button[1], 'color': button[2]})
    json_data = json.dumps({"buttons": button_list})
    #return jsonify(button_list), 200
    return json_data, 200

@app.route('/set_buttons', methods=['POST'])
def import_buttons():
    try:
        data = request.get_json()
        if not data or not isinstance(data["buttons"], list):
            return jsonify({'error': 'Invalid JSON data. Expected a list of button objects.'}), 400

        db = get_db()
        cursor = db.cursor()


        # IMPORTANT: Notify clients *before* inserting new buttons to clear the page
        for client in clients:
            client['queue'].put(({'type': 'clear'}))  # Signal to clear existing buttons


        try:  # Wrap the entire import process in a try block for transaction control
            cursor.execute("DELETE FROM buttons")  # Clear the database
            db.commit()

            for button_data in data["buttons"]:
                if not isinstance(button_data, dict) or 'label' not in button_data or 'color' not in button_data:
                    return jsonify({'error': 'Each button object must have "label" and "color" keys.'}), 400

                label = button_data['label']
                color = button_data['color']

                cursor.execute("INSERT INTO buttons (label, color) VALUES (?, ?)", (label, color))
                db.commit()

                # Notify clients about the new button (after clearing and inserting)
                for client in clients:
                    client['queue'].put(({'type': 'add', 'button': {'id': cursor.lastrowid, 'label': label, 'color': color}}))

            return jsonify({'message': 'Buttons imported and database cleared successfully'}), 200

        except sqlite3.Error as e:  # Catch any database error during the entire process
            db.rollback()  # Rollback the entire transaction if any error occurs
            return jsonify({'error': f'Database error during import: {e}'}), 500

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred during import: {e}'}), 500


@app.route('/', methods=['GET'])
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Buttons</title>
</head>
<body>
    <div id="buttons-container"></div>

    <script>
        const buttonsContainer = document.getElementById('buttons-container');

        function loadButtons() { // Initial load
            fetch('/get_buttons')
                .then(response => response.json())
                .then(get_buttons => {
                    buttonsContainer.innerHTML = ''; // Clear existing buttons
                    get_buttons.buttons.forEach(button => addButtonToPage(button));
                });
        }

        function addButtonToPage(button) {
            const btn = document.createElement('button');
            btn.style.backgroundColor = button.color;
            btn.textContent = button.label;
            btn.id = button.id;
            buttonsContainer.appendChild(btn);
            buttonsContainer.appendChild(document.createElement('br'));
        }

        function removeButtonFromPage(id) {
            const buttons = buttonsContainer.querySelectorAll('button');
            buttons.forEach(button => {
              // Get the button's text content
              const buttonText = button.textContent;
              // Remove the button if its text content matches the label of the removed button
              //if (buttonText === id) {
              if (Number(button.id) == id){    
                buttonsContainer.removeChild(button);
              }
            });
        }
        

        const eventSource = new EventSource('/stream');

        eventSource.onmessage = function(event) {
            //console.log("Raw SSE data:", event);
            //console.log("Raw SSE data:", event.data);
            if (event.data instanceof Blob) {
                const reader = new FileReader();

                reader.onload = function() {
                    const decodedString = reader.result; // UTF-8 string
                    try {
                        const data = JSON.parse(decodedString); // If it's JSON
                        //console.log("Received JSON:", jsonObject);
                    } catch (error) {
                        //console.log("Received data (not JSON):", decodedString); // If it's not JSON
                    }
                };

                reader.readAsText(event.data, 'UTF-8'); // Read as UTF-8 text
            } else if (typeof event.data === 'string') { // Handle string messages (if any)
                //console.log("Received string (unlikely but possible):", event.data); // If it's not JSON
                data = JSON.parse(event.data);
            } else {
                //console.log("Received unknown data type:", event.data);
                const data = JSON.parse(event.data);
            }
            //const data = JSON.parse(jsonString);
            //const data = jsonObject
            if (data.type === 'add') {
                console.log("call addButtonToPage");
                addButtonToPage(data.button);
            } else if (data.type === 'remove') {
                removeButtonFromPage(data.id);
            } else if (data.type === 'clear') { // New: Handle the 'clear' event
                buttonsContainer.innerHTML = ''; // Clear the buttons container
            }
        };

        loadButtons(); // Initial button load
    </script>
    <button onclick="addButton()">Add Button</button>
</body>
</html>
"""
from queue import Queue  # Import Queue

@app.route('/stream')
def stream():
    q = Queue()
    db = get_db()
    cursor = db.cursor()

    client = {'queue': q, 'cursor': cursor}  # Store client with queue and cursor
    clients.append(client)  # Add client to the list

    def generate():
        try:
            while True:
                message = q.get()  # Get message from the queue
                yield f"data: {json.dumps(message)}\n\n"

        except GeneratorExit: # Client disconnected
            clients.remove(client) # Remove client
            db.close() # Close connection
            print("Client Disconnected")

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Make accessible from outside the container