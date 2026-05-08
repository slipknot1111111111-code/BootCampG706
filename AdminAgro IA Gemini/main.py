from flask import Flask, render_template, request, jsonify
from chatbot.ai import predict_answer

app = Flask(__name__, static_folder='chatbot/static')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/cultivos")
def cultivos():
    return render_template("cultivos.html")

@app.route("/animales")
def animales():
    return render_template("animales.html")

@app.route("/inventario")
def inventario():
    return render_template("inventario.html")

@app.route("/labores")
def labores():
    return render_template("labores.html")

@app.route("/gastos")
def gastos():
    return render_template("gastos.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.form.get("message", "")
    if not user_text.strip():
        return jsonify({"response": "Por favor escribe algo para ayudarte."})
    response = predict_answer(user_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
