from process import preparation, generate_response
from flask import Flask, render_template, request,jsonify

# download nltk
preparation()

#Start Chatbot
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

@app.post('/predict')
def predict():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return jsonify(result)
if __name__ == "__main__":
    app.run(debug=True)