from process import preparation, generate_response
from flask import Flask, render_template, request,jsonify

# download nltk
preparation()

#Start Chatbot
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/get", methods=["GET", "POST"])
def get_bot_response():
    if request.method == "POST":
        user_input = str(request.args.get('msg'))
        result = generate_response(user_input)
        print("Methodnya post bree")
        return jsonify (result)
    else:
        print("Methodnya get bree")
        return 0
        
@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    response = generate_response(text)
    message = {"answer": response}
    return jsonify(message)
if __name__ == "__main__":
    app.run(debug=True)
