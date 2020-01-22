from flask import Flask

app = Flask(__name__)

# URL example: http://127.0.0.1:5000/
@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"


if __name__ == '__main__':
    app.run(debug=True)