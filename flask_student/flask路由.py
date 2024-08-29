from flask import Flask
app = Flask(__name__)
@app.route('/hello')
def hello_world():
    return 'Hello World2'

if __name__ == '__main__':
    app.run()



