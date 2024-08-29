from flask import Flask
app = Flask(__name__)
# 如果在浏览器中输入URL: http://localhost:5000/hello/YiibaiYiibai，那么 ‘YiibaiYiibai’ 将作为参数提供给hello()函数。
@app.route('/hello/<name>')
def hello_name(name):
    return 'Hello %s!' % name

if __name__ == '__main__':
    app.run(debug = True)

