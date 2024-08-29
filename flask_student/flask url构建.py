from flask import Flask, redirect, url_for
app = Flask(__name__)

@app.route('/admin')
def hello_admin():
    return 'Hello Admin'

@app.route('/guest/<guest>')
def hello_guest(guest):
    return 'Hello %s as Guest' % guest

@app.route('/user/<name>')
def user(name):
    if name =='admin':
        return redirect(url_for('hello_admin'))
    else:
        return redirect(url_for('hello_guest',guest = name))

if __name__ == '__main__':
    app.run(debug = True)

#User()函数检查收到的参数是否与’admin’匹配。 如果匹配，则使用url_for()将应用程序重定向到hello_admin()函数，
# 否则将该接收的参数作为guest参数传递给hello_guest()函数。




