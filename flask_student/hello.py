from flask import Flask
app = Flask(__name__)
# Flask类的route()函数是一个装饰器，它告诉应用程序哪个URL应该调用相关的函数。
# 第一个参数表示与该函数绑定的URL
# 第二个参数表示转发给底层的参数列表
@app.route('/')
def hello_world():
    return 'Hello World'

if __name__ == '__main__':
    app.run()
# Flask类的run()方法在本地开发服务器上运行应用程序。
# app.run(host, port, debug, options)
# host 监听的主机名。默认为127.0.0.1(localhost)。 设置为'0.0.0.0'使服务器在外部可用
# port 监听端口号，默认为:5000
#debug 默认为:false。 如果设置为:true，则提供调试信息
# options 被转发到底层的Werkzeug服务器。



