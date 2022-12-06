from flask import Flask

# Flask constructor takes the __name__ of current module as an argument

app = Flask(__name__)

@app.route('/home/')

def welcome():
    return "Hello Folks, welcome back to the Flask class!"


@app.route('/sub/<name>')

def sub(name):
    return 'this is %s!' %name


@app.route('/page1/<int:duration>')

def dur(duration):
    return "the duration of the course would be %d" %duration


if __name__ == '__main__':
    app.run(debug=True)


