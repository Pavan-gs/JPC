# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:20:23 2022

@author: Deepstrats
"""


from flask import Flask

# Flask constructor takes the __name__ of current module as an argument

app = Flask(__name__)

@app.route('/')

def welcome():
    return "Hello Folks, welcome back to the Flask class! "


@app.route('/sub/<name>')

def sub(name):
    return 'this is %s!' %name


@app.route('/page1/<int:duration>')

def dur(duration):
    return "the duration of the course would be %d" %duration

def sub_fee():
    return "This is the fee for subject"

app.add_url_rule('/home/subfee',view_func=sub_fee)

if __name__ == '__main__':
    app.run(debug=True)