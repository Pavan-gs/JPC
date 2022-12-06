# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:10:58 2022

@author: Deepstrats
"""
from flask import Flask, redirect, url_for

# Flask constructor takes the __name__ of current module as an argument

app = Flask(__name__)

@app.route('/home/')
def welcome():
    return "Hello Folks, welcome back to the Flask class!"

@app.route('/passed/<int:score>')
def passed(score):
    return "Congrats!, you've passed and your marks is " + str(score)

@app.route('/failed/<int:score>')
def failed(score):
    return "oops!, you've failed and your marks is " + str(score)

@app.route('/results/<int:marks>')
def results(marks):
    result = ""
    if marks<35:
        result = "failed"
    else:
        result = "passed"
    return redirect(url_for(result, score = marks))

if __name__ == '__main__':
    app.run(debug=True)