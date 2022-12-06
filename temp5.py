# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:32:10 2022

@author: Deepstrats
"""
from flask import Flask, redirect, url_for, render_template

# Flask constructor takes the __name__ of current module as an argument

app = Flask(__name__)

@app.route('/home/<int:score>')

def index(score):
    return render_template('index2.html', marks = score)

if __name__ == '__main__':
    app.run(debug=True)