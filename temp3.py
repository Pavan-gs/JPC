# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:48:59 2022

@author: Deepstrats
"""

from flask import Flask, render_template

# Flask constructor takes the __name__ of current module as an argument

app = Flask(__name__)

@app.route('/home/<user>')

def index(user):
    return render_template('index1.html', name = user)

if __name__ == '__main__':
    app.run(debug=True)