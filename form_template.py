# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:10:58 2022

@author: Deepstrats
"""
from flask import Flask, redirect, url_for, render_template, request

# Flask constructor takes the __name__ of current module as an argument

app = Flask(__name__)

@app.route('/home/')

def welcome():
    return render_template('forms.html')


@app.route('/evaluate/<int:score>')

def eval(score):
    res = ""
    if score>35:
        res = "passed"
    else:
        res = "failed"
    return render_template('result.html', result = res)


@app.route('/submit', methods = ['POST','GET'])

def submit():
    
    total_score = 0
    
    if request.method == 'POST':
        Python = float(request.form['Python'])
        ML = float(request.form['ML'])
        Cloud = float(request.form['Cloud'])

        total_score = (Python+ML+Cloud)/3
                      
    return redirect(url_for('eval',score=total_score))

if __name__ == '__main__':
    app.run(debug=True)