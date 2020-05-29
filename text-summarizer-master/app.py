from flask import Flask, redirect, request

# For flashing a  message
from flask import flash

# For rendering html templates
from flask import render_template

# For linking files, generates a URL
from flask import url_for

# Import our forms
from forms import LinkForm

from summarizer import getSum


# intatiates flask app, configures the application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'This is a secret'

def stringtolist(li):
    return [x.strip("'") for x in li.strip('][').split(', ')]

@app.route("/summary", methods=["GET", "POST"])
def summary():
    form = LinkForm()
    u = request.form.getlist('link')
    print(u)
    text, keywords = getSum(u)
    print(text,' ',keywords)
    text = ' '.join(text)
    if form.validate_on_submit():
        flash(f'Summarized!', 'success')
    return render_template('summary.html', form=form, text=text, u1=u[0],u2=u[1],u3=u[2], keywords=keywords)


@app.route("/",  methods=['GET', 'POST'])
def main():
    form = LinkForm()
    
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
