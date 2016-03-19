from flask import Flask,render_template
app = Flask(__name__, static_url_path='')


import pandas as pd

FPS = 20


def flatten_df(df):
     
    return dict(
        )

@app.route("/")
def root():
    print 'hi'
    df = pd.read_csv('fused.csv')
    frame_per_second = FPS
    return render_template('index.html', fused=df.to_dict(), video=dict(fps=frame_per_second))

if __name__ == "__main__":
    app.run(debug=True)
