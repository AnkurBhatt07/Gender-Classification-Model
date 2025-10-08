from flask import Flask 

from app import views 

app = Flask(__name__)

app.add_url_rule('/' , endpoint = 'home' , view_func = views.index)
app.add_url_rule('/app/' , endpoint = 'app' , view_func=views.app)
app.add_url_rule('/app/gender/' , 
                 endpoint = 'gender' , 
                 view_func=views.genderapp,
                  methods = ['GET' , 'POST'] )





if __name__ == "__main__":
    app.run(debug = True)


