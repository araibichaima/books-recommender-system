#for loading datasets
import pandas as pd 
#for creating arrays men datasets
import numpy as np

from surprise import Reader
from surprise import Dataset

#path to the book's dataset 
books_dataset=r"C:\Users\IP330\Desktop\data collab\collaborativefiltering\samples\books.csv"
#pah to the ratings' dataset
ratings_dataset=r"C:\Users\IP330\Desktop\data collab\collaborativefiltering\samples\ratings.csv"





#pd.read returns a data frame 
books = pd.read_csv(books_dataset)
ratings = pd.read_csv(ratings_dataset)

#a dataframe the contains both and books
books_ratings = pd.merge(books, ratings, on='book_id')
reader = Reader(rating_scale=(1, 5))
# Loads Pandas dataframe
data = Dataset.load_from_df(books_ratings[["user_id", "book_id", "rating"]], reader)
# Loads the builtin Movielens-100k data
data = Dataset.load_builtin('ml-100k')


from surprise import KNNWithMeans

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)


'''trainset = data.build_full_trainset()
algo.fit(trainset)
prediction = algo.predict(10,123)
prediction.est'''

from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data,test_size=.20)
algo.fit(trainset)
predictions=algo.test(testset)
pred_list=[]
est_list=[]




from flask import Flask
app = Flask(__name__)
	
from flask import request, redirect, url_for	
@app.route('/form', methods = ["POST", "GET"])
def hello():
	if request.method == 'POST':
		my_user_id= request.form.get('enter your id')
		my_user_id= request.form.get('enter your id')
		return redirect(url_for("recommendation", rcmd='my_user_id'))
	return ''' enter your id
	<form action= '#' method="POST"> <input type="number" name="enter you id"> 
	<input type ="submit" value "my_item_id"></form> '''	
	
@app.route("/<rcmd>")
def recommendation(rcmd):
	for x in range (0,30):
		pred=algo.predict(str(rcmd), str(x), r_ui=4, verbose=True)
		pred_list.append(str(pred)[str(pred).index('item'): str(pred).index('item')+15: 1])
		est_list.append(str(pred)[str(pred).index('est')+6: str(pred).index('est')+10: 1])
		
	final_list = list(zip(est_list, pred_list))
	final_list.sort(reverse=True)	   
	return "ids of books you might like: "+  str(final_list[:3])
if __name__ == '__main__':
	app.debug=True
	app.run()

#