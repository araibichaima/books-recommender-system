


#for loading datasets
import pandas as pd 
#for creating arrays men datasets
import numpy as np

#path to the dataset you'll be working on
path_to_dataset=r"C:\Users\IP330\Desktop\data collab\contentbasedbookrecommender\BigML_Dataset_books.csv"
#columns in the dataset you believe you can drop
non_useful_columns=['url','price','save','pages','size','publisher','x_ray','lending','text_to_speech','customer_reviews','stars']
#number of rows from the dataset's tail you want to drop
#you can change it to 0 
n=45000
#the column that contains the data on which the similarity between items is based: 
#here, we consider two books are similar if they have similar descriptions
my_column='description'
#id of the item your recommendation is based on
my_item_id=123
#number of recommendations 
number_of_recommendations=3


#pd.read_csv take a csv file from and returns a data frame
ds1 = pd.read_csv(path_to_dataset)
#dropping non useful columns
ds = ds1.drop(columns=non_useful_columns)
#dropping rows if the dataset is too large
#you can choose not to do this
ds.drop(ds.tail(n).index,inplace=True)
#my dataset does not contain an id column
#the two folloing lines create one
ds['id'] = range(1, len(ds) + 1)
ds = ds.reset_index()

from sklearn.feature_extraction.text import TfidfVectorizer
#scikit-learn provides you a pre-built TF-IDF vectorizer that calculates the TF-IDF score for each documents description, word-by-word
#stop words are simply words that add no significant value to our system, like 'an', 'is', 'the', and hence are ignored by the system
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
#nans are replaced with empty strings this way
ds [my_column] = ds [my_column].fillna('') 
#the tfidf_matrix is the matrix containing each word and its TF-IDF score with regard to each document, or item in this case
tfidf_matrix = tf.fit_transform(ds[my_column])


#linear_kernel k(x, y) = x^T y + c
#linear kernel in our case calculates the cosine similarity
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 



#results is a matrix. Each line contains an item (in the first column) then the list of the items alike sorted according to cosine similarity
results = {}
for idx, row in ds.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices] 
   results[row['id']] = similar_items[1:]
   



def item(id):  
  return ds.loc[ds['id'] == id][my_column].tolist()[0].split(' - ')[0] 
# Just reads the results out of the dictionary.def 
def recommend(item_id, num):
	recommendation = "Recommending " + str(num) + " products similar to " + item(item_id) + "..."+ "\n" 
	print()   
	print("-------")    
	recs = results[item_id][:num]
	for rec in recs:
		recommendation= recommendation + "\n" + "Recommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")"
	return recommendation


recommendation=recommend(item_id=my_item_id, num=number_of_recommendations)	   
from flask import Flask
app = Flask(__name__)
	
from flask import request, redirect, url_for	
@app.route('/form', methods = ["POST", "GET"])
def hello():
	if request.method == 'POST':
		my_item_id= request.form.get('enter a book id you liked')
		return redirect(url_for("recommendation", rcmd=my_item_id))
	return ''' enter a book id you liked 
	<form action= '#' method="POST"> <input type="number" name="enter a book id you liked"> 
	<input type ="submit" value "my_item_id"></form> '''	
	
@app.route("/<rcmd>")
def recommendation(rcmd):
	my_recommendation=recommend(item_id=int(rcmd), num=number_of_recommendations)	   
	return my_recommendation
if __name__ == '__main__':
	app.debug=True
	app.run()

