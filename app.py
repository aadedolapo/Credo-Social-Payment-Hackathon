import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
from random import randint
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

def load_data(data):
    df = pd.read_csv(data, encoding='unicode_escape')[:10000]
    return df

df = load_data("Global_Superstore2.csv")

# Collaborative filtering using Cosine Similairties
pivot_df = pd.pivot_table(df,index = 'Order ID',columns = 'Product Name',values = 'Order Priority',aggfunc = 'count')
pivot_df.reset_index(inplace=True)
pivot_df = pivot_df.fillna(0)
pivot_df = pivot_df.drop('Order ID', axis=1)

co_matrix = pivot_df.T.dot(pivot_df)
np.fill_diagonal(co_matrix.values, 0)

cos_score_df = pd.DataFrame(cosine_similarity(co_matrix))
cos_score_df.index = co_matrix.index
cos_score_df.columns = np.array(co_matrix.index)

#Take top five scoring recs that aren't the original product
product_recs = []
for i in cos_score_df.index:
    product_recs.append(cos_score_df[cos_score_df.index!=i][i].sort_values(ascending = False)[0:5].index)
recommendations = pd.DataFrame(product_recs)
recommendations.index = cos_score_df.index  
 
@st.cache   
def get_recommendations(df,item):
    """Generate a set of product recommendations using item-based collaborative filtering.

    Args:
        df (dataframe): Pandas dataframe containing matrix of items purchased.
        item (string): Column name for target item. 

    Returns: 
        recommendations (dataframe): Pandas dataframe containing product recommendations. 
    """
    recs_products = list(df.loc[item])
    
    url = ['https://'+each +'.com' for each in df.loc[item].str.replace(' ','')]
    price = [randint(100,1000) for _ in range(len(df.loc[item]))]
    
    result = pd.DataFrame(url,recs_products, columns=['Url']).reset_index().rename(columns={'index':'Recommended Products'})
    result['Price'] =price

    return result
    
RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
</div>
"""

@st.cache
def search_term_if_not_found(search_term,df):
	result_df = df[df['Product Name'].str.contains(search_term)].drop_duplicates()
	return result_df['Product Name']

def main():
    st.title("Product Recommendation App")
    
    menu = ["Home", "Recommend", "About"]
    choice= st.sidebar.selectbox("Menu", menu)
      
    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))
    elif choice == "Recommend":
        st.subheader("Recommend Products")
        search_term = st.text_input("Search")
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendations(recommendations,search_term)
                    for row in results.iterrows():
                      rec_product = row[1][0]
                      rec_url = row[1][1]
                      rec_price = row[1][2]
    						
                      stc.html(RESULT_TEMP.format(rec_product,rec_url,rec_price),height=150)
                                       
                except:
                    results= "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term,df)
                    st.dataframe(result_df)
                
    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas by TeamLocalhost")
        
if __name__ == '__main__':
    main()
    
