import pandas as pd
import collections
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq, whiten


df = pd.read_csv('un.csv')
df.head()
print df.info()

#There are 207 rows + the header; there are 14 columns
#Best to cluster on infant mortality, tfr, or gdp since these variables have the largest number of non-null numerical results
#country and region are categorical (or string) variables, the remaining variables are real numbers
#207 countries
coords = df.as_matrix(columns=['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita'])
w = whiten(coords)

#This is the code to create 2 clusters, but whenever I try to run the following script, Python Shuts down
centroids,_ = kmeans(w,2)
# assign each sample to a cluster
idx,_ = vq(coords,centroids)

# some plotting using numpy's logical indexing
plot(df[idx==0,0],df[idx==0,1],'ob',
     df[idx==1,0],df[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()
