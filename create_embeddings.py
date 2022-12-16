from course_helpers import *
from helpers import *

DATA_PATH = 'full_data/'
n_emd = 50
Dataset = read(DATA_PATH)
print('reading done')
coco_pkl = DATA_PATH+'coco_pkl.pkl'
embd = DATA_PATH+'embeddings_'+str(n_emd)
glove(coco_pkl, embd, n_emd)


