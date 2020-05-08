import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from word2vec.w2v import Word2Vec
from RandomIndexing.random_indexing import RandomIndexing
import os
import numpy as np


def draw_interactive(x, y, text):
    """
    Draw a plot visualizing word vectors with the posibility to hover over a datapoint and see
    a word associating with it
    
    :param      x:     A list of values for the x-axis
    :type       x:     list
    :param      y:     A list of values for the y-axis
    :type       y:     list
    :param      text:  A list of textual values associated with each (x, y) datapoint
    :type       text:  list
    """
    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn

    fig,ax = plt.subplots()
    sc = plt.scatter(x, y, c='b', s=100, cmap=cmap, norm=norm)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        note = "{}".format(" ".join([text[n] for n in ind["ind"]]))
        annot.set_text(note)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def load_words_and_embeddings(fname): 
    
    try:
        with open(fname, 'r') as f:
            vocab, dim = (int(a) for a in next(f).split())

            words, X =  [] , np.zeros((vocab, dim))

            for i, line in enumerate(f):
                parts = line.split()
                word = parts[0].strip()
                X[i] = list(map(float, parts[1:])) # vectors of H dimensions
                words.append(word)
       
    except:
        print("Error: failing to load the model to the file")
    return words, X



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding visualization toolkit')
    parser.add_argument('--file', type=str, help='A textual file containing word vectors')
    parser.add_argument('-v', '--vector-type', default='w2v', choices=['w2v', 'ri','glove'])
    parser.add_argument('-d', '--decomposition', default='pca', choices=['svd', 'pca'],
                        help='Your favorite decomposition method')
    args = parser.parse_args()

    # YOUR CODE HERE

    # Note: Both model is built from only the corpus from the first book to avoid crashing
    
    # If Random Indexing is used
    if (args.vector_type == 'ri'):
        words, X = load_words_and_embeddings("random_indexing_300d_wsize3.txt")
    # If Word2Vec is used
    elif (args.vector_type == 'w2v'):
        words, X = load_words_and_embeddings("w2v_ps_uniform_300d_wsize3_LRS_on_focus.txt")
    # If Glove is used
    else:
        words, X = load_words_and_embeddings("glove_300d.txt")
    
    # if PCA is used
    if (args.decomposition == 'pca'):
        pca = PCA(n_components=2)
        reduced_X = pca.fit_transform(X) # performing dimensionality reduction 
        draw_interactive(reduced_X[:, 0], reduced_X[:, 1], words)
    # if SVD is used
    else:
        svd = TruncatedSVD(n_components=2)
        reduced_X = svd.fit_transform(X)
        draw_interactive(reduced_X[:, 0], reduced_X[:, 1], words)
