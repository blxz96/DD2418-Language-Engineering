import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from word2vec.w2v import Word2Vec
from RandomIndexing.random_indexing import RandomIndexing
import os


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding visualization toolkit')
    parser.add_argument('--file', type=str, help='A textual file containing word vectors')
    parser.add_argument('-v', '--vector-type', default='ri', choices=['w2v', 'ri'])
    parser.add_argument('-d', '--decomposition', default='svd', choices=['svd', 'pca'],
                        help='Your favorite decomposition method')
    args = parser.parse_args()

    #
    # YOUR CODE HERE
    #

    if (args.vector_type == 'ri'):
        dir_name = "RandomIndexing/data"
        filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]

        ri = RandomIndexing(filenames)
        ri.train()
        x = ri.getMatrix()
        words = ri.getWords()
        # x is the weight matrix, words is the corespoding word list


        if (args.decomposition == 'pca'):
            pca = PCA(n_components=2)
            x_new = pca.fit_transform(x)
            draw_interactive(x_new[:, 0], x_new[:, 1], words)
        else:
            svd = TruncatedSVD(n_components=2)
            x_new = svd.fit_transform(x)
            draw_interactive(x_new[:, 0], x_new[:, 1], words)

        # pca = PCA(n_components = 10)
        # x_new = pca.fit_transform(x)
        # filewrite = open("WordsandMatrix.txt", "w")
        # for i in range(len(x_new)):
        #     filewrite.write(words[i])
        #     filewrite.write("$")
        #     filewrite.write(" ".join(map(str, x_new[i])))
        #     filewrite.write('\n')