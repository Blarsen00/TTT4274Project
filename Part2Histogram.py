import matplotlib.pyplot as plt
from Part1 import ClassData30
# Part 2 - Seperability and its effect on the performance

# (a) For each of the four features compare the feature distribution for the four classes
    # - pop
    # - disco
    # - metal
    # - classical


genres = ['pop', 'disco', 'metal', 'classical']
features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']

def plotHistogram(data, genres, features, grid=True):
    """The function produces a grid of histograms for corresponding genres and features

        Produces a nxm grid where n is the number of genres and m is the number of features.
    
        Parameters
        ----------
        data : pd.Dataframe
            A dataframe of all the relevant data
        genres : [str]
            A list of genres that is to be considered. Strings need to match with the keys of data
        features : [str]
            A list of the features that is to be considered. Strings need to match with the keys of data
        grid : Boolean, optional
            True sets a grid on each plot
    """
    # Declare a subplot grid
    fig, ax = plt.subplots(len(genres), len(features))
    
    for g in range(len(genres)):
        for f in range(len(features)):
            genre = genres[g]
            # dg is all the relevant data for the genre 
            dg = data.query("Genre == @genre")
            ax[g][f].hist(dg[features[f]])
            if grid:
                ax[g][f].grid()
            if g == len(genres)-1:
                ax[g][f].set_xlabel(features[f])
            if f == 0:
                ax[g][f].set_ylabel(genres[g])

    fig.align_labels() 
    plt.show()
    return fig
    


# fig  = plotHistogram(data=ClassData30, genres=genres, features=features)
