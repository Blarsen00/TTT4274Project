import matplotlib.pyplot as plt
from Part1 import ClassData30
# Part 2 - Seperability and its effect on the performance

# (a) For each of the four features compare the feature distribution for the four classes
    # - pop
    # - disco
    # - metal
    # - classical

# genres2 = ['pop', 'disco', 'metal', 'classical']

pop = ClassData30.query("Genre == 'pop'")
disco = ClassData30.query("Genre == 'disco'")
metal = ClassData30.query("Genre == 'metal'")
classical = ClassData30.query("Genre == 'classical'")

fig, ((ax00, ax01 , ax02, ax03), (ax10, ax11 , ax12, ax13), (ax20, ax21 , ax22, ax23), (ax30, ax31 , ax32, ax33)) = plt.subplots(nrows=4,ncols=4)
fig.align_labels()


# fig.suptitle("Histogram analysis")

ax00.hist(pop['spectral_rolloff_mean'])
ax00.set_ylabel('Pop')
# ax00.set_xlabel('Spectral rolloff mean')
ax00.grid()

ax01.hist(pop['mfcc_1_mean'])
# ax01.set_ylabel('Pop')
# ax01.set_xlabel('MFCC 1 mean')
ax01.grid()

ax02.hist(pop['spectral_centroid_mean'])
# ax02.set_ylabel('Pop')
# ax02.set_xlabel('Spectral centroid mean')
ax02.grid()

ax03.hist(pop['tempo'])
# ax03.set_ylabel('Pop')
# ax03.set_xlabel('Tempo')
ax03.grid()

ax10.hist(disco['spectral_rolloff_mean'])
ax10.set_ylabel('Disco')
# ax10.set_xlabel('Spectral rolloff mean')
ax10.grid()

ax11.hist(disco['mfcc_1_mean'])
# ax11.set_ylabel('Disco')
# ax11.set_xlabel('MFCC 1 mean')
ax11.grid()

ax12.hist(disco['spectral_centroid_mean'])
# ax12.set_ylabel('Disco')
# ax12.set_xlabel('Spectral centroid mean')
ax12.grid()

ax13.hist(disco['tempo'])
# ax13.set_ylabel('Disco')
# ax13.set_xlabel('Tempo')
ax13.grid()

ax20.hist(metal['spectral_rolloff_mean'])
ax20.set_ylabel('Metal')
# ax20.set_xlabel('Spectral rolloff mean')
ax20.grid()

ax21.hist(metal['mfcc_1_mean'])
# ax21.set_ylabel('Metal')
# ax21.set_xlabel('MFCC 1 mean')
ax21.grid()

ax22.hist(metal['spectral_centroid_mean'])
# ax22.set_ylabel('Metal')
# ax22.set_xlabel('Spectral centroid mean')
ax22.grid()

ax23.hist(metal['tempo'])
# ax23.set_ylabel('Metal')
# ax23.set_xlabel('Tempo')
ax23.grid()

ax30.hist(classical['spectral_rolloff_mean'])
ax30.set_ylabel('Classical')
ax30.set_xlabel('Spectral rolloff mean')
ax30.grid()

ax31.hist(classical['mfcc_1_mean'])
# ax31.set_ylabel('Classical')
ax31.set_xlabel('MFCC 1 mean')
ax31.grid()

ax32.hist(classical['spectral_centroid_mean'])
# ax32.set_ylabel('Classical')
ax32.set_xlabel('Spectral centroid mean')
ax32.grid()

ax33.hist(classical['tempo'])
# ax33.set_ylabel('Classical')
ax33.set_xlabel('Tempo')
ax33.grid()

fig.align_labels()
plt.show()
# plt.savefig(fname="HistogramsP2.pdf", format='pdf', bbox_inches='tight')