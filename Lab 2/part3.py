import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# use seaborn plotting defaults
import seaborn as sns; sns.set()

# Creates datapoints
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()

# Draws lines that separate the data
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
 plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5)
plt.show()

# Add margins to the lines from earlier
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
 yfit = m * xfit + b
 plt.plot(xfit, yfit, '-k')
 plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.show()

# --------------------------------------- Fitting support ------------------------------------

# Creates a linear kernel with a large C (Small margin)
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

# Function to plot decision boundaries
def plot_svc_decision_function(model, ax=None, plot_support=True):
 if ax is None:
    ax = plt.gca()
 xlim = ax.get_xlim()
 ylim = ax.get_ylim ()
 
 # create grid to evaluate model
 x = np.linspace(xlim[0], xlim[1], 30)
 y = np.linspace(ylim[0], ylim[1], 30)
 Y, X = np.meshgrid(y, x)
 xy = np.vstack([X.ravel(), Y.ravel()]).T
 P = model.decision_function(xy).reshape(X.shape)
 
 # plot decision boundary and margins
 ax.contour(X, Y, P, colors='k',
 levels=[-1, 0, 1], alpha=0.5,
 linestyles=['--', '-', '--'])
 
 # plot support vectors
 if plot_support:
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
 s=300, linewidth=1, facecolors='none')
 ax.set_xlim(xlim)
 ax.set_ylim(ylim)
 
# Calls the above function and draws a line with a margin
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()


print(model.support_vectors_)

# Applies model to the first 60 datapoints and the first 120 datapoints respectively
def plot_svm(N=10, ax=None):
 X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
 X = X[:N]
 y = y[:N]
 model = SVC(kernel='linear', C=1E10)
 model.fit(X, y)
 ax = ax or plt.gca()
 ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
 ax.set_xlim(-1, 4)
 ax.set_ylim(-1, 6)
 plot_svc_decision_function(model, ax)
 
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
 plot_svm(N, axi)
 axi.set_title('N = {0}'.format(N))
plt.show()


# Data which is not linearly seperable
from sklearn.datasets import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)
plt.show()

# Projection which is a radial basis function centered on the middle clump
r = np.exp(-(X ** 2).sum(1))

# Visualize data in 3-dimensions
from mpl_toolkits import mplot3d
ax = plt.subplot(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], r, s=50, cmap='autumn')
ax.view_init(elev=30, azim=30)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')
plt.show()


# Creates rbf kernel (rbf = Radial Basis Function)
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=300, lw=1, facecolors='none')
plt.show()

# -------------------------------------- Tuning --------------------------------------

# Data with a lot of overlaps
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()

# Shows how changing C affects the softness of the margin
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, C in zip(ax, [10.0, 0.1]):
 model = SVC(kernel='linear', C=C).fit(X, y)
 axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
 plot_svc_decision_function(model, axi)
 axi.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],s=300, lw=1, facecolors='none')
 axi.set_title('C = {0:.1f}'.format(C), size=14)
plt.show()