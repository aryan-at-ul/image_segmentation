import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import stats


img = cv2.imread("test.jpg")

# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN

from sklearn.mixture import GaussianMixture as GMM

#covariance choices, full, tied, diag, spherical

k = 7
gmm_model = GMM(n_components=k, covariance_type='full').fit(img2)  #tied works better than full
gmm_labels = gmm_model.predict(img2)

#Put numbers back to original shape so we can reconstruct segmented image
original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
cv2.imwrite("test2.jpg", segmented)

data = img2.ravel()
#data = data[data != 0]
#data = data[data != 1]  #Removes background pixels (intensities 0 and 1)

gmm = GMM(n_components = k)
gmm = gmm.fit(X=np.expand_dims(data,1))
gmm_x = np.linspace(0,255,256)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))


gmm_model.means_

gmm_model.covariances_

gmm_model.weights_


# Plot histograms and gaussian curves
fig, ax = plt.subplots()
ax.hist(img2.ravel(),255,[2,256], density=True, stacked=True)
ax.plot(gmm_x, gmm_y, color="crimson", lw=2, label="GMM")

ax.set_ylabel("Frequency")
ax.set_xlabel("Pixel Intensity")

plt.legend()
plt.grid(False)
plt.xlim([0, 256])

plt.show()

for m in range(gmm_model.n_components):


    pdf = gmm_model.weights_[m] * stats.norm(gmm_model.means_[m, 0],
                                       np.sqrt(gmm_model.covariances_[m, 0])).pdf(gmm_x.reshape(-1,1))


    fig, ax = plt.subplots()
    ax.hist(img2.ravel(),255,[2,256], density=True, stacked=True)
    ax.plot(gmm_x, gmm_y, color="crimson", lw=2, label="GMM")
    plt.fill(gmm_x, pdf, facecolor='gray',
             edgecolor='none')
    plt.xlim(0, 256)
    plt.ylim(0, .06)
