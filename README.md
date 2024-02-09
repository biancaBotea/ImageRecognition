The goal of this project is to introduce me to image recognition. Specifically, I will examine the task of scene recognition starting with very simple methods – tiny images and nearest neighbour classification – and then move on to techniques that resemble the state-of-the-art – bags of quantized local features and linear classifiers learned by support vector machines.

I have a set of labelled development images to develop and tune my classifiers from. Additionally, I have a set of unlabelled images for which I produce predictions of the correct class.

Run #1: I have developed a simple k-nearest-neighbor classifier using the “tiny image” feature. The “tiny image” feature is one of the simplest possible image representations. One simply crops each image to a square about the center, and then resizes it to a small, fixed resolution (16x16 is recommended). The pixel values can be packed into a vector by concatenating each image row. It tends to work slightly better if the tiny image is made to have zero mean and unit length. I can choose the optimal k-value for the classifier.

Run #2: I have developed a set of linear classifiers using a bag-of-visual-words feature based on fixed size densely-sampled pixel patches. I am starting with 8x8 patches, sampled every 4 pixels in the x and y directions. A sample of these is clustered using K-Means to learn a vocabulary (around 500 clusters to start).

Run #3: For this run, I have switched to python. I used an aggreation of an CNN and a simpel Neural Network to form a final prediction.