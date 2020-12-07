import cv2
import numpy as np
import matplotlib.pyplot as plt



imgname = 'image01.jpeg'

def GenTarget_Hist(input_hist):
	l_2 = int(len(input_hist)/2)
	target_hist = input_hist.copy()
	Sum = 0
	for i in range(0, l_2):
		Sum = Sum+input_hist[i]
	h = 2*Sum/l_2
	
	for i in range(0, l_2):
		target_hist[i][0] = h*i/l_2
	return target_hist

def intensity(img):
	R = np.array([img.T[0]]).T
	G = np.array([img.T[1]]).T
	B = np.array([img.T[2]]).T
	I_float = np.around(R/3+G/3+B/3)
	I = np.zeros(shape = I_float.shape, dtype = 'uint8')
	for i, r in enumerate(I_float):
		for j, e in enumerate(r):
			I[i][j][0] = int(e[0])
	return I

def calAlpha(G, I_mean):
	alpha = np.zeros(shape = (G.shape[0], G.shape[1], 1))
	for i in range(0,G.shape[0]):
		for j in range(0,G.shape[1]):
			t = np.abs(G[i][j]-I_mean)/I_mean
			alpha[i][j] = [t] if 1>t else [1]
	return alpha


def main():
	img = cv2.imread('input/'+imgname)
	print(img.shape)	
	
	print('Calculate intensity')	
	# Calculate intensity of img
	I = intensity(img)

	print('Calculate alpha')	
	# Calculate alpha
	G = cv2.GaussianBlur(I, (5,5), 0)
	I_mean = np.mean(I)
	alpha = calAlpha(G, I_mean)
	
	print('Generate Histogram')
	# Calculate the iinput images' histogram
	input_hist = cv2.calcHist([I], [0], None, [256], [0,256])

	#Normalize the input histogram
	total = sum(input_hist)
	input_hist /= total
	
	# Calculate the cumulateive input histogram
	cum_input_hist = []
	cum = 0.0
	for i in range(0, len(input_hist)):
		cum+=input_hist[i][0]
		cum_input_hist.append(cum)


	print('Generate target Histogram and lookup table')
	#Generate target_histograms
	target_hist = GenTarget_Hist(input_hist)
	
	# Calculate the cumulative target histogram
	cum_target_hist = []
	cum = 0.0
	for j in range(0, len(target_hist)):
		cum+=target_hist[j][0]
		cum_target_hist.append(cum)
	

	# Obtain the mapping from the input hist to target hist
	lookup = {}
	for i in range(len(cum_input_hist)):
		min_val = abs(cum_target_hist[0] - cum_input_hist[i])
		min_j=0
		
		for j in range(1, len(cum_target_hist)):
			val = abs(cum_target_hist[j] - cum_input_hist[i])
			if (val < min_val):
				min_val = val
				min_j = j

		lookup[i] = min_j

	print('Intensity calcuation')
	# Create the transformed image using the img's pixel values and the lookup table
	trans_img = img.copy()
	I_HS = I.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			I_HS[i][j][0] = lookup[I[i][j][0]]
	
	I_ = alpha*I_HS+(1-alpha)*I
	
	for i, r in enumerate(I):
		for j, e in enumerate(r):
			if e[0] ==0: 
				I[i][j][0] = 1
	
	ratio = I_/I

	print('Color image convertion')
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			t = ratio[i][j][0]*img[i][j]
			r = 255/t[np.argmax(t)] if np.max(t)>255 else 1	
			for k in range(len(t)):
				t[k] = int(t[k]*r)
			trans_img[i][j] = t
	
	# Write the transformed image to a jpeg file
	cv2.imwrite('output/'+imgname, trans_img)

'''
	plt.subplot(2, 3, 1)
	plt.title('Original hist')
	plt.plot(input_hist)

	plt.subplot(2, 3, 2)
	plt.title('Comulated Original hist')
	plt.plot(cum_input_hist)


	plt.subplot(2, 3, 3)
	plt.title('Target pdf')
	plt.plot(target_hist)
		
	plt.subplot(2, 3, 4)
	plt.title('Comulated Target pdf')
	plt.plot(cum_target_hist)
	
	plt.show()
'''
	
if __name__ == '__main__':
	main()
