from utils import*
import random

def generate_noise(image, noise_type):
    if noise_type == "gaussian_noise":

        gaussian_noise = np.random.normal(0,10,(image.shape[0], image.shape[1]))
        save_image("Gaussian random noise.jpg", gaussian_noise)
    
        return gaussian_noise

    if noise_type == "uniform_noise":

        uniform_noise =  np.random.uniform(0,50,(image.shape[0], image.shape[1]))
        save_image("Uniform random noise.jpg", uniform_noise)

        return uniform_noise

###################################################################
# Add salt and pepper noise to image , prob: Probability of the noise
def sp_noise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

###############################################################
# adding noisy image to original image gaussian & uniform
def image_plus_noise(image, noise_image):

    noisy_image = np.add(image, noise_image)
    return noisy_image
################################################################33
