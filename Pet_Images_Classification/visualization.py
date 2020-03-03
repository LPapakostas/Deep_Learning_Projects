import matplotlib.pyplot as plt
import random
import os

def plot_samples(file_path,idf):
    # Get list of file names in file path 
    sample_images = os.listdir(file_path) 

    # Prepare a 3x3 subplot for different <id> images
    _,ax = plt.subplots(3,3,figsize =(20,10))
    for idx,img in enumerate(random.sample(sample_images,9)):
        img_read = plt.imread(file_path+'/'+img)
        # img_read.shape 1,2 show pixels
        ax[int(idx/3),idx%3].imshow(img_read)
        ax[int(idx/3),idx%3].axis('off')
        ax[int(idx/3),idx%3].set_title(idf+img)
    plt.show()

def plot_on_grid(test_v_set,plot_idx,title,img_size):
    fig,ax = plt.subplots(3,3,figsize=(20,10))
    for i,idx in enumerate(random.sample(plot_idx,9)):
        img = (test_v_set.__getitem__(idx)[0]).reshape(img_size,img_size,3)
        #img_read = plt.imread(test_v_set+'/'+str(idx))
        ax[int(i/3), i%3].imshow(img)
        ax[int(i/3), i%3].axis('off')
    fig.suptitle(title)
    plt.show()
