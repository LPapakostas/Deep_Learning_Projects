import time

def plot_mnist_digits(X,Y):
    import matplotlib.pyplot as plt
    # Plot all MNIST digits for 0 to 9
    _,((ax1,ax2,ax3,ax4,ax5) , (ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5,figsize = (10,5))
    ax_lst = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]

    for idx,ax in enumerate(ax_lst):
        for i in range(1000):
            if( Y[i] == idx):
                ax.imshow(X[i],cmap ='gray')
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                break
    plt.tight_layout()
    plt.show()

def autoencoder_mnist():
    #Supress FutureWarning
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from keras.datasets import mnist
    import matplotlib.pyplot as plt
	
	# Loading MNIST Dataset and separating it into training and testing set
    training_set,testing_set = mnist.load_data()
    X_train,Y_train = training_set
    X_test,Y_test = testing_set
	
	# Show all digits from 0-9 
    plot_mnist_digits(X_test,Y_test)

    # Data Preprocessing
    # Reshaping mnist images into 2D vectors 
    X_train_reshaped = X_train.reshape( [X_train.shape[0],X_train.shape[1]*X_train.shape[2]] )
    X_test_reshaped = X_test.reshape( [X_test.shape[0],X_test.shape[1]*X_test.shape[2]] )
    # Scaling from [0,255] to [0,1]
    X_train_reshaped = X_train_reshaped/255
    X_test_reshaped = X_test_reshaped/255
    
    input_size = X_train.shape[1]*X_train.shape[2] # 784
    
	# Building Autoencoder Function with one 
    def create_basic_autoencoder(hidden_layer_size,input_size = (784,)):
        # Create an autoencoder // Function for those
        from keras.models import Sequential
        from keras.layers import Dense

        model = Sequential()
        model.add( Dense(units = hidden_layer_size,input_shape = input_size,activation='relu') )
        model.add( Dense(units = input_size[0],activation = 'sigmoid') )
        return model
    
    # Create and train autoencoders with 1,2,8,32 hidden_layer_size and train them using 
	# <X_test_reshaped> vector
    model1 = create_basic_autoencoder(hidden_layer_size = 1)
    model1.compile(optimizer = 'adam',loss = 'mean_squared_error')
    model1.fit(X_train_reshaped,X_train_reshaped,epochs = 10,verbose = 0)

    model2 = create_basic_autoencoder(hidden_layer_size = 2)
    model2.compile(optimizer = 'adam',loss = 'mean_squared_error')
    model2.fit(X_train_reshaped,X_train_reshaped,epochs = 10,verbose = 1)

    model8 = create_basic_autoencoder(hidden_layer_size = 8)
    model8.compile(optimizer = 'adam',loss = 'mean_squared_error')
    model8.fit(X_train_reshaped,X_train_reshaped,epochs = 10,verbose = 0)

    model32 = create_basic_autoencoder(hidden_layer_size = 32)
    model32.compile(optimizer = 'adam',loss = 'mean_squared_error')
    model32.fit(X_train_reshaped,X_train_reshaped,epochs = 10,verbose = 1)

    output1 = model1.predict(X_test_reshaped)
    output2 = model2.predict(X_test_reshaped)
    output8 = model8.predict(X_test_reshaped)
    output32 = model32.predict(X_test_reshaped)
	
	# Choose 5 random images and plotting all of previous autoencoders output
	# in correspondance with original digit images
    n_of_rsamples = 5
    _,axes = plt.subplots(5,5,figsize=(15,15))
    import random
    random_images = random.sample(range(output1.shape[0]),n_of_rsamples)
    outputs = [X_test,output1,output2,output8,output32]

    for row_num,row in enumerate(axes):
        for col_num,ax in enumerate(row):
            ax.imshow(outputs[row_num][random_images[col_num]].reshape((28,28)),cmap = 'gray')
            # row_num if else for label
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    
    # Use autoencoders for image denoising 
    
    # Put Gausian noise into training and testing input data
    import numpy as np

    X_train_noisy = X_train_reshaped + np.random.normal(0,0.5,size=X_train_reshaped.shape)
    X_test_noisy = X_test_reshaped + np.random.normal(0,0.5,size=X_test_reshaped.shape)

    # Chopping noisy input into [0,1] to normalize images
    X_train_noisy = np.clip(X_train_noisy,a_min =0,a_max=1)
    X_test_noisy = np.clip(X_test_noisy,a_min=0,a_max=1)
    
    # Create denoising autoencoder with 16 hidden layers and train our model
    denoise_model = create_basic_autoencoder(hidden_layer_size = 16)
    denoise_model.compile(optimizer = 'adam',loss = 'mean_squared_error')
    denoise_model.fit(X_train_noisy,X_train_reshaped,epochs = 10)

    outputn = denoise_model.predict(X_test_noisy)
	
	# Chosse 5 random images indeces and plot original images, original images
	# with presence of noise and denoising autoencoder output 
    _,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),   
    (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(20,15))
    top_axis = [ax1,ax2,ax3,ax4,ax5]
    mid_axis = [ax6,ax7,ax8,ax9,ax10]
    bottom_axis = [ax11,ax12,ax13,ax14,ax15]

    import random
    n_of_rsamples_noise = 5
    random_images_noisy = random.sample(range(outputn.shape[0]),n_of_rsamples_noise)

    # Original Images
    for idx,ax in enumerate(top_axis):
        ax.imshow(X_test_reshaped[random_images_noisy[idx]].reshape(28,28),cmap = 'gray')
        if (idx == 0) : ax.set_ylabel("Original \n Images",size = 10)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Input images with added noise
    for idx,ax in enumerate(mid_axis):
        ax.imshow(X_test_noisy[random_images_noisy[idx]].reshape(28,28),cmap = 'gray')
        if (idx == 0) : ax.set_ylabel("Input Images \n with noise addition",size = 10)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Output Images from the autoencoder
    for idx,ax in enumerate(bottom_axis):
        if (idx == 0) : ax.set_ylabel("Denoised \n Output",size = 10)
        ax.imshow(outputn[random_images_noisy[idx]].reshape(28,28),cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
     # Create a Deep Convolutional denoising autoencoder
    input_s = int(X_train.shape[1])
    from keras.models import Sequential
    from keras.layers import Conv2D

    denoise_conv = Sequential()

    # Encoder layer
    denoise_conv.add( Conv2D(filters = 16,kernel_size = (3,3),padding = 'same',
                    activation = 'relu', input_shape = (input_s,input_s,1)) )
    denoise_conv.add( Conv2D(filters = 8,kernel_size = (3,3),padding = 'same',
                    activation = 'relu'))
    
    # Decoder layer 
    denoise_conv.add( Conv2D(filters = 8,kernel_size = (3,3),padding = 'same',
                    activation = 'relu'))
    denoise_conv.add( Conv2D(filters = 16,kernel_size = (3,3),padding = 'same',
                    activation = 'relu'))
    
    # Output layer
    denoise_conv.add( Conv2D(filters = 1,kernel_size = (3,3),padding = 'same',
                    activation = 'sigmoid'))
    denoise_conv.summary()

    # Model training 
    n_of_samples = 60_000
    denoise_conv.compile(optimizer = 'adam',loss = 'binary_crossentropy')
    denoise_conv.fit(X_train_noisy.reshape(n_of_samples,input_s,input_s,1),
                    X_train_reshaped.reshape(n_of_samples,input_s,input_s,1),
                    epochs = 10)

    n_of_test_samples = 10_000
    output_conv = denoise_conv.predict(X_test_noisy.reshape(n_of_test_samples,input_s,input_s,1))

	# Chosse 5 random images indeces and plot original images, original images
	# with presence of noise and denoising autoencoder output 
    _,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),   
    (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3,5,figsize=(20,13))
    top_axis = [ax1,ax2,ax3,ax4,ax5]
    mid_axis = [ax6,ax7,ax8,ax9,ax10]
    bottom_axis = [ax11,ax12,ax13,ax14,ax15]

    import random
    n_of_rsamples_conv = 5
    random_images_noisy_conv = random.sample(range(output_conv.shape[0]),n_of_rsamples_conv)

    # Original Images
    for idx,ax in enumerate(top_axis):
        ax.imshow(X_test_reshaped[random_images_noisy_conv[idx]].reshape(28,28),cmap = 'gray')
        if (idx == 0) : ax.set_ylabel("Original \n Images",size = 10)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Input images with added noise
    for idx,ax in enumerate(mid_axis):
        ax.imshow(X_test_noisy[random_images_noisy_conv[idx]].reshape(28,28),cmap = 'gray')
        if (idx == 0) : ax.set_ylabel("Input Images \n with noise addition",size = 10)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Output Images from the autoencoder
    for idx,ax in enumerate(bottom_axis):
        if (idx == 0) : ax.set_ylabel("Denoised \n Output",size = 10)
        ax.imshow(output_conv[random_images_noisy_conv[idx]].reshape(28,28),cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
if (__name__ == '__main__'):
    start = time.time()
    autoencoder_mnist()
    print("%.2f" % ((time.time()-start)/60) )
