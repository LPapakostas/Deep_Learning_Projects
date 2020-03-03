import time

def main():
    from matplotlib import pyplot as plt
    import os
    import random
    import piexif
    from visualization import plot_samples
    import warnings
    #Supress FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import model_from_json
        
    source_folder = os.getcwd() + '/PetImages/'

    # Running only once in order to fix data folders 
    # from utilization import train_test_split
    #train_test_split(source_folder)
    
    cat_file_path = source_folder + r'Cat'
    dog_file_path = source_folder + r'Dog'

    plot_samples(cat_file_path,'Cat')
    plot_samples(dog_file_path,'Dog')
    
    # Each argument control how much of a modification is done
    # to an existing image
    # See how ImageDataGenerator works
    '''
    image_generator = ImageDataGenerator(rotation_range = 30,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
    
    _,ax = plt.subplots(2,3,figsize=(20,10))
    all_images = []

    dog_augmented_images = os.listdir(dog_file_path)
    random_img = random.sample(dog_augmented_images,1)[0]
    random_img = plt.imread(dog_train_file_path + '/'+random_img)
    all_images.append(random_img)

    random_img = random_img.reshape((1,) + random_img.shape)
    sample_augmented_images = image_generator.flow(random_img)

    # Plot a Dog image with different characteristics
    for _ in range(2*3-1):
        augmented_imgs = sample_augmented_images.next()
        for img in augmented_imgs:
            all_images.append(img.astype('uint8'))

    for idx,img in enumerate(all_images):
        ax[int(idx/3),idx%3].imshow(img)
        ax[int(idx/3),idx%3].axis('off')
        if (idx == 0):
            ax[int(idx/3),idx%3].set_title('Original Image')
        else:
            ax[int(idx/3),idx%3].set_title('Augmented Image {}'.format(idx))

    plt.show()
    '''

    INPUT_SIZE = 128
    BATCH_SIZE = 16
    ########
    model_simple_filename = 'model_simple.json'
    weights_filename = 'model_simple.h5'
    ####
    load_flag = (os.path.exists(model_simple_filename) and os.path.exists(weights_filename) )
    
    if (load_flag):
        # The model weights with pevious training sets 
        # load json and create model
        json_file = open(model_simple_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_simple = model_from_json(loaded_model_json)
        # load weights into new model
        model_simple.load_weights(weights_filename)
        print("Loaded model from disk")
        model_simple.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])
        model_simple.summary()
    else:  # Create neural network from the start
        # Model HyperParameters
        FILTER_SIZE = 3
        NUM_FILTERS = 32 
        MAXPOOL_SIZE = 2
        SAMPLES_LENGTH = 20000 # cat + dog train images
        STEPS_PER_EPOCH = SAMPLES_LENGTH//BATCH_SIZE
        EPOCHS = 10
     
        # Building a simple CNN model
        from keras.models import Sequential
        from keras.layers import Conv2D,MaxPooling2D
        from keras.layers import Dropout,Flatten,Dense
        
        model_simple = Sequential()
        # convolutional and max pooling layers
        model_simple.add(Conv2D(NUM_FILTERS,(FILTER_SIZE,FILTER_SIZE),
                    input_shape = (INPUT_SIZE,INPUT_SIZE,3),
                    activation = 'relu'))
        model_simple.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE,MAXPOOL_SIZE)))
        model_simple.add(Conv2D(NUM_FILTERS,(FILTER_SIZE,FILTER_SIZE),
                    input_shape = (INPUT_SIZE,INPUT_SIZE,3),
                    activation = 'relu'))
        model_simple.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE,MAXPOOL_SIZE)))
        # we need to add a Flatten layer in order our input matches Dense layers
        model_simple.add(Flatten())
        model_simple.add(Dense(units = 128, activation = 'relu' ))
        # set 50% of weights into 0 to reduce overfitting
        model_simple.add(Dropout(0.5))
        # binary classification
        model_simple.add(Dense(units = 1 , activation = 'sigmoid' ))
        model_simple.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])
        model_simple.summary()

        training_data_generator = ImageDataGenerator(rescale= 1./255)
        training_set = training_data_generator.flow_from_directory(os.getcwd()+r'/PetImages/Train/',
                                 target_size = (INPUT_SIZE,INPUT_SIZE) , batch_size = BATCH_SIZE,   
                                  class_mode = 'binary')
    
        model_simple.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, 
                                epochs = EPOCHS, verbose = 1)     

        # save model and weight in order to not run again
        # serialize model to JSON
        model_simple_json = model_simple.to_json()
        with open("model_simple.json", "w") as json_file:
            json_file.write(model_simple_json)
        # serialize weights to HDF5 --> put lib on .yaml
        model_simple.save_weights("model_simple.h5")
        print("Saved model to disk")
    
    # Create a test data to validate accuracy
    testing_data_generator = ImageDataGenerator(rescale = 1./255)
    testing_set = testing_data_generator.flow_from_directory(os.getcwd()+r'/PetImages/Test/',
                                 target_size = (INPUT_SIZE,INPUT_SIZE) , batch_size = BATCH_SIZE,   
                                  class_mode = 'binary')
    score = model_simple.evaluate_generator(testing_set, steps = len(testing_set))
    
    for _ in range(2):print(" ")
    # Show accuracy and loss metrics
    for idx,metric in enumerate(model_simple.metrics_names):
        print("{}: {}".format(metric,score[idx]))
    
    
    # Generate test for data visualization
    testing_data_generator = ImageDataGenerator(rescale = 1./255)
    test_visual_set = testing_data_generator.flow_from_directory(os.getcwd()+r'/PetImages/Test/',
                                    target_size = (INPUT_SIZE,INPUT_SIZE),
                                    batch_size = 1,class_mode = 'binary')
    
    # Create image evaluation categories
    strongly_right_p_idx = []
    strongly_wrong_p_idx = []
    weakly_wrong_p_idx = []

    for i in range(test_visual_set.__len__()):
        img = (test_visual_set.__getitem__(i))[0]
        pred_prob = (model_simple.predict(img))[0][0]
        pred_label = int(pred_prob > 0.5)
        actual_label = int((test_visual_set.__getitem__(i))[1][0])   
        if ( (pred_label == actual_label) and ( (pred_prob > 0.8) or (pred_prob < 0.2)  ) ):
            strongly_right_p_idx.append(i)
        elif ( (pred_label != actual_label) and ( (pred_prob > 0.8) or (pred_prob < 0.2) ) ):
            strongly_wrong_p_idx.append(i)
        elif ( (pred_label != actual_label) and ( (pred_prob > 0.4) or (pred_prob < 0.6) ) ):
            weakly_wrong_p_idx.append(i)
        x,y,z  = len(strongly_right_p_idx),len(strongly_wrong_p_idx),len(weakly_wrong_p_idx)
        if (all([x,y,z]) >= 9): break

    from visualization import plot_on_grid

    # NEED TO PUT TITLES 
    
    plot_on_grid(test_visual_set,strongly_right_p_idx,"Strong Right Predictions",INPUT_SIZE)
    plot_on_grid(test_visual_set,strongly_wrong_p_idx,"Strong Wrong Predictions",INPUT_SIZE)
    plot_on_grid(test_visual_set,weakly_wrong_p_idx,"Weakly Wrong Predictions",INPUT_SIZE)
    

if (__name__ == '__main__'):
    start = time.time()
    main()
    print(" ")
    print( "%.2f minutes" %((time.time()-start)/60) )