'''
Utility funcions for iso-response project

Author: Santiago Cadena & Dylan Paiton
Last Update: Dec 2019
'''


def load_model():
    '''
    Returns the cnn model trained to predict responses that performed best in Cadena et.al PLosCB 2019
    '''
    data_dict = Dataset.get_clean_data()
    data = MonkeyDataset(data_dict, seed=1000, train_frac=0.8 ,subsample=2, crop = 30)
    model = ConvNet(data, log_dir='cnn-weights', log_hash='7fdb18e061', obs_noise_model='poisson')
    model.build(filter_sizes=[13, 3, 3],
              out_channels=[32, 32, 32],
              strides=[1, 1, 1],
              paddings=['VALID', 'SAME', 'SAME'],
              smooth_weights=[0.0003, 0, 0],
              sparse_weights=[0.0, 0.00025, 0.00025],
              readout_sparse_weight= 0.0002,
              output_nonlin_smooth_weight = 0)

    model.load_best()
    return model


def get_activations(model, images):
    '''
    Returns the activations from a model for given input images
    Args:
        :model:  an object from the ConvNet class
        :images: an array with NumImages x W x H
    Output:
        :activations: a vector of length #neurons
    '''
    
    if len(images.shape) == 3:
        images = images[:,:,:,np.newaxis]
    
    activations = model.prediction.eval(session=model.session,
                                        feed_dict={model.images: images, 
                                                   model.is_training: False})
    
    return activations


def get_activations_cell(model, images, neuron):
    '''
    Returns the activations from a model for given input images
    Args:
        :model:  an object from the ConvNet class
        :images: an array with NumImages x W x H
        :neuron: int that points to the neuron index
    Output:
        :activations: a vector of length #neurons
    '''
    
    if len(images.shape) == 3:
        images = images[:,:,:, np.newaxis]
    
    activations = model.prediction[:, neuron].eval(session=model.session,
                                        feed_dict={model.images: images, 
                                                   model.is_training: False})
    
    return activations
