from keras.layers import Layer
import numpy
import theano.tensor as T


class MyLayer(Layer):

    def __init__(self,size, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.size = size


    def build(self, input_shape):
        # non ho pesi!.
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x,**kwargs):
        return x[:,-self.size:]

    def compute_output_shape(self, input_shape):
        output_shape = (None,self.size,input_shape[2])
        return output_shape
