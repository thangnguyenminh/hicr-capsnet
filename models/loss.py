from keras import backend as K

gamma = 0.5
m_pos = 0.9 
m_neg = 0.1

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., m_pos - y_pred)) + \
        gamma * (1 - y_true) * K.square(K.maximum(0., y_pred - m_neg))

    return K.mean(K.sum(L, 1))