import os
import numpy as np
import tensorflow as tf
from models import *
from losses import *
from utils import *
import matplotlib.image as mpim
from sys import stdout
import pickle

# class weights
class_weights = np.load('/content/drive/MyDrive/Fitria/vox2vox_fix_dropout(0,2)/weight_coba.npy').astype('float32')
print('Class_weights:', class_weights, 'Sum:', class_weights.sum())

path =  '/content/drive/MyDrive/Fitria/Result2/vox2vox/linear'
if os.path.exists(path)==False:
    os.mkdir(path)
    os.mkdir(path + '/best')


# Models
G = Generator()
D = Discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(image, target, alpha):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = G(image, training=True)

        disc_real_output = D([image, target], training=True)
        disc_fake_output = D([image, gen_output], training=True)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
        
        gen_loss, iou_loss, dice_loss, disc_loss_gen = generator_loss(target, 
                                                    gen_output, 
                                                    disc_fake_output, 
                                                    class_weights, 
                                                    alpha)

    generator_gradients = gen_tape.gradient(gen_loss, G.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, D.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D.trainable_variables))
        
    return gen_loss, iou_loss, dice_loss, disc_loss_gen
        
@tf.function
def test_step(image, target, alpha):
    gen_output = G(image, training=False)

    disc_real_output = D([image, target], training=False)
    disc_fake_output = D([image, gen_output], training=False)
    disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

    gen_loss, iou_loss, dice_loss, disc_loss_gen = generator_loss(target, 
                                                gen_output, 
                                                disc_fake_output, 
                                                class_weights, 
                                                alpha)
        
    return gen_loss, iou_loss, dice_loss, disc_loss_gen

def fit(train_gen, valid_gen, alpha, epochs, continue_training=False):
    
    Nt = len(train_gen)
    
    if continue_training:
        # Load the training losses history
        history = pickle.load(open(path + "/history.pkl", 'rb'))
        # Get the number of epoch with the best valid loss
        last_epoch = history['last_epoch']
        prev_loss = history['valid'][last_epoch-1]['v2v']
        # Truncate the history from epoch with the best valid loss
        history['train'] = history['train'][:last_epoch]
        history['valid'] = history['valid'][:last_epoch]
        print('Epoch of the last saved model: {}. Validation loss: {:.4f}\n'.format(
            last_epoch, prev_loss))
        # Load last saved models and losses
        if last_epoch == history['best_epoch']:
            G.load_weights(path + '/best/Generator.h5')
            D.load_weights(path + '/best/Discriminator.h5')
        else:
            G.load_weights(path + '/Generator.h5')
            D.load_weights(path + '/Discriminator.h5')
    else: # Start training from scratch
        history = {'train': [], 'valid': []}
        last_epoch = 0
        prev_loss = np.inf

    
    # DEFINE LOSSES
    epoch_v2v_loss = tf.keras.metrics.Mean()
    epoch_dice_loss = tf.keras.metrics.Mean()
    epoch_iou_loss = tf.keras.metrics.Mean()
    epoch_disc_loss = tf.keras.metrics.Mean()
    epoch_v2v_loss_val = tf.keras.metrics.Mean()
    epoch_dice_loss_val = tf.keras.metrics.Mean()
    epoch_iou_loss_val = tf.keras.metrics.Mean()
    epoch_disc_loss_val = tf.keras.metrics.Mean()
    # Losses name
    loss_name = ['v2v','iou' ,'dice', 'disc']
    
    for e in range(last_epoch+1, last_epoch+epochs+1):
        
        print('Epoch {}/{}'.format(e, last_epoch+epochs))
        b = 0
        # TRAIN AND CALCULATE TRAIN LOSS
        for Xb, yb in train_gen:
            b += 1
            
            losses = train_step(Xb, yb, alpha)
            epoch_v2v_loss.update_state(losses[0])
            epoch_dice_loss.update_state(losses[1])
            epoch_iou_loss.update_state(losses[2])
            epoch_disc_loss.update_state(losses[3])

            stdout.write(
                '\rBatch: {}/{} - loss: {:.4f} - iou_loss: {:.4f} - dice_loss: {:.4f} - disc_loss: {:.4f}'
                .format(b, Nt, 
                    epoch_v2v_loss.result(), 
                    epoch_dice_loss.result(), 
                    #epoch_hd.result(), 
                    epoch_iou_loss.result(),
                    epoch_disc_loss.result()))
            stdout.flush()
        
        result = [epoch_v2v_loss.result(), epoch_iou_loss.result(), epoch_dice_loss.result(), 
                  epoch_disc_loss.result()]
        history['train'].append(
            {loss_name[i]: float(result[i].numpy()) for i in range(len(result))}
        )
        
        # CALCULATE VALID LOSS
        for Xb, yb in valid_gen:
            losses_val = test_step(Xb, yb, alpha)
            epoch_v2v_loss_val.update_state(losses_val[0])
            epoch_dice_loss_val.update_state(losses_val[1])
            epoch_iou_loss_val.update_state(losses[2])
            epoch_disc_loss_val.update_state(losses_val[3])
            
        stdout.write(
            '\n               loss_val: {:.4f} - iou_loss_val: {:.4f} - dice_loss_val: {:.4f} - disc_loss_val: {:.4f}'
                .format(epoch_v2v_loss_val.result(), 
                        epoch_dice_loss_val.result(),
                        epoch_iou_loss_val.result(),
                        epoch_disc_loss_val.result()))
        stdout.flush()
        
        result = [epoch_v2v_loss_val.result(), epoch_iou_loss_val.result(), epoch_dice_loss_val.result(),
                  epoch_disc_loss_val.result()]
        history['valid'].append(
            {loss_name[i]: float(result[i].numpy()) for i in range(len(result))}
        )

        # SAME PREDICTED IMAGE AT THE END OF EACH EPOCH
        y_pred = G.predict(Xb)
        y_true = np.argmax(yb, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        canvas = np.zeros((128, 128*3))
        idx = np.random.randint(len(Xb))
        
        x = Xb[idx,:,:,64,0] 
        canvas[0:128, 0:128] = (x - np.min(x))/(np.max(x)-np.min(x)+1e-6)
        canvas[0:128, 128:2*128] = y_true[idx,:,:,64]/3
        canvas[0:128, 2*128:3*128] = y_pred[idx,:,:,64]/3
        
        fname = (path + '/pred@epoch_{:03d}.png').format(e)
        mpim.imsave(fname, canvas, cmap='gray')
        
        # SAVE MODELS
        # Save the best model
        if epoch_v2v_loss_val.result() < prev_loss:    
            G.save_weights(path + '/best/Generator.h5', overwrite=True) 
            D.save_weights(path + '/best/Discriminator.h5', overwrite=True)
            print("\nValidation loss decreased from {:.4f} to {:.4f}. Models' weights are now saved.\n".format(prev_loss, epoch_v2v_loss_val.result()))
            prev_loss = epoch_v2v_loss_val.result()
            history['best_epoch'] = e
            history['last_epoch'] = e
        else:
            print("\nValidation loss did not decrese from {:.4f}.\n".format(prev_loss))
            
            # In case the model does not get better,
            if e % 5 == 0: # save every 5 epochs, start from e = 5
                G.save_weights(path + '/Generator.h5', overwrite=True) 
                D.save_weights(path + '/Discriminator.h5', overwrite=True)
                history['last_epoch'] = e
        
        # SAVE THE LOSSES TO A JSON FILE
        pickle.dump(history, open(path + "/history.pkl", 'wb'))
        
        # resets losses states
        epoch_v2v_loss.reset_states()
        epoch_dice_loss.reset_states()
        epoch_iou_loss.reset_states()
        epoch_disc_loss.reset_states()
        epoch_v2v_loss_val.reset_states()
        epoch_dice_loss_val.reset_states()
        epoch_iou_loss_val.reset_states()
        epoch_disc_loss_val.reset_states()
        
        del Xb, yb, canvas, y_pred, y_true, idx
        
    return history
