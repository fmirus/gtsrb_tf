from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import csv
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

class GTSRB_Classifier(object):
    def __init__(self):
        pass

    def create_classifier(self, model_dir):

        self.classifier_ = tf.estimator.Estimator(model_fn=self.cnn_model_fn, model_dir=model_dir)
        self.initialized = True

    def train(self, train_data, train_labels, num_steps, batch_size, num_epochs=500, b_shuffle=True):
        # Train the model
        if self.initialized:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data},
                y=train_labels,
                batch_size=batch_size,
                num_epochs=num_epochs,
                shuffle=b_shuffle)
            self.classifier_.train(input_fn=train_input_fn, steps=num_steps)


    def evaluate(self, eval_data, eval_labels, num_epochs, b_shuffle=False):
        if self.initialized:
            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=num_epochs,
                shuffle=b_shuffle)
            eval_results = self.classifier_.evaluate(input_fn=eval_input_fn)
            print(eval_results)

    def predict(self, eval_data, eval_labels, num_epochs, b_shuffle=False):
        if self.initialized:
            # Predict the model and print results
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=num_epochs,
                shuffle=b_shuffle)
            predictions = self.classifier_.predict(input_fn=predict_input_fn)
            return predictions

    def detailed_evaluation(self, eval_data, eval_labels, num_epochs, class_names, b_shuffle=False):
        if self.initialized:
            # Predict the model and print results
            predictions = self.predict(eval_data, eval_labels, num_epochs, b_shuffle)
            # here we assume that every class is contained in the evaluation data set at least once
            nb_classes = len(class_names)

            false_classifications = []
            correctly_classified = 0
            correctly_classified_per_class = np.zeros(nb_classes)
            correctly_classified_per_class_perc = np.zeros(nb_classes)
            total_num_per_class = np.zeros(nb_classes)
            num = len(eval_labels)

            for ind, p in enumerate(predictions):
                total_num_per_class[eval_labels[ind]] += 1
                if p['classes'] == eval_labels[ind]:
                    correctly_classified += 1
                    correctly_classified_per_class[eval_labels[ind]] += 1
                else:
                    false_classifications.append([ind, p['classes'], eval_labels[ind], p['probabilities'][p['classes']]])

            print("total classification performance: %1.2f percent"%(float(correctly_classified/float(num))*100))
            for ind in range(len(correctly_classified_per_class)):
                print("classification performance for class %i (%s): %1.2f"%(ind, class_names[ind], float(correctly_classified_per_class[ind]/float(total_num_per_class[ind]))*100))
                correctly_classified_per_class_perc[ind] = (correctly_classified_per_class[ind]/float(total_num_per_class[ind]))*100

            # write results to file
            with open("results.csv", 'w', newline='') as f:
                writer = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['total classification performance: %1.2f'%(float(correctly_classified/float(num))*100)])
                for ind in range(len(correctly_classified_per_class)):
                    writer.writerow(["classification performance for class %i (%s): %1.2f"%(ind, class_names[ind], float(correctly_classified_per_class[ind]/float(total_num_per_class[ind]))*100)])
                writer.writerow(['false classification percentage: %1.2f'%(float((float(num)-correctly_classified)/float(num))*100)])
                writer.writerow(['number of correct classification: %i'%(int(correctly_classified))])
                writer.writerow(['total number of images: %i'%(int(num))])
                writer.writerow(['image index', 'classified id', 'correct class id'])
                writer.writerows(false_classifications)

            return false_classifications, correctly_classified_per_class_perc

    @staticmethod
    def cnn_model_fn(features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 60, 60, 3])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
                  inputs=input_layer,
                  filters=100,
                  kernel_size=[7, 7],
                  padding="valid",
                  activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
                  inputs=pool1,
                  filters=150,
                  kernel_size=[3, 3],
                  padding="valid",
                  activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Convolutional Layer #3 and Pooling Layer #3
        conv3 = tf.layers.conv2d(
                  inputs=pool2,
                  filters=250,
                  kernel_size=[3, 3],
                  padding="valid",
                  activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        # Dense Layer
        dropout1 = tf.layers.dropout(inputs=pool3, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
        pool3_flat = tf.reshape(dropout1, [-1, 5 * 5 * 250])
        dense = tf.layers.dense(inputs=pool3_flat, units=300, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout2, units=43)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
              "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}

        return tf.estimator.EstimatorSpec(
                  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
