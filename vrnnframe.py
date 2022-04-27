import numpy as np
import sklearn.utils
import tensorflow as tf
import time
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer


def preprocess(timestamps, events, dt, t_max=None, dtype=np.float_):
    timestamps = [np.asarray(timestamps_, dtype=dtype)
                  for timestamps_ in timestamps]
    events = [np.asarray(events_, dtype=np.bool) for events_ in events]
    assert len(timestamps) == len(events)
    assert all(timestamps_.shape[0] == events_.shape[0]
               for timestamps_, events_ in zip(timestamps, events))

    n_samples = len(timestamps)
    n_events = events[0].shape[1]
    if t_max is None:
        t_max = max(timestamps_[-1] for timestamps_ in timestamps)

    n_frames = int(np.ceil(t_max / dt))
    x = np.zeros([n_samples, n_frames, n_events], dtype=dtype)
    for i in range(n_samples):
        for j in range(n_events):
            x[i,:,j] = np.histogram(timestamps[i][events[i][:,j]],
                                    bins=n_frames, range=(0, n_frames * dt))[0]

    return x


class Generator:
    def __init__(self, x, batch_size=None, n_steps=None,
                 shuffle=False, dtype=K.floatx()):
        if batch_size is None:
            batch_size = x.shape[0]
        if n_steps is None:
            n_steps = x.shape[1]

        output_types = ['bool', dtype]
        output_shapes = [tf.TensorShape([]),
                         tf.TensorShape([None, None, x.shape[-1]])]

        self.x = x
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.shuffle = shuffle
        self.output_types = tuple(output_types)
        self.output_shapes = tuple(output_shapes)

    def __iter__(self):
        x = self.x
        if self.shuffle:
            x = sklearn.utils.shuffle(x)

        n_batches = int(np.ceil(x.shape[0] / self.batch_size))
        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = (i + 1) * self.batch_size

            n_slices = int(np.ceil(x.shape[1] / self.n_steps))
            for j in range(n_slices):
                slice_start = j * self.n_steps
                slice_end = (j + 1) * self.n_steps
                yield (j == 0), x[batch_start:batch_end, slice_start:slice_end]


class VRNNCell(Layer):
    def __init__(self, n_latent, rnn_cell,
                 x_network=None, encoder_network=None, decoder_network=None,
                 prior_network=None, z_network=None, recurrence_network=None,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_latent = n_latent
        self.rnn_cell = rnn_cell
        self.x_network = x_network
        self.encoder_network = encoder_network
        self.decoder_network = decoder_network
        self.prior_network = prior_network
        self.z_network = z_network
        self.recurrence_network = recurrence_network

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.state_size = rnn_cell.state_size

    def build(self, input_shape):
        batch_size, n_events = input_shape

        if self.x_network is None:
            n_x_hidden = n_events
        else:
            shape = self.x_network.compute_output_shape(input_shape)
            n_x_hidden = shape[-1]

        n_h = self.rnn_cell.units
        if self.encoder_network is None:
            n_encoder_hidden = n_x_hidden + n_h
        else:
            shape = [batch_size, n_x_hidden + n_h]
            shape = self.encoder_network.compute_output_shape(shape)
            n_encoder_hidden = shape[-1]

        if self.decoder_network is None:
            n_decoder_hidden = n_h
        else:
            shape = [batch_size, n_h]
            shape = self.decoder_network.compute_output_shape(shape)
            n_decoder_hidden = shape[-1]

        if self.prior_network is not None:
            n_prior_hidden = n_h
        else:
            shape = self.prior_network.compute_output_shape([batch_size, n_h])
            n_prior_hidden = shape[-1]

        if self.z_network is None:
            n_z_hidden = self.n_latent
        else:
            shape = [batch_size, self.n_latent]
            shape = self.z_network.compute_output_shape(shape)
            n_z_hidden = shape[-1]

        self.kernel_encoder = self.add_weight(
            shape=[n_encoder_hidden, 2 * self.n_latent],
            name='kernel_encoder',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.bias_encoder = self.add_weight(
            shape=[2 * self.n_latent],
            name='bias_encoder',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        self.kernel_decoder = self.add_weight(
            shape=[n_decoder_hidden, n_events],
            name='kernel_event',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.bias_decoder = self.add_weight(
            shape=[n_events],
            name='bias_event',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        self.kernel_prior = self.add_weight(
            shape=[n_prior_hidden, 2 * self.n_latent],
            name='kernel_prior',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.bias_prior = self.add_weight(
            shape=[2 * self.n_latent],
            name='bias_prior',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        super().build(input_shape)

    def encode(self, x_hidden, h):
        kernel_encoder_mean, kernel_encoder_std = tf.split(
            self.kernel_encoder, num_or_size_splits=2, axis=-1
        )
        bias_encoder_mean, bias_encoder_std = tf.split(
            self.bias_encoder, num_or_size_splits=2, axis=-1
        )

        encoder_hidden = tf.concat([x_hidden, h], -1)
        if self.encoder_network is not None:
            encoder_hidden = self.encoder_network(encoder_hidden)

        encoder_mean = tf.matmul(encoder_hidden, kernel_encoder_mean)
        encoder_mean = encoder_mean + bias_encoder_mean
        encoder_std = tf.matmul(encoder_hidden, kernel_encoder_std)
        encoder_std = tf.math.softplus(encoder_std + bias_encoder_std)

        #encoder_std = tf.minimum(encoder_std, 0.5)

        return encoder_mean, encoder_std

    def decode(self, h):
        decoder_hidden = h
        if self.decoder_network is not None:
            decoder_hidden = self.decoder_network(decoder_hidden)

        intensity = tf.matmul(decoder_hidden, self.kernel_decoder)
        intensity = tf.math.softplus(intensity + self.bias_decoder)

        return intensity

    def prior(self, h):
        kernel_prior_mean, kernel_prior_std = tf.split(
            self.kernel_prior, num_or_size_splits=2, axis=-1
        )
        bias_prior_mean, bias_prior_std = tf.split(
            self.bias_prior, num_or_size_splits=2, axis=-1
        )

        prior_hidden = h
        if self.prior_network is not None:
            prior_hidden = self.prior_network(prior_hidden)

        prior_mean = tf.matmul(prior_hidden, kernel_prior_mean)
        prior_mean = prior_mean + bias_prior_mean
        prior_std = tf.matmul(prior_hidden, kernel_prior_std)
        prior_std = tf.math.softplus(prior_std + bias_prior_std)

        return prior_mean, prior_std

    def log_likelihood(self, x, intensity):
        log_likelihood = x * tf.math.log(tf.maximum(intensity, 1e-9))
        log_likelihood -= intensity + tf.math.lgamma(x + 1)

        return log_likelihood

    def recurrence(self, x_hidden, z_hidden, states):
        recurrence_hidden = tf.concat([x_hidden, z_hidden], -1)
        if self.recurrence_network is not None:
            recurrence_hidden = self.recurrence_network(recurrence_hidden)

        _, states = self.rnn_cell(recurrence_hidden, states)
        return states

    def generate_frame(self, intensity):
        x = tf.random.poisson([1], intensity)
        x = tf.reshape(x, tf.shape(intensity))
        #x = tf.stop_gradient(x)

        return x

    def infer(self, inputs, states, training=None):
        h = states[0]

        x_hidden = inputs
        if self.x_network is not None:
            x_hidden = self.x_network(x_hidden)

        encoder_mean, encoder_std = self.encode(x_hidden, h)
        eps = tf.random.normal([tf.shape(encoder_mean)[0], self.n_latent])
        z = z_hidden = encoder_mean + encoder_std * eps
        if self.z_network is not None:
            z_hidden = self.z_network(z_hidden)

        intensity = self.decode(h)
        log_likelihood = self.log_likelihood(inputs, intensity)

        prior_mean, prior_std = self.prior(h)

        outputs = [encoder_mean, encoder_std, z,
                   prior_mean, prior_std,
                   intensity, log_likelihood]
        states = self.recurrence(x_hidden, z_hidden, states)

        return outputs, states

    def generate(self, inputs, states, training=None):
        h = states[0]

        intensity = self.decode(h)
        x = self.generate_frame(intensity)

        x_hidden = x
        if self.x_network is not None:
            x_hidden = self.x_network(x_hidden)

        prior_mean, prior_std = self.prior(h)
        z = z_hidden = inputs #prior_mean + prior_std * inputs
        if self.z_network is not None:
            z_hidden = self.z_network(z_hidden)

        outputs = [intensity, x]
        states = self.recurrence(x_hidden, z_hidden, states)

        return outputs, states

    def get_config(self):
        config = {'n_latent': self.n_latent,
                  'kernel_initializer':
                      tf.keras.initializers.serialize(self.kernel_initializer),
                  'bias_initializer':
                      tf.keras.initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      tf.keras.regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer':
                      tf.keras.regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint':
                      tf.keras.constraints.serialize(self.kernel_constraint),
                  'bias_constraint':
                      tf.keras.constraints.serialize(self.bias_contraint)}

        for arg_name in ['x_network', 'encoder_network', 'decoder_network',
                         'prior_network', 'z_network', 'recurrence_network']:
            arg = getattr(self, arg_name)
            if arg is None:
                config[arg_name] = None
            else:
                config[arg_name] = {'class_name': arg.__class__.__name__,
                                    'config': arg.get_config()}

        config['rnn_cell'] = {'class_name': self.rnn_cell.__class__.__name__,
                              'config': self.rnn_cell.get_config()}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.rnn_cell.get_initial_state(inputs=inputs,
                                               batch_size=batch_size,
                                               dtype=dtype)


class VRNN(tf.keras.layers.RNN):
    def __init__(self, n_latent, rnn_cell,
                 x_network=None, encoder_network=None, decoder_network=None,
                 prior_network=None, z_network=None, recurrence_network=None,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        vrnn_cell = VRNNCell(
            n_latent, rnn_cell,
            x_network=x_network, encoder_network=encoder_network,
            decoder_network=decoder_network,
            prior_network=prior_network, z_network=z_network,
            recurrence_network=recurrence_network,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )
        super().__init__(vrnn_cell,
                         return_sequences=True, return_state=True, **kwargs)

    @property
    def n_latent(self):
        return self.cell.n_latent

    def infer(self, inputs,
              training=None, initial_state=None,
              return_encoder_mean=False, return_encoder_std=False,
              return_z=False,
              return_prior_mean=False, return_prior_std=False,
              return_intensity=False, return_log_likelihood=False):
        self.cell.call = self.cell.infer
        outputs_ = self(inputs,
                        training=training,
                        initial_state=initial_state)

        outputs = []
        if return_encoder_mean:
            outputs.append(outputs_[0])
        if return_encoder_std:
            outputs.append(outputs_[1])
        if return_z:
            outputs.append(outputs_[2])
        if return_prior_mean:
            outputs.append(outputs_[3])
        if return_prior_std:
            outputs.append(outputs_[4])
        if return_intensity:
            outputs.append(outputs_[5])
        if return_log_likelihood:
            outputs.append(outputs_[6])

        states = outputs_[7:]
        if len(states) == 1:
            states = states[0]

        return outputs, states

    def generate(self, inputs,
                 training=None, initial_state=None,
                 return_intensity=False, return_x=False):
        self.cell.call = self.cell.generate
        outputs_ = super().call(inputs,
                                training=training,
                                initial_state=initial_state)

        outputs = []
        if return_intensity:
            outputs.append(outputs_[0])
        if return_x:
            outputs.append(outputs_[1])

        states = outputs_[2:]
        if len(states) == 1:
            states = states[0]

        return outputs, states


class VRNNModel:
    def __init__(self, vrnn, discriminator):
        self.vrnn = vrnn
        self.disc = discriminator

    @property
    def dtype(self):
        return self.vrnn.dtype

    @tf.function(experimental_relax_shapes=True)
    def train_step_disc(self, z, optimizer, metrics=None):
        with tf.GradientTape() as tape:
            y = self.disc(z)
            y_gen = self.disc(tf.random.normal(tf.shape(z)))

            y = tf.maximum(1 - y, 1e-9)
            y_gen = tf.maximum(y_gen, 1e-9)
            loss = -tf.reduce_mean(tf.math.log(y) + tf.math.log(y_gen))

        trainable_variables = self.disc.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        if metrics is not None:
            metrics[-1](loss)

    def discrepancy(self, z1, z2=None):
        n_steps = tf.cast(tf.shape(z1)[1], self.dtype)

        n1 = tf.cast(tf.shape(z1)[0], self.dtype)
        if z2 is None:
            z2_ = z1
            n2 = n1 - 1
        else:
            z2_ = z2
            n2 = tf.cast(tf.shape(z2)[0], self.dtype)

        z1_ = tf.expand_dims(z1, 0)
        z2_ = tf.expand_dims(z2_, 1)

        discrepancy = z1_ - z2_
        discrepancy = tf.reduce_sum(tf.square(discrepancy), axis=-1)
        discrepancy = tf.reduce_sum(discrepancy, axis=-1)

        C = 2 * self.vrnn.n_latent * n_steps
        discrepancy = C / (C + discrepancy)
        discrepancy = tf.reduce_sum(discrepancy) / (n1 * n2)

        return discrepancy

    @tf.function(experimental_relax_shapes=True)
    def train_step_vrnn(self, x, lambda_1, lambda_2, optimizer,
                        initial_state=None, metrics=None):
        with tf.GradientTape() as tape:
            outputs, states = self.vrnn.infer(
                x, training=True, initial_state=initial_state,
                return_z=True, return_prior_mean=True, return_prior_std=True,
                return_log_likelihood=True
            )
            z_tilde, prior_mean, prior_std, log_likelihood = outputs
            #z_tilde = (z_tilde - prior_mean) / tf.maximum(prior_std, 1e-9)
            z = tf.random.normal(tf.shape(z_tilde))

            log_likelihood = tf.reduce_mean(log_likelihood, axis=1)
            log_likelihood = tf.reduce_mean(log_likelihood)

            discrepancy = self.discrepancy(z_tilde)
            discrepancy += self.discrepancy(z)
            discrepancy -= 2 * self.discrepancy(z_tilde, z2=z)

            gan_penalty = -tf.math.log(tf.maximum(self.disc(z_tilde), 1e-9))
            gan_penalty = tf.reduce_mean(gan_penalty)

            loss = (-log_likelihood +
                    lambda_1 * discrepancy + lambda_2 * gan_penalty)

        trainable_variables = self.vrnn.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        if metrics is not None:
            metrics[0](log_likelihood)
            metrics[1](discrepancy)
            metrics[2](gan_penalty)
            metrics[3](loss)

        return states

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, lambda_1, lambda_2, n_disc,
                   vrnn_optimizer, disc_optimizer,
                   initial_state=None, metrics=None):
        [z], states = self.vrnn.infer(x, training=False,
                                      initial_state=initial_state,
                                      return_z=True)
        for i in tf.range(n_disc):
            self.train_step_disc(z, disc_optimizer, metrics=metrics)

        return self.train_step_vrnn(x, lambda_1, lambda_2, vrnn_optimizer,
                                    initial_state=initial_state,
                                    metrics=metrics)

    @tf.function(experimental_relax_shapes=True)
    def _infer(self, x, initial_state=None,
               return_encoder_mean=False, return_encoder_std=False,
               return_z=False,
               return_prior_mean=False, return_prior_std=False,
               return_intensity=False, return_log_likelihood=False):
        outputs, states = self.vrnn.infer(
            x, training=False, initial_state=initial_state,
            return_encoder_mean=return_encoder_mean,
            return_encoder_std=return_encoder_std,
            return_z=return_z,
            return_prior_mean=return_prior_mean,
            return_prior_std=return_prior_std,
            return_intensity=return_intensity,
            return_log_likelihood=return_log_likelihood,
        )

        if not isinstance(states, list):
            states = [states]

        return outputs, states

    @tf.function
    def _generate(self, n_samples, n_steps, initial_state=None,
                  return_intensity=False, return_x=False):
        z = tf.random.normal([n_samples, n_steps, self.vrnn.n_latent])
        outputs, states = self.vrnn.generate(
            z, training=False, initial_state=initial_state,
            return_intensity=return_intensity, return_x=return_x
        )

        if not isinstance(states, list):
            states = [states]

        return outputs, states

    def train(self, x, lambda_1=1, lambda_2=1, n_disc=1,
              batch_size=100, n_steps=None, epochs=1, shuffle=True,
              vrnn_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              disc_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)):
        generator = Generator(x, batch_size=batch_size, n_steps=n_steps,
                              shuffle=shuffle)
        output_types = generator.output_types
        output_shapes = generator.output_shapes
        dataset = tf.data.Dataset.from_generator(lambda: generator,
                                                 output_types,
                                                 output_shapes=output_shapes)

        if not isinstance(lambda_1, tf.Tensor):
            lambda_1 = tf.constant(lambda_1, dtype=self.dtype)
        if not isinstance(lambda_2, tf.Tensor):
            lambda_2 = tf.constant(lambda_2, dtype=self.dtype)
        if not isinstance(n_disc, tf.Tensor):
            n_disc = tf.constant(n_disc, dtype=tf.int32)

        metrics = [tf.keras.metrics.Mean('Log-Likelihood', dtype=self.dtype),
                   tf.keras.metrics.Mean('Discrepancy', dtype=self.dtype),
                   tf.keras.metrics.Mean('GAN-Penalty', dtype=self.dtype),
                   tf.keras.metrics.Mean('VRNN-Loss', dtype=self.dtype),
                   tf.keras.metrics.Mean('Disc-Loss', dtype=self.dtype)]
        for epoch in range(epochs):
            start_time = time.time()
            print('Training epoch {}/{}'.format(epoch + 1, epochs))
            for reset, x in dataset:
                if reset:
                    states = self.vrnn.cell.get_initial_state(
                        batch_size=x.shape[0], dtype=self.dtype
                    )

                states = self.train_step(x, lambda_1, lambda_2, n_disc,
                                         vrnn_optimizer, disc_optimizer,
                                         initial_state=states, metrics=metrics)

            duration = time.time() - start_time
            print('Epoch {}: {:.1f} seconds'.format(epoch + 1, duration))
            for metric in metrics:
                print('{}: {}'.format(metric.name, metric.result()))
                metric.reset_states()

    def infer(self, x, batch_size=100,
              return_encoder_mean=False, return_encoder_std=False,
              return_z=False,
              return_prior_mean=False, return_prior_std=False,
              return_intensity=False, return_log_likelihood=False,
              return_state=False):
        generator = lambda: Generator(x, batch_size=batch_size)
        generator_instance = generator()
        output_types = generator_instance.output_types
        output_shapes = generator_instance.output_shapes
        dataset = tf.data.Dataset.from_generator(generator, output_types,
                                                 output_shapes=output_shapes)

        outputs = []
        states = []
        for reset, x in dataset:
            outputs_, states_ = self._infer(
                x,
                return_encoder_mean=return_encoder_mean,
                return_encoder_std=return_encoder_std,
                return_z=return_z,
                return_prior_mean=return_prior_mean,
                return_prior_std=return_prior_std,
                return_intensity=return_intensity,
                return_log_likelihood=return_log_likelihood
            )

            outputs_ = [output.numpy() for output in outputs_]
            outputs.append(outputs_)
            if return_state:
                states_ = [state.numpy() for state in states_]
                states.append(states_)

        outputs = [np.vstack(output) for output in zip(*outputs)]
        if outputs and len(outputs) == 1:
            outputs = outputs[0]

        states = [np.vstack(state) for state in zip(*states)]
        if states and len(states) == 1:
            states = states[0]

        if len(outputs) > 0 and len(states) > 0:
            return outputs, states
        elif len(outputs):
            return outputs
        elif len(states):
            return states
        else:
            return None

    def generate(self, n_samples, n_steps,
                 initial_state=None, batch_size=1000,
                 return_intensity=False, return_x=False, return_state=False):
        outputs = []
        states = []
        n_samples_ = 0
        while n_samples_ < n_samples:
            batch_size_ = min(batch_size, n_samples - n_samples_)
            if initial_state is None:
                initial_state_ = self.vrnn.cell.get_initial_state(
                    batch_size=batch_size_, dtype=self.dtype
                )
            elif isinstance(initial_state, list):
                initial_state_ = [state[n_samples_:n_samples_ + batch_size_]
                                  for state in initial_state]
                initital_state_ = [tf.constant(state)
                                   for state in initial_state_]
            else:
                initial_state_ = initial_state[n_samples_:
                                               n_samples_ + batch_size_]
                initial_state_ = tf.constant(initial_state_)

            outputs_, states_ = self._generate(
                batch_size_, n_steps, initial_state=initial_state_,
                return_intensity=return_intensity, return_x=return_x
            )

            outputs_ = [output.numpy() for output in outputs_]
            outputs.append(outputs_)
            if return_state:
                states_ = [state.numpy() for state in states_]
                states.append(states_)

            n_samples_ += batch_size_

        outputs = [np.vstack(output) for output in zip(*outputs)]
        if outputs and len(outputs) == 1:
            outputs = outputs[0]

        states = [np.vstack(state) for state in zip(*states)]
        if states and len(states) == 1:
            states = states[0]

        if len(outputs) > 0 and len(states) > 0:
            return outputs, states
        elif len(outputs):
            return outputs
        elif len(states):
            return states
        else:
            return None

