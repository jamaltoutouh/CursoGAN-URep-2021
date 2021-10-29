"""
Author: Jamal Toutouh (www.jamal.es)

It contains the code to show an example about using Generative Adversarial Networks.
The GAN is used to generate vectors of a given size that contains float numbers that follow a given Normal distribution
defined my the mean and standard deviation.

Spanish: Ejemplo en el que se entrena una GAN para que cree vectores con valores aleatorios que cumplan con una
distribución normal dada la media y la desviación estandar


Original file is located at
    https://colab.research.google.com/drive/1gbTlefMoY6eQDlZXpCU9PVINo55Yb3ly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

"""Definimos los datos reales: distribución normal de media *real_data_mean* y desviación estándar *real_data_stddev*"""

# Real data set: Samples from a normal distribution given real_data_mean and real_data_sttdev
real_data_mean = 4.0
real_data_stddev = 0.2


def plot_data(real_mean, real_sigma, fake_data=None, epoch=None):
    x = np.linspace(1, 7, 100)
    plt.figure()

    real_data_distribution = norm.pdf(x, real_mean, real_sigma)
    plt.plot(x, real_data_distribution, 'b', linewidth=2, label='Real data')
    
    if not (fake_data is None):
      mu, std = norm.fit(fake_data.tolist())
      fake_data_distribution = norm.pdf(x, mu, std)
      plt.plot(x, fake_data_distribution, 'k', linewidth=2, label='Fake data')
      title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    else:
      title = "Real data: mu = %.2f,  std = %.2f" % (real_mean, real_sigma)
      epoch = "real"
    plt.xlim(2,6)
    plt.ylim(0, 2.5)
    plt.title(title)
    plt.legend()
    plt.savefig('01-1d-normal-gan-data-{}.png'.format(epoch))
    plt.show()

def get_real_sampler(mu, sigma):
    """
    Creates a lambda function to create samples of the real data (Normal) distribution
    :param mu: Mean of the real data distribution
    :param sigma: Standard deviation of the real data distribution
    :return: Lambda function sampler
    """
    dist = Normal(mu,sigma )
    return lambda m, n: dist.sample((m, n)).requires_grad_()


# Load samples from real data
get_real_data = get_real_sampler(real_data_mean, real_data_stddev)
plot_data(real_data_mean, real_data_stddev)


"""Función que crea los vectores del espacio latente que leerá el generador para crear los datos"""

def read_latent_space(batch_size, latent_vector_size):
    """
    Creates a tensor with random values fro latent space  with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with random values (z) with shape = size
    """
    z = torch.rand(batch_size, latent_vector_size)
    if torch.cuda.is_available(): return z.cuda()
    return z


"""Clase que define el generador"""

class Generator(nn.Module):
    """
    Class that defines the the Generator Neural Network
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, output_size),
            nn.SELU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


"""Clase que define el discriminador"""

class Discriminator(nn.Module):
    """
    Class that defines the the Discriminator Neural Network
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x


"""Función auxiliar para mostrar la evolución de la función de pérdida del generador y el discriminador"""

def plot_loss_evolution(discriminator_loss, generator_loss):
    x = range(len(discriminator_loss)) if len(discriminator_loss) > 0 else range(len(generator_loss))
    if len(discriminator_loss) > 0: plt.plot(x, discriminator_loss, '-b', label='Discriminator loss')
    if len(generator_loss) > 0: plt.plot(x, generator_loss, ':r', label='Generator loss')
    plt.legend()
    plt.savefig('01-1d-normal-gan-loss.png')
    plt.show()


"""Funciones para crear las etiquetas de dato real *real_data_target* y dato falso *fake_data_target* que emplea el discriminador para calcular la función de pérdida"""

def real_data_target(size):
    """
    Creates a tensor with the target for real data with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with real label value (ones) with shape = size
    """
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    """
    Creates a tensor with the target for fake data with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with fake label value (zeros) with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


"""Función principal que define el entrenamiento de la GAN"""

def main():
    vector_length = 10  # Defines the length of the vector that defines a data sample

    generator_input_size = 50  # Input size of the generator (latent space)
    generator_hidden_size = 150
    generator_output_size = vector_length

    discriminator_input_size = vector_length
    discriminator_hidden_size = 75
    discriminator_output_size = 1

    batch_size = 30
    number_of_batches = 600

    # Creating the GAN generator
    generator = Generator(input_size=generator_input_size, hidden_size=generator_hidden_size,
                          output_size=generator_output_size)
    generator_learning_rate = 0.0008
    generator_loss = nn.BCELoss()
    generator_optimizer = optim.SGD(generator.parameters(), lr=generator_learning_rate, momentum=0.9)

    # Creating the GAN discriminator
    discriminator = Discriminator(input_size=discriminator_input_size, hidden_size=discriminator_hidden_size,
                                  output_size=discriminator_output_size)

    discriminator_learning_rate = 0.001
    discriminator_loss = nn.BCELoss()
    discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=discriminator_learning_rate, momentum=0.9)

    epochs = 20 # Training epochs
    freeze_generator_steps = 1

    noise_for_plot = read_latent_space(batch_size, generator_input_size)
    discriminator_loss_storage, generator_loss_storage = [], []

    print('Real dataset loaded...')
    print('Starting adversarial GAN training for {} epochs.'.format(epochs))

    # Plot a little bit of trash
    generator_output = generator(noise_for_plot)
    plot_data(real_data_mean, real_data_stddev, generator_output, -1)

    for epoch in range(epochs):

        batch_number = 0

        # training discriminator
        while batch_number < number_of_batches:  

            # 1. Train the discriminator
            discriminator.zero_grad()
            # 1.1 Train discriminator on real data
            input_real = get_real_data(batch_size, discriminator_input_size)
            discriminator_real_out = discriminator(input_real)
            discriminator_real_loss = discriminator_loss(discriminator_real_out, real_data_target(batch_size))
            discriminator_real_loss.backward()
            # 1.2 Train the discriminator on data produced by the generator
            input_fake = read_latent_space(batch_size, generator_input_size)
            generator_fake_out = generator(input_fake).detach() # Synthetic data
            discriminator_fake_out = discriminator(generator_fake_out)
            discriminator_fake_loss = discriminator_loss(discriminator_fake_out, fake_data_target(batch_size))
            discriminator_fake_loss.backward()
            # 1.3 Optimizing the discriminator weights
            discriminator_optimizer.step()

            # discriminator_total_loss = discriminator_real_loss + discriminator_fake_loss
            # discriminator_total_loss.backward()
            # discriminator_optimizer.step()


            # 2. Train the generator
            if batch_number % freeze_generator_steps == 0:
              generator.zero_grad()
              # 2.1 Create fake data
              input_fake = read_latent_space(batch_size, generator_input_size)
              generator_fake_out = generator(input_fake)
              # 2.2 Try to fool the discriminator with fake data
              discriminator_out_to_train_generator = discriminator(generator_fake_out)
              discriminator_loss_to_train_generator = generator_loss(discriminator_out_to_train_generator,
                                                                    real_data_target(batch_size))
              discriminator_loss_to_train_generator.backward()
              # 2.3 Optimizing the generator weights
              generator_optimizer.step()

            batch_number += 1

        discriminator_loss_storage.append(discriminator_fake_loss + discriminator_real_loss)
        generator_loss_storage.append(discriminator_loss_to_train_generator)
        print('Epoch={}, Discriminator loss={}, Generator loss={}'.format(epoch, discriminator_loss_storage[-1],
                                                                          generator_loss_storage[-1]))

        if epoch % 1 == 0:
            generator_output = generator(noise_for_plot)
            plot_data(real_data_mean, real_data_stddev, generator_output, epoch)


    plot_loss_evolution(discriminator_loss_storage, generator_loss_storage)

main()