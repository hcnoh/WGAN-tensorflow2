# WGAN Implementation for Generating MNIST Images in TensorFlow2

This repository is for the TensorFlow2 implementation for WGAN. This repository provides the training module and Jupyter notebook for testing a generation of the trained models. MNIST dataset was used for this repository.

![](/assets/img/README/README_2019-12-07-18-50-47.png)

*The result is not so good...*

## Install Dependencies
1. Install Python 3.5.2.
2. Install TensorFlow ver 2.0.0. If you can use a GPU machine, install the GPU version of TensorFlow, or just install the CPU version of it.
3. Install Python packages(Requirements). You can install them simply by using following bash command.

    ```bash
    $ pip install -r requirements
    ```

    You can use `Virtualenv`, so that the packages for requirements can be managed easily. If your machine have `Virtualenv` package, the following bash command would be useful.

    ```bash
    $ virtualenv wgan-tf2-venv
    $ source ./wgan-tf2-venv/bin/activate
    $ pip install -r requirements.txt
    ```

## Training
*Note: TensorFlow provides dataset modules for some well known datasets such as MNIST, CIFAR-10 etc. In this repository, the only usage for TensorFlow MNIST dataset module was implemented yet. Usages for other datasets will be implemented too.*

1. **Modify the path for dataset in `config.py`.**

2. **Modify the path for directory for saving model checkpoint.**

3. **Execute training process by `train.py`.**

## Checking Results and Testing Generation
The Jupyter notebook for checking results and testing the image generation is provided. Please check `result_plot.ipynb`.

## Results

1. **Ploting the Generator and Discriminator Losses**

    ![](/assets/img/README/README_2019-12-07-18-54-05.png)

2. **Image Generation Results**

    ![](/assets/img/result_plot/image_generation_result_changes.gif)

## References
- GAN: [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets)
- WGAN : [Wasserstein GAN
](https://arxiv.org/abs/1701.07875)

## Author
Hyungcheol Non / [About Me](https://hcnoh.github.io/about)