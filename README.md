# How to use


# References

**Starting code:**

[MelGAN - GitHub](https://github.com/descriptinc/melgan-neurips)
[MelGAN - paper](https://arxiv.org/pdf/1910.06711.pdf)

**Audio theory:**

[Valerio Veraldo - YouTube](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&ab_channel=ValerioVelardo-TheSoundofAI)

**Other sources:**

[Audio Style Transfer with GANs - Towards Data Science](https://towardsdatascience.com/voice-translation-and-audio-style-transfer-with-gans-b63d58f61854) \
[Image Padding Techniques - Machinecurve](https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/) \
[Inpainting with AI - Towards Data Science](https://towardsdatascience.com/inpainting-with-ai-get-back-your-images-pytorch-a68f689128e5) \
[Genetic Algorithm Generates Music - Youtube](https://www.youtube.com/watch?v=aOsET8KapQQ&t=218s)

# ToDos

[x] <s>Experiment with spectograms</s> \
[_]  Create spectogram encoder-decoder

# Questions

Q1: Why use soundwave as target? \
A1: I think it's because the quality is way higher than the spectogram (we lose information when applying Fast Fourier Transform).

Q2: Why use 1D filters and not 2D? \
A2: 

# BASED on the Official repository for the paper MelGAN

Previous works have found that generating coherent raw audio waveforms with GANs is challenging. In this [paper](https://arxiv.org/abs/1910.06711), we show that it is possible to train GANs reliably to generate high quality coherent waveforms by introducing a set of architectural changes and simple training techniques. Subjective evaluation metric (Mean Opinion Score, or MOS) shows the effectiveness of the proposed approach for high quality mel-spectrogram inversion. To establish the generality of the proposed techniques, we show qualitative results of our model in speech synthesis, music domain translation and unconditional music synthesis. We evaluate the various components of the model through ablation studies and suggest a set of guidelines to design general purpose discriminators and generators for conditional sequence synthesis tasks. Our model is non-autoregressive, fully convolutional, with significantly fewer parameters than competing models and generalizes to unseen speakers for mel-spectrogram inversion. Our pytorch implementation runs at more than 100x faster than realtime on GTX 1080Ti GPU and more than 2x faster than real-time on CPU, without any hardware specific optimization tricks. Blog post with samples and accompanying code coming soon.

Visit our [website](https://melgan-neurips.github.io) for samples. You can try the speech correction application [here](https://www.descript.com/overdub) created based on the end-to-end speech synthesis pipeline using MelGAN.

Check the [slides](melgan_slides.pdf) if you aren't attending the NeurIPS 2019 conference to check out our poster.