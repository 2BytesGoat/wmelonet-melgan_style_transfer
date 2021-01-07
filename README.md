# Pretrained models
[Not great, not terrible](https://drive.google.com/file/d/1IY1F5uz0cKi5n3LcbribF_AZmgBpcCf8/view?usp=sharing)

# References

**Starting code:**

[MelGAN - GitHub](https://github.com/descriptinc/melgan-neurips) <-starting code \
[TraVelGAN - GitHub](https://github.com/clementabary/travelgan) <- starting code

**Dataset:**

[Gtzan Dataset - Kaggle](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

**Refferences:**

[MelGAN - paper](https://arxiv.org/pdf/1910.06711.pdf) <- paper on which the MelGAN is based \
[TraVelGAN - paper](https://arxiv.org/abs/1902.09631) <- paper on which TraVelGAN is based \
[Image Padding Techniques - Machinecurve](https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/) <- they used reflection padding in the paper

**My sources:**

[Audio Style Transfer with GANs - Towards Data Science](https://towardsdatascience.com/voice-translation-and-audio-style-transfer-with-gans-b63d58f61854) <- where I started from\
[Valerio Veraldo - YouTube](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&ab_channel=ValerioVelardo-TheSoundofAI) <- audio theory \
[Inpainting with AI - Towards Data Science](https://towardsdatascience.com/inpainting-with-ai-get-back-your-images-pytorch-a68f689128e5) <- this is where I got the inspiration for style transfer \
[Contrastive Loss - Towards Data Science](https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec) <- I may use this for style loss

**Other style-transfer ideas:**

[Genetic Algorithm Generates Music - Youtube](https://www.youtube.com/watch?v=aOsET8KapQQ&t=218s) <- This is how you can create musing using midi

# Questions

Q1: Why use soundwave as target? \
A1: It's because the quality is way higher than the spectogram (we lose information when making small histograms using Fast Fourier Transform).

Q2: Why use 1D filters and not 2D? \
A2: Because collumns in a spectogram have more meaning than square-crops.

Q3: What results did you have when using texture transfer on sound? \
A3: Content and style were merged, as if I was listening to both songs in the same time.
  
# ToDos

[x] <s>Experiment with spectograms</s> \
[x] <s>Understand how image style transfer works</s> \
[x] <s>Try image style transfer on sounds using discriminators</s> \
[x] <s>Use siamese loss for style points</s> \
[o] Create spectogram encoder-decoder \
[o] Build GAN for style transfer \
[o] Clean-up and explain better notebooks