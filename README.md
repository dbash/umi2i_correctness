# Evaluation of Correctness in Unsupervised Many-to-Many Image Translation

We propose a protocol for evaluation of disentanglement quality of unsupervised many-to-many image translation (UMI2I) methods. We show that modern UMI2I methods fail to correctly disentangle the domain-specific from shared factors and mostly rely on their corresponding inductive biases to determine which factors should be changed after translation.

**[Dina Bashkirova](https://cs-people.bu.edu/dbash/), [Ben Usman](https://cs-people.bu.edu/usmn/), [Kate Seanko](http://ai.bu.edu/ksaenko.html/)** </br>
Winter Conference on Applications of Computer Vision (WACV) 2022 </br>
[arxiv](https://arxiv.org/pdf/2103.15727.pdf) / [bib](https://cs-people.bu.edu/dbash/bib/i2i_eval.bib) / [data](#downloading-evaluation-data)

> Given an input image from a source domain and a guidance image from a target domain, unsupervised many-to-many image-to-image (UMMI2I) translation methods seek to generate a plausible example from the target domain that preserves domain-invariant information of the input source image and inherits the domain-specific information from the guidance image. For example, when translating female faces to male faces, the generated male face should have the same expression, pose and hair color as the input female image, and the same facial hairstyle and other male-specific attributes as the guidance male image. Current state-of-the art UMMI2I methods generate visually pleasing images, but, since for most pairs of real datasets we do not know which attributes are domain-specific and which are domain-invariant, the semantic correctness of existing approaches has not been quantitatively evaluated yet. In this paper, we propose a set of benchmarks and metrics for the evaluation of semantic correctness of these methods. We provide an extensive study of existing state-of-the-art UMMI2I translation methods, showing that all methods, to different degrees, fail to infer which attributes are domain-specific and which are domain-invariant from data, and mostly rely on inductive biases hard-coded into their architectures.

<!-- ![img](https://cs-people.bu.edu/dbash/img/i2i_eval.png) -->

<p align="center">
  <img src="https://cs-people.bu.edu/dbash/img/i2i_eval.png" />
</p>

### Downloading evaluation data

Proposed dataset splits are currently hosted on google drive [[link]](https://drive.google.com/drive/folders/1ELLH74aD9AMyHcU6jbGhRfC5s1lGJ7pz?usp=sharing). We will provide a `wget`-able links shortly. For CelebA and Shapes3D we provide image ids that can be used to generate splits using original datasets: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Shapes3D](https://github.com/deepmind/3d-shapes). For the SynAction split we provide original images.

### Instructions to reproduce results

TDB
