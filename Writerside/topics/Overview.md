# Genderative AI
An exploration of gender representation in generative AI images

## Introduction
Generative AI models have taken the internet by storm. They are increasingly used in many contexts, from stock photography to illustration. These models are used both to create new images and to enhance existing images. A user of Adobe Photoshop can remove an undesirable portion of a real photograph and replace it with generative content. Often, the blend between the real and generative portions of a photograph are hard to discern.

What assumptions are these models making? What biases are they learning? What are the implications of these biases? These are big questions that warrant years of research. This project analyzes the representation of gender in Stable Diffusion, a wildly popular open source generative AI model.


## Methodology

For each occupation, we generated images using three prompts: "A photograph of a(n) (occupation)", "A photograph of a male (occupation)", and "A photograph of a female (occupation)". Each of the three prompts was run using the same seed to ensure that gender represented the largest difference within each set of three images. We used the following negative prompt to reduce the number of non-photographic images being generated: "drawing, illustration, painting, cartoon, render, frame".

The gender-labeled images were used to train a model to classify the gender of each image. The unlabeled images form the basis of this paper. The trained gender classifier was applied to the unlabeled images in order to calculate the percentage of images displaying charactoristics that are primarily male or female. We used an active learning process to improve model accuracy. We trained the model and had it predict the gender of its training set to check for mislabeled images. We then predicted the gender of a special batch of unlabeled images and manually assigned labels to each image where the model was uncertain. We then added those models back to the training data, generated new labeled images to ensure an even gender balance, trained a new model, and went through the process again until we were satisfied with the results.