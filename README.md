# Learning-Residual-Images-for-Face-Attribute-Manipulation

This code is implemented by TensorFlow, and the method of the code is from the paper [Learning Residual Images for Face Attribute Manipulation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shen_Learning_Residual_Images_CVPR_2017_paper.pdf) which is published in CVPR2017.

This paper is very interesting, it use GAN, dual learning and residual image to yield a high quilty image. Especially the method of residual image, i think it's a very nice idea.

Method of the paper:
----------------------

![paper](https://github.com/MingtaoGuo/Learning-Residual-Images-for-Face-Attribute-Manipulation/raw/master/result/method.jpg)

Result of our code:
--------------------

![result](https://github.com/MingtaoGuo/Learning-Residual-Images-for-Face-Attribute-Manipulation/raw/master/result/ExpResult.jpg)

First row images are the original images, the second row images are the residual images, the third images are the result of the original image plus the residual image pixel element-wise. This result we just train about 50 epochs, quilty of the image is not well.

The dataset of our code is downloaded from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)which has 40 face attributes to use, we just select one attribute of glasses on the face or not,so the experiment here is only about adding and removing glasses on the face. If you want to manipulate other face attribute, you can select other face attribute and put them in the folder whose name include '+1' and '-1'. For example, attribute of male, you put all male images in '+1', and put all female images in '-1'.




