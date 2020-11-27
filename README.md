# GUIGAN: Learning to Generate GUI Designs Using Generative Adversarial Networks

Graphical User Interface (GUI) is ubiquitous in almost all modern desktop software, mobile applications and online websites. A good GUI design is crucial to the success of the
software in the market, but designing a good GUI which requires much innovation and creativity is difficult even to well-trained designers. In addition, the requirement of rapid development of GUI design also aggravates designers’ working load. So, the availability of various automated generated GUIs can help enhance the design personalization and specialization as they can cater to the taste of different designers. 

To assist designers, we develop a model GUIGAN to automatically generate GUI designs. Different from conventional image generation models based on image pixels, our GUIGAN is to reuse GUI components collected from existing mobile app GUIs for composing a new design which is similar to natural-language generation. Our GUIGAN is based on SeqGAN by modelling the GUI component style compatibility and GUI structure. The evaluation demonstrates that our model significantly outperforms the best of the baseline methods by 30.77% in Frechet Inception distance (FID) and 12.35% in 1-Nearest Neighbor Accuracy (1-NNA). Through a pilot user study, we provide initial evidence of the usefulness of our approach for generating acceptable brand new GUI designs. We formulate our task as generating a new GUI by selecting a list of compatible GUI subtrees.

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/Fig1_v2.jpg)

An overview of our approach is shown in the figure. First, We collect lots of GUI screenshots and their corresponding metainformation from the Android apps from Google Play and decompose them into component subtrees for re-using. Second, we develop a SeqGAN based model. Apart from the default generation and discrimination loss, we model the GUI component style compatibility and GUI layout structure for guiding the training. Therefore, our GUIGAN can generate sequences for reconstructing brand new GUI designs for designers’ inspiration. 

## Details
One GUI design image consists of two types of components i.e., widgets and spatial layouts. The widgets are organized by the layout as the structural tree for one GUI design. We take the subtree of existing GUIs as the basic unit for composing a new GUI design rather than plain pixels. 

We cut these candidate subtrees from the original screenshot according to certain rules. Given one GUI design with detailed component information, we cut out all the first-level subtrees from the original DOM tree . If the width of a subtree exceeds 90% of the GUI width, we continue to cut it to the next level, otherwise we stop splitting and this subtree is used as the smallest granularity unit. The procedure will be iterated until all the segmentation stops. Finally, we use all the smallest subtrees as indexes to identify templates. 

We remove the subtrees with duplicate bounds in one GUI and keep only one in the process. Besides, subtrees with partial overlap and too high or too low aspect ratio are also removed, which can not be cut from the original GUI.

<!-- ![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/Fig2_cut.jpg) --> 
<img src="https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/Fig2_cut.jpg" width="500">

The figure shows an example segmentation of a real GUI screen shot, and each subtree is used as the basic unit in our work. 


##  Learning to Generate GUI Designs Using Generative Adversarial Networks

As shown in the figure, based on subtrees automatically segmented from the original GUIs, we first convert all them into embedding by modeling their style. During the training process, the generator randomly generates a sequence with the given length and the discriminator acts as the environment, in which the reward can be calculated as the loss_g by Monte Carlo tree search (MCTS). We get the homogeneity value of the generated result as loss_c. By measuring the distance between the generated result and the original GUI design, the model captures the structural information with loss_s calculated by the minimum edit distance. By integrating all the loss functions above, the parameters of the generator are updated with the backpropagation algorithm.

<!-- ![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/Fig3.jpg) --> 
<img src="https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/Fig3.jpg" width="650">

### Style Embedding of Subtree

We adopt a siamese network to model the GUI design with a dual-channel CNN structure. We apply a pair of GUI images (g1,g2) as the input and the goal of the siamese network is to distinguish whether the two images are from the same app. 

### Modeling Subtree Compatibility

We apply the homogeneity (HOM) to evaluate the aesthetic compatibility of subtrees in the sequence. 

### Modeling Subtree Structure

We use the structure strings of the subtrees from their meta data to represent their structures. Then we apply the minimum edit distance (MED) to evaluate the structural similarity between the generated samples and the real ones.

### Multi-Loss Fusion

By adding the trainable noise parameters, we balance the three loss values (the feedback loss from the discriminator, compatibility loss, loss_c, and structure loss, loss_s) to the same scale, and the parameters of the generator can be updated.

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/Fig5.jpg)

As shown in the figure above, two samples generated by the GUIGAN are actually reconstructed by the pieces from the real GUI.

## Implementation

### Dataset Construction

Our data comes from Rico(from **[`Rico dataset`](http://interactionmining.org/rico)**), an open source mobile app dataset for building data-driven design applications. 

Through manual selection, we have collected relatively professional and more suitable apps for this study. The apps with more images, animation or game screens are not selected. In addition, the GUIs with large pop-up areas, Web links waiting, and full screen ads are not selected. (our experimental data collection: **[`Download`](https://drive.google.com/file/d/1I4U6TNIPqK8VW1gNlSy192WetKoYGA8O/view?usp=sharing)**)

### Model Implementation

The Long Short-Term Memory (LSTM) is used as the generative network. The word vector dimension is selected to be 32 and the hidden layer feature dimension is selected
to be 32. We use a CNN network that joins the highway architecture (same as the discriminative model in SeqGAN) as the discriminator. The batch size is 32 and the learning rate is set to 0.05. 

The siamese network used for learning GUI design style is basically a two-channel CNNs with shared weights, and there are 4 Conv->Pool layers blocks in the CNN structure.

All networks are implemented on the PyTorch platform and trained on a GPU.

## Results

### Automated Evaluation

We try to test our model’s capability in capturing that characteristic by preparing separated dataset for five most frequent app categories in Rico dataset, including News & Magazines, Books & Reference, Shopping, Communication, and Travel & Local. In addition, we prepare three kinds of GUIs from three big companies with most apps in our dataset i.e., Google, Yinzcam, and Raycom as shown in Table 1.

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/T1.jpg)

Frechet Inception distance (FID) and 12.35% in 1-Nearest Neighbor Accuracy (1-NNA) are used to quantify and measure the similarity between the real data distribution and the generated sample distribution. In our experiment, the lower the score of these two metrics, the better the performance

We use WGAN-GP(image generation from pixel level) and FaceOff(template search) as the baselines. Two other derivation baselines(GUIGAN-style with GUI design style information only and GUIGAN-structure with GUI structural information only) are from our own model by changing the mutiloss in the generator for exploring the impact of style information and structure information on the generated results.

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/T2.jpg)

Table 2 and Table 3 show the experimental results of our model and baseline methods on two metrics in the category and company specific development scanarios. The results show that the our model has better performance in the two metrics than the other baselines in most development scanarios. 

**Examples of GUIGAN**

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/Display1.jpg)

The samples generated by GUIGAN can be seen in the figure above, which have a comfortable appearance, and reasonable structure composed of different components. At the same time, it also keeps the overall harmonious design style. Both the structure and style of the GUIs are also very diverse which can provide developers or designers with different candidates for their GUI design. (more samples can be found: **[`Download`](https://drive.google.com/file/d/1c98iQQrX8Jgxj_V7nYQoCI5g-aDQYuMW/view?usp=sharing)**)

**Examples of Component Reused**

<!-- ![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/ComponentReuse.jpg) --> 
<img src="https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/ComponentReuse.jpg" width="650">

Two examples of the same component used in different generated GUIs.

**Examples of baselines**

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/baseline1.png)

Generated GUI examples by WGAN-GP (a) with blurred in detail. And FaceOff (b, c, d) with diversity loss and same color schema.

**Examples of GUIGAN Derivation**

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/baseline2.png)

Generated GUI examples by GUIGAN-style (a, b), with harmonious color combinations as seen in Fig 7 (a) and (b), but without very good structural designs. And GUIGAN-structure (c, d), with reasonable and diverse layouts of generated GUIs, but terrible color schema.

### Human Evaluation

We propose three novel metrics, i.e., design aesthetics, color harmony, and structure rationality for five participants(with Android development experience about GUI implementation and some GUI design) to rate the quality of the GUI design from three aspects by considering the characteristics of the mobile GUIs. For each metric, the participants will give a score ranging from 1 to 5 with 1 representing the least satisfactoriness while 5 as the highest satisfactoriness. We select 5 app categories (News & Magazines, Books & Reference, Shopping, Communication, and Travel & Local)  for specific GUI generation. For each category, we randomly generate 10 GUI designs for each method. The participants do not know which GUI design is from which method and all of them will evaluate the GUI design individually without any discussion.

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/T4.jpg)

As shown in Table IV, the generated GUI designs from our model outperforms that of FaceOff significantly in overall aesthetics, color harmony and structure. We also carry out the Mann-Whitney U test on three metrics and the results suggests that our GUIGAN can contribute significantly to the GUI design. 

FaceOff: 
![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Results/UserStudy/Fig_HistFaceoff.jpg)

GUIGAN: 
![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Results/UserStudy/Fig_HistGUIGAN.jpg)

Real GUI: 
![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Results/UserStudy/Fig_HistReal.jpg)

As shown in the histogram, the scores of the three metrics (color harmony, structure rationality, and aesthetics, from left to right) are respectively counted, with the horizontal axis representing the scores and the vertical axis representing the times.

### Examples of pre-built Components

Our model can generate other components to complete a whole GUI design based on the unfinished pre-built components (the leftmost column), and the final generated samples are diverse. 

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/pre1.jpg)

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/pre2.jpg)

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/pre3.jpg)

![Alt text](https://github.com/GUIDesignResearch/GUIGAN/blob/master/Display/pre4.jpg)

Given some pre-built GUI components (which can be navigation, pictures, or most of the GUI has been designed with few parts to be finished), our model can generate other parts of the GUI to complete the whole GUI design. The structures and styles of the generated samples are different and diverse. 

