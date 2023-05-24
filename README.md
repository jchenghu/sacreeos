
### SacreEOS

[SacreEOS](https://arxiv.org/abs/2305.12254) is a signature generator and 
implementation helper for the [Self Critical Sequence Training](https://arxiv.org/abs/1612.00563). <br>

### Motivation

SacreEOS main goal consists of spreading awareness about easy-to-overlook aspects
of the Self-Critical Squence Training (SCST), in particular over the `End-of-Sequence (EOS)`
token. Currently most Image Captioning projects follow two different approaches, we call for simplicity, 
Standard and No&lt;Eos&gt;. The two versions differ in few subtle implementation details that significantly 
impact the outputs and performances, ultimately posing an obstacle to the evaluation and comparison of models.


The generation of shareable signatures serves the purpose of spreading awareness about the issue,
incrase transparency over previous works and guiding future works into an informed implementation.
To this end, the package provides also useful classes in the hope of saving time and preventing headaches 
to those implementing SCST for the first time.


### Features

##### Main feature

<ul>
    <li> <b>SacreEOS signature generation. </b>
    The package provides an easy interface for the generation of signatures
    that inform regarding the key aspects of the SCST implementation. This can be done
    either manually or automatically using the implementation helper.<br><br>
    </li>
</ul>

##### Optional features

The adoption of SacreEOS as an implementation helper is completely optional.
Established projects are only invited to manually generate the SacreEOS signature.


<ul>
    <li> <b>Mode and metric selection. </b>
    The user can customize the target SCST configuration according to its needs. The package currently 
    supports 2 SCST modes, 4 metrics and 2 bases. <br><br>
    </li>
    <li> <b>Implementation helper. </b>
    Through a collection of exceptions and input conditions the user is guided toward an informed selection of SCST implementation. 
    <br><br></li>
    <li> <b>Efficiency. </b>
    For each metric, originally proposed in Python, an additional and optional C implementation 
    is provided. It can be toggled by setting one of the loss computation arguments. In case of 
    platform incompatibilities, the portability is still preserved thanks to the Python implementation. <br><br>
    </li>
</ul>


- - -

### Installation

Requirements:
<ul>
    <li> torch, numpy, typing.
    </li>
    <li> Python >= 3.7
    </li>
</ul>


You can install the package using pip:
```
python -m pip install sacreeos
```
or clone the repository:
```
git clone https://github.com/jchenghu/sacreeos
```

- - -

### Usage


#### Signature generation

##### Manual 

The SacreEOS signature can be generated in two ways.
In case you installed the package using `pip` simply write: 
```
$ python3.7 -m sacreeos
```
otherwise in case you cloned the repository:
```
$ cd sacreeos
$ python sacreeos
```

It will ask few information regarding the SCST implementation. A default
answer is provided in most cases based on the most popular configurations.

Examples of the two most common signatures: <br>
`STANDARD_wInit+Cider-D[n4,s6.0]+average[nspi5]+1.0.0` <br>
`NO<EOS>_wInit+Cider-D[n4,s6.0]+average[nspi5]+1.0.0` <br>



##### Automatic

The signature is automatically generated in case the package is adopted as 
implementation helper. Once the `Scst` class is constructed, the 
signature can be generated using the method `get_signature()`.  We refer to the sections below.

#### Implementation helper

SacreEOS provide the `Scst` class which computes the SCST loss over the inputs using the
selected metrics and performs a series of assersions and input checks based on the selected
class.

`Scst` Constructor: 
```
class: Scst(scst_class, metric_class, base_class, eos_token, corpus_refss=None,
            verbose=True, metric_args=None, base_args=None):
--------------------------------------------------------------------------------------
Arguments:

    <> scst_class: scst mode class
              - Choices: Scst.SCST_CONFIG_STANDARD, Scst.SCST_CONFIG_NO_EOS                

    <> metric_class: metric class to be selected 
              - Choices: Scst.METRIC_BLEU: requires, Scst.METRIC_CIDER, Scst.METRIC_CIDER_D, Scst.METRIC_CIDER_R.
    
    <> base_class: base reward type 
              - Choices: Scst.BASE_AVERAGE, Scst.BASE_GREEDY.
    
    <> eos_token: str : end of sequence token
    
    <> corpus_refss: list of list of sentences. 
                     The document frequencies calculation won't be affected by how sentences 
                     are placed inside the list of list, in fact, the format is required only
                     for sake of consistency with the one required by the loss computation

    <> verbose: bool
    
    Further Customizations: 
        leave these arguments to None to keep the standard configurations.

    <> metric_args: dictionary with metric arg name as key and custom arg values as value
            - Scst.METRIC_BLEU: requires no args
            - Scst.METRIC_CIDER:  {'n': int}                # max number of ngram   
            - Scst.METRIC_CIDER_D : {'n': int,  
                                     'sigma': float}         
            - Scst.METRIC_CIDER_R: {'n': int,               
                                    'repeat_coeff': float,  # length and repeatition penalty weights, the two must sum to 1
                                    'length_coeff': float,  
                                    'alpha': float}         # 

    <> base_args: dictionary with base arg name as key and custom arg values as value
            - {'nspi': int}  # number of samples per input/image.



```

Scst loss computation method:
```
method compute_scst_loss(sampled_preds: List[List[str]], sampled_logprobs: torch.tensor,
                         refss: List[List[str]], base_preds=None, eos_pos_upthresh=None,
                         reduction='mean', get_stat_data=False):
--------------------------------------------------------------------------------------
ARGUMENTS:

    <> sampled_preds: List[List[str]] of shape (batch_size, nspi): is the list of predicted sequences.    

    <> sampled_logprobs: tensor of shape (batch size, nspi, seq_len): refers to the log-probabilities of the predicted sequences.           
            -  The function does not perform padding on `sampled_logprobs`. The value of padding elements must be zero.
             Because of the popular practice of sub-word tokenization (such as BPE), this method cannot perform padding
             safely by itself by aligning the elements in `sampled_logprobs` and `sampled_preds`.

    <> base_preds: None or List[List[str]] of shape (batch_size, nspi): if not None, it contains the base sequences.

    <> refss: List[List[str]] of shape (batch_size, int): is the list of references list.
            - refss[i] are expected to be the references of sampled_preds[i]
            - * can be any number 
    
    <> eos_pos_upthresh (end of sequence position upper threshold) defines the length up until which the
     method will ensure the `eos_token` termination. Since SCST is a learning process, in some cases,
     especially in the early iterations, sampled sequences may not end with the `eos_token` because they
     reached the maximum sequence length defined by the model. In case of `None`, is set to the last dimension of `sampled_logprobs` argument.

    <> It should preferably be set to the model's max_len if no sub-word techniques are applied, but
       None (hence the `sampled_logprobs` last dimension) should ensure a wide enough error catching web.

    <> reduction: str: reduction method (None, 'sum', 'mean').
    <> get_stat_data: bool: if true, return not only the loss, but also reward and base reward
    
RETURN:
      (r-b) * (-sampled_logprobs.sum(dim=-1))
    Where:
        <> r represents the reward of `sampled_preds`
        <> b is the reward calculated on base sentences.
```

For automatic signature generation, call `get_signature()` on the class instance.

An example of usage can be found in the demo.


- - - 

#### Demo

The demo sub-package provide an usage example of SacreEOS helper functionality. <br>
It implements a small image captioning system to be trained with SCST.

##### Requirements

The demo requires a pre-trained model and data sampled from COCO. They
can be found in this [drive](https://drive.google.com/drive/folders/1dCFLY0zBRKlV3QQlv6AiadzKaatnQHG8?usp=share_link)
(~170MB), all contents are expected to be placed in `/demo/data/`. <br>

The directories and files in `/demo/` should look like this:
```
demo
├── data
│   ├── demo_data.pickle            -> model pre-trained on XE and training samples
│   ├── coco_train_refss.pickle     -> COCO corpus references
│   ├── dataset.py                  
│   ├── sample_coco_refss.py        -> additional samples for testing
│   └── sample_coco_test.py
├── demo_launcher.py                -> training script
├── __init__.py
├── net
    ├── layers.py                   -> model sub-layers
    ├── model.py                    -> model definition and sampling procedures
    └── utils.py                    -> naive utility functions

```

The training script does not require arguments and can be launched as follows:
```
python demo_launcher.py
```

This is the expected result, although it may be different because of stochastic factors and
small dataset sample:

```
Standard Scst training started...
Standard Scst signature: STANDARD_wInit+Cider-D[n4,s6.0]+average[nspi5]+1.0.0
Training epoch 49: 100%|████████████████████████| 50/50 [45:20<00:00, 54.42s/it]
No artifacts.
Note: because of the poor train set, the standard case may present few artifacts as well
NoEos Scst training started...
NoEos scst signature: NO<EOS>_wInit+Cider-D[n4,s6.0]+average[nspi5]+1.0.0
Training epoch 49: 100%|████████████████████████| 50/50 [42:36<00:00, 51.12s/it]
There are 531 artifacts.
End.
```


- - -

#### Credits

This project is based on the work of [Rennie et al., 2017](https://arxiv.org/abs/1612.00563)
and inspired by [SacreBLEU (Matt Post, 2018)](https://arxiv.org/abs/1804.08771).

Reference:
```
@article{hu2023request,
  title={A request for clarity over the End of Sequence token in the Self-Critical Sequence Training},
  author={Hu, Jia Cheng and Cavicchioli, Roberto and Capotondi, Alessandro},
  journal={arXiv preprint arXiv:2305.12254},
  year={2023}
}
```


