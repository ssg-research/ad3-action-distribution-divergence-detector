# AD<sup>3</sup>-action-distribution-divergence-detector

This repo contains the code to reproduce experiments for the detection scheme presented in *Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses* ([arXiv report](https://arxiv.org/abs/2106.08746)). The paper will appear in the Proceedings of ESORICS 2022.

DISCLAIMER: The provided source code does NOT include experiments with the generation of universal adversarial perturbations. Usage and distribution of such code is potentially harmful and should be done separately at their authors' disclosure. In this repo we provide:

* All victim agents and the code to re-train victim agents from scratch
* The AD<sup>3</sup> algorithm 
* [visual foresight](https://arxiv.org/abs/1702.02284) modules and the code to train these modules from scratch
* One example of an adversarial mask (UAP-S and UAP-O) to evaluate defense methods.
* Please refer to the [original repository](https://github.com/chenhongge/SA_DQN) for the code and State-Adversarial DQN models used in our paper. 


<img src="images/overview.png" width="600">
