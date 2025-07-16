<p align="center">
<h1 align="center"><strong>Real-time and Controllable Reactive Motion Synthesis</strong></h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=MHQRNggAAAAJ" target="_blank">Xiaotang Zhang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=gHhQNlYAAAAJ&hl" target="_blank">Ziyi Chang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=t1hraiAAAAAJ&hl" target="_blank">Qianhui Men</a><sup>2</sup>,
    <a href="http://hubertshum.com/" target="_blank">Hubert Shum</a><sup>1&dagger;</sup>
    <br>
      <sup>1</sup>Durham University  
      <sup>2</sup>University of Bristol
    <br>
      &dagger; Corresponding Author
  </p>
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/Computer%20Graphics%20Forum-green)](http://doi.org/10.1111/cgf.70222)
[![](https://img.shields.io/badge/Paper-%F0%9F%93%83-blue)](http://doi.org/10.1111/cgf.70222)
[![](https://img.shields.io/badge/Video-%F0%9F%8E%AC-red)](https://youtu.be/jt3Vu2rmD38?si=j5eosbbRhq1FW-XR)

</div>

![Teaser](/materials/Teaser.png)
### Abstract
We propose a real-time method for reactive motion synthesis based on the known trajectory of input character, predicting instant reactions using only historical, user-controlled motions. Our method handles the uncertainty of future movements by introducing an intention predictor, which forecasts key joint intentions to make pose prediction more deterministic from the historical interaction. The intention is later encoded into the latent space of its reactive motion, matched with a codebook which represents mappings between input and output. It samples a categorical distribution for pose generation and strengthens model robustness through adversarial training. Unlike previous offline approaches, the system can recursively generate intentions and reactive motions using feedback from earlier steps, enabling real-time, long-term realistic interactive synthesis. Both quantitative and qualitative experiments show our approach outperforms other matching-based motion synthesis approaches, delivering superior stability and generalizability. In our method, user can also actively influence the outcome by controlling the moving directions, creating a personalized interaction path that deviates from predefined trajectories.

### Usage
1. Install Unity3D Hub and import `Unity3D` project.
2. Open scene `Unity3D/Assets/Demo/Motion Synthesis/ReactionSynthesis`.
3. Load pre-trained models from `PyTorch/Checkpoints` to the `__main__` function of `PyTorch/Models/CodebookMatching/Inference.py`.
4. Run `Inference.py` in python and wait a few seconds for loading checkpoints.
5. Click the play button in Unity3D and wait a few seconds for socket connection.

### Acknowledgement
The Unity3D visualization tool is based on [AI4Animation](https://github.com/sebastianstarke/AI4Animation) and the network's backbone is from [Codebook Matching](https://dl.acm.org/doi/10.1145/3658209). We gratefully acknowledge [Sebastian Starke](https://github.com/sebastianstarke) for his outstanding open-source contributions.

### BibTex
```
@inproceedings{zhang2025real,
  title={Real-time and Controllable Reactive Motion Synthesis via Intention Guidance},
  author={Zhang, Xiaotang and Chang, Ziyi and Men, Qianhui and Shum, Hubert and others},
  booktitle={Computer Graphics Forum},
  year={2025},
  organization={Wiley}
}
```
