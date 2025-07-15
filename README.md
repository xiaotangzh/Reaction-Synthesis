> *Computer Graphics Forum (CGF), 2025*
# Real-time and Controllable Reactive Motion Synthesis
[Xiaotang Zhang](https://scholar.google.com/citations?hl=en&user=MHQRNggAAAAJ), [Ziyi Chang](https://scholar.google.com/citations?user=gHhQNlYAAAAJ&hl), [Qianhui Men](https://scholar.google.com/citations?user=t1hraiAAAAAJ&hl), [Hubert Shum](http://hubertshum.com/)

📃[Paper](https://arxiv.org/abs/2507.09704) 🎬[Video](https://www.youtube.com/watch?v=jt3Vu2rmD38)

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

### Citation
```
todo
```
