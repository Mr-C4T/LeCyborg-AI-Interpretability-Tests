# LeCyborg-AI-Interpretability

This is a **fork of [physical-AI-interpretability](https://github.com/villekuosmanen/physical-AI-interpretability)**, adapted to provide **interpretability tools for the [LeCyborg project](https://github.com/Mr-C4T/LeCyborg)**.

While the original work focused on **visual attention mapping** + **proprioceptive features**, this fork's goal is to extend the attention visualization tools to support our **EMG sensor data** collected from the LeCyborg wearable myo sensor. (🚧 Work in progress 🚧)

## 🦾 LeCyborg Dataset

You can view our **LeCyborg** dataset online here:  
[LeRobot-worldwide-hackathon/7-LeCyborg-so100_emg_sensor](https://lerobot-visualize-dataset.hf.space/LeRobot-worldwide-hackathon/7-LeCyborg-so100_emg_sensor/episode_0)

<img src="assets/emg_dataset.gif" width="400">

## Visual Attention Mapping

This attention map highlights which regions of the camera input are most important for the model when predicting the robot’s next actions.

<img src="assets/emg_attention.gif" width="400">

## EMG Attention

These graphs were generated using my modified version of the attention map visualizer to extract EMG sensor attention values into a .csv file, which I then plotted using a Python script with Matplotlib.

<img src="assets/sensor+rawAtt.gif" width="400">

<img src="assets/sensor+normAtt.gif" width="400">

## 📖 Original Work & Credits

#### Huge thanks to [villekuosmanen](https://github.com/villekuosmanen) for sharing such great work. 
I encourage anyone interested in interpretability and physical AI to go check his repo and give it a star !

⭐ [physical-AI-interpretability](https://github.com/villekuosmanen/physical-AI-interpretability)

You can find the original README here:

📃 [`old_README.md`](./old_README.md)

