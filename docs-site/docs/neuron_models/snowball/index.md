# Snowball

## Circuit description
The circuit implemented for exponential integrate and fire neuron has been used from [1]. Part (a) in Fig.2 in [1] implements the exponential integrate and fire neuron. The neuron receives input currents using the input DPI filter [2]. This input current is integrated on the node Vmem by the membrane capacitance. The membrane potential leaks in the absence of an input spike which can be set by the bias Vleak. The Vmem potential node is connected to a cascoded source follower formed by the P14-15 and N5-6. A threshold voltage of the neuron can be set by the bias Vthr which is compared to the membrane potential. When the membrane potential is just near the threshold voltage, it starts the positive feedback block which exponentially increases membrane potential and causes the neuron to spike. As the neuron spikes, the membrane potential gets reset to ground and the refractory bias helps to stop the neuron from spiking during the refractory period as similar to a biological neuron. The circuit implemented for this experiment does not exercise either adaptability or needs a pulse extender as implemented in [1]. The Vdd used in the simulation is 1V. The neuron receives 5nA input pulses with a pulse width of 100μs.

Input current mirror W/l = 0.2 <br>
All other transistors W/L = 4/3
## Circuit Simulation

![snowball, output plot](/docs-site/docs/img/exif_plot.png) Fig.1 The dynamics of Exponential integrate and fire neuron. The light blue signal is the input spikes, the yellow signal is the membrane potential and the dark blue is the output spikes from the neuron.
## References
1. Rubino, Arianna, Melika Payvand, and Giacomo Indiveri. "Ultra-low power silicon neuron circuit for extreme-edge neuromorphic intelligence." 2019 26th IEEE International Conference on Electronics, Circuits and Systems (ICECS). IEEE, 2019.
2. Bartolozzi, Chiara, Srinjoy Mitra, and Giacomo Indiveri. "An ultra low power current-mode filter for neuromorphic systems and biomedical signal processing." 2006 IEEE Biomedical Circuits and Systems Conference. IEEE, 2006.
