Hello!

This tool is currently under active development, but seeks to simulate a multibeam echosounder for 
visualization of the influence of operating parameters on the derived swath or particular sounding.

A full feature list will be detailed here, once all capabilities have been robustly implemented.

Notable limitations include the assumption of a uniform volume of water, a perfectly flat seafloor, 
perfect pitch, roll, and yaw stabilization, and no information or distinction of the physical locations 
of the tx and rx arrays.

Much of the UI was created with assistance from Gemini, and the mathematical basis was primarily sourced from lecture and lab materials from the 
Integrated Seabed Mapping Systems course by John Hughes Clarke and Semme Dijkstra at the University of New Hampshire. Additional classical
references were utilized such as:

Sonar for Practising Engineers 3rd Edition by A.D. Waite
An Introduction to Underwater Acoustics: Principles and Applications by Xavier Lurton
Principles of Underwater Sound 3rd Edition by Robert J. Urick

## Key Simplifications/Limitations

**Homogenous Volume of Water:** the simulator assumes the medium consists of identical oceanographic properties and has no capability to interact with a sound speed profile. This was done out of simplicity for now. I would like to add this functionality eventually.

**Mills Cross Intersection:** an algebraic transformation matrix is utilized in lieu of calculating and rendering an intersection of 3D meshes for some semblance of tool speed (it is already exceptionally slow).The tool instead creates an orthogonal local coordinate system where the TX vector is the X-axis, the cross-product of the TX and RX is the Z-axis, and the cross-product of Z and X forms the Y-axis. The intersection is solved in the local frame and then transformed back to global space.
This would need to be reworked to accommodate a complex seafloor, but notes about that implementation can be seen in point 4 below.

**Acoustic Footprint Area:** The tool calculates 3 distinct areas (1. elliptical intersection of TX and RX beamwidths, 2. Area ensonified by the pulse width at nadir, and 3. Expanding annulus of acoustic energy as the pulse travels outward across a sloped surface) and takes the minimum of the 3 to determine the actual ensonified area. A key simplification occurs where the footprint is drawn as a flat ellipse projected onto the 2.5D seafloor. 
This is an approximation which is again made to make the tool a tiny bit faster, as there wasn’t a noticeable benefit in computing it rigorously on a flat seafloor. If I were to add complex topography, however, this would absolutely need to be revised.

**Flat Seafloor:** I have attempted to develop a version of the tool that does properly model dynamic and complex topography, but performance slows down by at least 5x in a tool that is already exceedingly unresponsive. I am certain that a programmer with more skill than I could optimize this into a solution that works smoothly, but I was unable to make a version that had any semblance of interactivity via the web-based deployment of Streamlit.

**Array Directivity Factor:** I have implemented a rather crude “hard baffle”, where if the z-component of a calculated vector is above the horizontal plane of the transducer, the amplitude is multiplied by 0.001 to effectively reduce it to zero. This simplification is not true to the actual acoustics, but I felt it was confusing to model lobes above the horizontal for the purposes of this tool. I would absolutely welcome suggestions for how to make this better!

**Amplitude Detection:** Amplitude detection in this script looks for the peak energy return. To do so, it calculates the total time of arrival variance by summing variances of speckle-induced noise, ambient noise, and the baseline pulse envelope length. Footprint length is approximated simply as twice the semi-minor axis of the projected ellipse. This is again a simplification made given the flat seafloor of the tool. If a complex seafloor was added, this would need to be reworked as well.

**Phase Detection:** Simulated split aperture array length is set to be exactly ½ the length of the receiver array. It would probably be better to simulate some degree of overlap, but I wasn’t sure how to arrive at a justifiable value there. Input on how to better implement this would be greatly appreciated!

**Detection Method Binary:** The tool employs a simple binary selection of amplitude or phase detection based on whichever method achieves the higher QF value. I’m not sure if there is a better way to implement this, but I was not able to confirm how manufacturers arrive at their detection methods. It is probably a bit more involved than what I have implemented.

**Target Strength:** Volumetric scattering is not modeled in this tool nor acoustic impedance mismatches between water and the seafloor type. The tool utilizes a simple set of parameters for BS(Lambert), BS(Specular), and 10log10(A) to arrive at the target strength value.

**Ambient Noise:** Noise is assumed to be completely isotropic, so has no directivity component. I didn’t see any need to model specific directivity for things like prop or engine noise, as it seemed well beyond the scope of this tool.

**Absorption:**  While Francois-Garrison is used, I removed the option to adjust pH, and hardcoded it to be 8.0. This was going to introduce a lot of headaches with the bidirectional fields for oceanography, with very little (if any) impact on performance in the range allowed by this tool (1 kHz - 1Mhz).

**Matched Filter:** The current implementation of the matched filter assumes a 100% efficiency in the correlation.
If FM pulse is selected (Time-Bandwidth product > 1.5)m the script add the following to the sonar equation (per A.D. Waite, 1996):
PG = 10log10(tau*BW)

**Motion and Alignment:** The IMU is assumed to produce perfect results and that it is already corrected for latency. Additionally, the sonar array is assumed to have lever arm offsets perfectly accounted for and that it is perfectly compensated for angular offsets unless the user introduces mounting errors.
