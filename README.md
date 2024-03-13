# Physics-Inspired Machine Learning For Orbit Determination in Low-Earth-Orbit
This research aims to utilize physically informed machine learning techniques to improve **orbit determination** and **model atmospheric density**, considering the physical laws that govern the system.
This repository includes the **code** developed, the **dissertation document** and a **paper** ([arXiv](https://arxiv.org/abs/2311.10012)) concerning the first part of the dissertation (SINDy) presented in the [International Astronautical Congress 2023 @ Baku](https://dl.iafastro.directory/event/IAC-2023/paper/76700/) alongside its [interactive presentation](https://iac2023-iaf.ipostersessions.com/default.aspx?s=2C-89-37-69-FB-A5-38-0C-95-37-6A-05-85-A4-90-1D&guestview=true). This paper was also presented in [Lisbon Young Mathematicians Conference](https://sites.uab.pt/lymc2023/).

# Abstract
We have always been told since we were little that space is infinite. Having this in mind,
it would not make much sense to be so cautious and aware of the space that lies above us.
However, the area right above the Earth’s surface up to 2000 km is heavily contaminated
with space debris which can have all kinds of origins and dimensions both man-made
(inactive satellites, parts of rockets, minuscule flecks of paint) as well as from natural
sources (small meteoroids). Considering that satellites have their propellant carefully
measured to fulfill the planned trajectory and cannot afford evasion maneuvers at the
slightest danger signal, it is important to quantify the uncertainty on the predictions
made.


To predict when two objects will collide, one will need to model their orbits with
the goal of knowing their positions. Among the multiple elements involved, such as the
gravity potential or the shape of the object, space weather is the most difficult to predict.
Because of these stochastic variables, the early discoveries from multiple scientists in
the eighteenth century were only enough to describe an orbit in the perfect case scenario.
These variables make the modeling of a real orbit more challenging since they are random
and have to be considered when modeling them since their effects are not negligible. One
of the variables that has the most impact on calculating the orbit of a space object is
atmospheric density. 

Since we are dealing with a physical system that abides by physical
laws, even if not perfectly, this will be used to our advantage to improve the predictions
made. As aforementioned, these laws known for centuries can be too tailored for the
perfect-case scenario and new equations can be discovered to better model a real-case
scenario. The objective of this research is to employ physically informed machine learning
techniques for orbit determination as well as to model atmospheric density by leveraging
physical domain knowledge and improving upon the standard approach.

**Keywords:** Physically-Informed Neural Networks, Data-driven physical discovery, Space
Debris, Orbital Mechanics, Orbit determination

<p align="center">
       <img src="https://i.imgur.com/6b5FZvY.png" width="1200" height="400" alt="Layout of the website">
        <br>
        <em>Architecture of the PINNs</em>
</p>

This project was done in the context of my dissertation in partnership with [FCT-NOVA](https://www.fct.unl.pt/) & [Neuraspace](https://www.neuraspace.com/).
