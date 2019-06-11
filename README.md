# Splitting with Near-Circulant Linear Systems: Applications to Total Variation CT and PET

This code implements the experiments for [`reference`](https://arxiv.org/abs/1810.13100).

> [1] E. K. Ryu, S. Ko, J.-H. Won, "Splitting with Near-Circulant Linear Systems: Applications to Total Variation CT and PET," arXiv:1810.13100, 2019.
Date:  June 14, 2019

#### Authors
- [**Ernest K. Ryu**](http://www.math.ucla.edu/~eryu/)
- [**Seyoon Ko**](https://kose-y.github.io/)
- [**Joong-Ho Won**](https://sites.google.com/site/johannwon/)

#### Requires MIRT by Fessler Lab
http://web.eecs.umich.edu/~fessler/irt/fessler.tgz

#### Code
- par_beam.m
- fan_beam.m	
- cone_beam_temp.m
- PET.m

#### Notes 
- cone_beam.m does not run on Windows and does not utilize GPUs as the ct_geom of MIRT does not support Windows or GPUs.
- We cannot relase the real patient data used for par_beam.m. You can experiment instead with the Shepp-Logan phantom.
