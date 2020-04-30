# Splitting with Near-Circulant Linear Systems: Applications to Total Variation CT and PET

This code implements the experiments for [`reference`](https://epubs.siam.org/doi/abs/10.1137/18M1224003) (click [`here`](https://arxiv.org/abs/1810.13100) for Arxiv preprint).

> [1] E. K. Ryu, S. Ko, J.-H. Won (2020). "Splitting with Near-Circulant Linear Systems: Applications to Total Variation CT and PET," _SIAM Journal of Scientific Computing_, 42(1), B185-B206.
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
