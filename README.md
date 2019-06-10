# Splitting with Near-Circulant Linear Systems: Applications to Total Variation CT and PET

This code comes jointly with the following [`reference`](https://arxiv.org/abs/1810.13100).

> [1] E. K. Ryu, S. Ko, J.-H. Won, "Splitting with Near-Circulant Linear Systems: Applications to Total Variation CT and PET," arXiv:1810.13100, 2019.
Date:  June 14, 2019

#### Authors
- [**Ernest Ryu**](http://www.math.ucla.edu/~eryu/)
- [**Seyoon Kop**](https://kose-y.github.io/)
- [**Joong-Ho Won**](https://sites.google.com/site/johannwon/)


#### Requires MIRT by Fessler Lab. Download from
http://web.eecs.umich.edu/~fessler/irt/fessler.tgz

#### Code
- (01) par_beam.m
- (02) fan_beam.m	
- (03) cone_beam_temp.m
- (04) PET.m


#### Notes 
- cone_beam.m does not run on Windows and does not utilize GPUs as the XXX function of MIRT toolbox only supports CPU code on Linux and MacOS.
- We do not real patient data used for par_beam.m. Rather we provide the option to replace it with the Shepp-Logan phantom.
