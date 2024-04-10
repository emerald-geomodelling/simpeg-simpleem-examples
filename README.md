# experimental-simpeg-ext

SimpleEM is a simple wrapper around SimPEG for EM inversion of data where the instrument setup does not change between soundings.

This repository contains example notebooks showcasing how to use this for synthetic data and measured data delivered in xyz format.


```
import libaarhusxyz
import SimPEG.electromagnetics.utils.static_instrument.single
```

```
xyz = libaarhusxyz.XYZ("outputs/em1dtm_stitched_data.xyz")
xyz.plot_line(0)
```

![image](https://user-images.githubusercontent.com/104229/231162050-c354e1f6-e0e4-4e93-aef6-44e20add3ed0.png)

```
class MySystem(SimPEG.electromagnetics.utils.static_instrument.single.SingleMomentTEMXYZSystem):
    area=340
    i_max=1
```

```
inv = MySystem(xyz, n_layer=30)
xyzsparse, xyzl2 = inv.invert()

xyzsparse.plot_line(0, cmap="jet")
xyzl2.plot_line(0, cmap="jet")
```
![image](https://user-images.githubusercontent.com/104229/231162720-b6071eb1-a01f-4fbc-9239-f5fe858fe3c8.png)
