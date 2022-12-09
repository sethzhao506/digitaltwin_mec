# Readme

Download `Train.csv` from: https://drive.google.com/file/d/1pgm970Lvat83tiGi6zwz30031yhj_RuH/view?usp=sharing
Download `Test.csv` from: https://drive.google.com/file/d/1QfguEz5EEsPNrcpDMOr4NU5h-KGbe6BI/view?usp=sharing

They should be in this form: {"img":..., "points":..., "class":...}, where `img` is a [Nx47x47x3] vector, `points` is a [Nx300x3] vector, and `class` is a [N] vector, where values range from 0 to 7.

How to load:
```
data = np.load("./Data/Train.npy", allow_pickle=True)
for key in data.item():
    print(key, data.item()[key].shape)
```