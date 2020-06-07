# Data

Development of a sophisticated urban water masking algorithm.

```
# Parameter adjustments 
i = input("i: ")
j = input("i: ")
k = input("i: ")
l = input("i: ")

# Apply Water separation index
WI = ((i * ((SWIR2 - NIR) / (SWIR2 + NIR))) +
      (j*((green - SWIR2) / (green + SWIR2))) +
      (k * ((green - NIR) / (green + NIR)))) + 
      (l * SAR)
      
# Interpolate      
WI[WI > 0] = 1
WI[WI < 0] = 0
```


## Areas of interest 

- Shanghai 
- New York 
- Dhaka 
- Rotterdam
- Osaka
- Buenos Aires

### Additional Data
- Canada (Rural lakes)




