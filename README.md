# Data

Development of a sophisticated urban water masking algorithm.

```
WI = ((i * ((SWIR2 - NIR) / (SWIR2 + NIR))) +
      (j*((green - SWIR2) / (green + SWIR2))) +
      (k * ((green - NIR) / (green + NIR)))) + (l * SAR)
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
- Canada 
