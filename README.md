# PayneSpeckRemoval

```swift
let rawDataFile = "nir-test-data.bin"
let (before, after) = payneSpeckRemoval(rawDataFile, 400, 640)
outputNirDataAsGrayscaleImage(before, rawDataFile + "-before.png", .normalizeMinMax)
outputNirDataAsGrayscaleImage(after, rawDataFile + "-after.png", .normalizeMinMax)
```

```java
String rawDataFile = "nir-test-data.bin";
Pair<Mat, Mat> beforeAfter = PayneSpeckRemoval.payneSpeckRemoval(rawDataFile, 400, 640);
PayneSpeckRemoval.outputNirDataAsGrayscaleImage(beforeAfter.first, rawDataFile + "-before.png", NormalizeMinMax);
PayneSpeckRemoval.outputNirDataAsGrayscaleImage(beforeAfter.second, rawDataFile + "-after.png", NormalizeMinMax);
```
